"""
hpo_catboost_v2.py — Optuna hyperparameter search for CatBoost v2 (MAE objective).

Design:
  - Rolling forward validation: each fold uses train = [MIN_YEAR, val_year), val = val_year.
  - Primary metric: MAE on log_resale_price — same scale as CatBoost loss_function/eval_metric MAE (Option A).
  - Each trial scores median over folds of (median over random seeds of val MAE on log target). Optuna minimizes that.

Final production training (train_catboost_v2.py) uses train 2020–2024, val 2025, test 2026.
Rolling HPO folds use only years before the final val year (see DEFAULT_FOLD_VAL_YEARS).

Usage:
    python hpo_catboost_v2.py --trials 80
    python hpo_catboost_v2.py --trials 20 --fold-years 2021 2022 2023 2024 --seeds 42 43

Study resume:
    study = joblib.load("artifacts/hpo_catboost_v2_study.pkl")

If you previously ran HPO when the objective was MAE in SGD (expm1), delete or
rename that study pickle before resuming — trial values are not comparable.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).parent))
from train_catboost_v2 import (  # noqa: E402
    DATA_PATH,
    RPI_PATH,
    TARGET,
    TRAIN_YEAR_START,
    VAL_YEAR,
    CAT_FEATURES,
    DROP_COLS,
    LOCAL_ARTIFACTS,
    MLFLOW_URI,
    EXPERIMENT_NAME,
    add_macro_interaction_features,
    add_official_rpi,
    compute_spatial_features,
    engineer_features,
    prepare_X,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

STUDY_PATH = LOCAL_ARTIFACTS / "hpo_catboost_v2_study.pkl"

# HPO training cap; early stopping trims.
HPO_ITERATIONS = 3000
EARLY_STOP = 100

# Align with train_catboost_v2: rolling val years must be < VAL_YEAR (2025).
MIN_TRAIN_YEAR = TRAIN_YEAR_START
DEFAULT_FOLD_VAL_YEARS = tuple(y for y in (2021, 2022, 2023, 2024) if y < VAL_YEAR)
DEFAULT_HPO_SEEDS = (42, 142, 242)

MIN_ROWS_TRAIN = 3_000
MIN_ROWS_VAL = 400

# Populated by prepare_rolling_folds(): val_year -> (train_pool, val_pool, y_val_log Series)
_FOLD_POOLS: dict[int, tuple[Pool, Pool, pd.Series]] = {}


def _log_target_mae(y_log: pd.Series | np.ndarray, pred_log: np.ndarray) -> float:
    yv = np.asarray(y_log, dtype=np.float64).ravel()
    pv = np.asarray(pred_log, dtype=np.float64).ravel()
    return float(mean_absolute_error(yv, pv))


def prepare_rolling_folds(fold_val_years: tuple[int, ...]) -> None:
    """Build CatBoost pools per forward-validation year (features + spatial, no leakage)."""
    global _FOLD_POOLS
    _FOLD_POOLS.clear()

    logger.info("Loading %s", DATA_PATH)
    raw = pd.read_csv(DATA_PATH, low_memory=False)

    for val_year in fold_val_years:
        raw_train = raw[(raw["Tranc_Year"] >= MIN_TRAIN_YEAR) & (raw["Tranc_Year"] < val_year)]
        raw_val = raw[raw["Tranc_Year"] == val_year]
        if len(raw_train) < MIN_ROWS_TRAIN or len(raw_val) < MIN_ROWS_VAL:
            logger.warning(
                "Skip fold val_year=%s (train=%s val=%s)",
                val_year,
                len(raw_train),
                len(raw_val),
            )
            continue

        mall_med = raw_train["Mall_Nearest_Distance"].median()
        train_df = engineer_features(raw_train, mall_med)
        val_df = engineer_features(raw_val, mall_med)
        train_df = add_official_rpi(train_df, RPI_PATH)
        val_df = add_official_rpi(val_df, RPI_PATH)
        train_df = add_macro_interaction_features(train_df)
        val_df = add_macro_interaction_features(val_df)

        empty_te = train_df.iloc[:0].copy()
        tr_sp, va_sp, _ = compute_spatial_features(train_df, val_df, empty_te)
        for feat, vals in tr_sp.items():
            train_df[feat] = vals
        for feat, vals in va_sp.items():
            val_df[feat] = vals

        feature_cols = [c for c in train_df.columns if c not in DROP_COLS]
        cat_cols = [c for c in CAT_FEATURES if c in feature_cols]

        X_tr = prepare_X(train_df, feature_cols, cat_cols)
        X_va = prepare_X(val_df, feature_cols, cat_cols)
        y_tr = train_df[TARGET]
        y_va = val_df[TARGET]

        tr_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
        va_pool = Pool(X_va, y_va, cat_features=cat_cols)
        _FOLD_POOLS[val_year] = (tr_pool, va_pool, y_va)
        logger.info(
            "Fold val_year=%s | train rows=%s | val rows=%s",
            val_year,
            len(X_tr),
            len(X_va),
        )

    if not _FOLD_POOLS:
        raise RuntimeError("No valid rolling folds — check years and MIN_ROWS_* thresholds")


def make_objective(seeds: tuple[int, ...]):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 100.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
        }

        fold_aggregate_scores: list[float] = []

        for val_year, (tr_pool, va_pool, y_va) in sorted(_FOLD_POOLS.items()):
            seed_maes: list[float] = []
            for seed in seeds:
                model = CatBoostRegressor(
                    iterations=HPO_ITERATIONS,
                    loss_function="MAE",
                    eval_metric="MAE",
                    early_stopping_rounds=EARLY_STOP,
                    random_seed=seed,
                    thread_count=-1,
                    verbose=0,
                    **params,
                )
                model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
                pred_log = model.predict(va_pool)
                seed_maes.append(_log_target_mae(y_va, pred_log))

            med_seed = float(np.median(seed_maes))
            fold_aggregate_scores.append(med_seed)
            trial.set_user_attr(f"fold_{val_year}_medseed_mae_log", med_seed)

        score = float(np.median(fold_aggregate_scores))
        trial.set_user_attr("median_fold_median_seed_mae_log", score)
        trial.set_user_attr("fold_years_used", tuple(sorted(_FOLD_POOLS.keys())))
        return score

    return objective


def _mlflow_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    mlflow.log_metrics(
        {
            "trial_val_med_mae_log": trial.value,
            "best_med_mae_log_so_far": study.best_value,
        },
        step=trial.number,
    )


def main(
    n_trials: int,
    fold_val_years: tuple[int, ...],
    seeds: tuple[int, ...],
    study_seed: int,
) -> dict:
    LOCAL_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    prepare_rolling_folds(fold_val_years)

    if STUDY_PATH.exists():
        study = joblib.load(STUDY_PATH)
        n_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        logger.info("Resuming study — %s completed trials", n_done)
    else:
        study = optuna.create_study(
            direction="minimize",
            study_name="catboost_v2_hpo_mae_rolling",
            sampler=optuna.samplers.TPESampler(seed=study_seed),
        )

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    t0 = time.perf_counter()
    folds_str = ",".join(str(y) for y in sorted(_FOLD_POOLS.keys()))
    seeds_str = ",".join(str(s) for s in seeds)

    with mlflow.start_run(run_name="catboost_v2_hpo_mae_rolling_medseed"):
        mlflow.set_tags({
            "model":           "catboost",
            "phase":           "hpo",
            "optimizer":       "optuna_tpe",
            "objective":       "median_fold_median_seed_val_MAE_log_target",
            "rolling_folds":   folds_str,
            "hpo_seeds":       seeds_str,
        })
        mlflow.log_params({
            "n_trials":          n_trials,
            "hpo_iterations":   HPO_ITERATIONS,
            "early_stopping":   EARLY_STOP,
            "loss_function":    "MAE_log_target",
            "eval_metric":      "MAE_log_target",
            "study_sampler_seed": study_seed,
            "fold_val_years":   folds_str,
        })

        study.optimize(
            make_objective(seeds),
            n_trials=n_trials,
            callbacks=[_mlflow_callback],
            show_progress_bar=True,
        )

        elapsed = time.perf_counter() - t0
        best_trial = study.best_trial
        best = study.best_params

        mlflow.log_metrics({
            "best_median_fold_med_seed_MAE_log": study.best_value,
            "t_hpo_total_s": elapsed,
            "t_per_trial_s_approx": elapsed / max(n_trials, 1),
        })
        mlflow.log_params({f"best_{k}": v for k, v in best.items()})

    joblib.dump(study, STUDY_PATH)
    logger.info("Study saved → %s", STUDY_PATH)

    logger.info("\n%s", "=" * 60)
    logger.info(
        "HPO complete — metric = median over folds of median-over-seeds val MAE on log_resale_price"
    )
    logger.info("Folds (val years): %s", sorted(_FOLD_POOLS.keys()))
    logger.info("Seeds per fold: %s", seeds)
    logger.info(
        "Best trial #%s | objective MAE(log) %.6f",
        best_trial.number,
        study.best_value,
    )
    logger.info("\nPaste into train_catboost_v2.py:")
    for k, v in best.items():
        fmt = f"{v:.6g}" if isinstance(v, float) else str(v)
        logger.info("  %-24s = %s,", k, fmt)
    logger.info("%s\n", "=" * 60)

    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rolling CV + multi-seed median MAE HPO for CatBoost v2")
    parser.add_argument("--trials", type=int, default=100, help="Optuna trials")
    parser.add_argument(
        "--fold-years",
        type=int,
        nargs="+",
        default=list(DEFAULT_FOLD_VAL_YEARS),
        help="Each year becomes the forward-validation holdout year",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_HPO_SEEDS),
        help="Random seeds per fold; trial fold score = median of seed MAEs",
    )
    parser.add_argument(
        "--study-seed",
        type=int,
        default=42,
        help="Optuna TPESampler seed (search reproducibility)",
    )
    args = parser.parse_args()
    main(
        n_trials=args.trials,
        fold_val_years=tuple(args.fold_years),
        seeds=tuple(args.seeds),
        study_seed=args.study_seed,
    )
