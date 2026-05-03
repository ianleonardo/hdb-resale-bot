"""
hpo_v4.py — Hyperparameter optimisation for CatBoost v4 (ARIMA + CatBoost).

Design
------
ARIMA fitting is expensive (~30-60 s per bundle).  Rather than refitting inside
each trial, ARIMA bundles are pre-computed once per rolling fold before Optuna
starts.  Trials then only run CatBoost — keeping per-trial cost identical to v2.

Rolling forward CV folds
  fold 2022: train 2020-2021 | val 2022   (ARIMA fit on 2017-2021)
  fold 2023: train 2020-2022 | val 2023   (ARIMA fit on 2017-2022)
  fold 2024: train 2020-2023 | val 2024   (ARIMA fit on 2017-2023)

Objective: median over folds of (median over seeds of val log-MAE).
           Identical structure to hpo_catboost_v2.py for fair comparison.

Usage
-----
  python hpo_v4.py                 # 80 trials, default folds + seeds
  python hpo_v4.py --trials 40     # fewer trials
  python hpo_v4.py --fresh         # ignore saved study, start fresh
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

_REPO_ROOT   = Path(__file__).resolve().parents[2]
_BACKEND_DIR = _REPO_ROOT / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
sys.path.insert(0, str(Path(__file__).parent))

from app.inference_features import (  # noqa: E402
    add_macro_interaction_features,
    add_official_rpi,
    build_spatial_bundle_dict,
    compute_spatial_features,
    engineer_features,
    prepare_X,
)
from arima_v4 import ARIMABundle, ARIMA_FEATURES  # noqa: E402
from features_v4 import (  # noqa: E402
    CAT_FEATURES,
    DROP_COLS,
    HIST_YEAR_START,
    TRAIN_YEAR_START,
    VAL_YEAR,
    get_feature_cols,
)
from train_v4 import DATA_PATH, LOCAL_ARTIFACTS, MLFLOW_URI, EXPERIMENT_NAME, RPI_PATH, TARGET  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

STUDY_PATH = LOCAL_ARTIFACTS / "hpo_v4_study.pkl"

HPO_ITERATIONS = 3000
EARLY_STOP     = 100

DEFAULT_FOLD_VAL_YEARS = tuple(y for y in (2022, 2023, 2024) if y < VAL_YEAR)
DEFAULT_HPO_SEEDS      = (42, 142, 242)

MIN_ROWS_TRAIN = 3_000
MIN_ROWS_VAL   = 400

# Populated by prepare_rolling_folds(); val_year → (tr_pool, va_pool, y_va)
_FOLD_POOLS: dict[int, tuple[Pool, Pool, pd.Series]] = {}


# ── Fold construction ─────────────────────────────────────────────────────────

def _build_one_fold(
    raw: pd.DataFrame,
    rpi_df: pd.DataFrame,
    val_year: int,
) -> tuple[Pool, Pool, pd.Series] | None:
    """
    Build CatBoost pools for one rolling fold, including pre-fitted ARIMA features.
    ARIMA is fit on HIST_YEAR_START..(val_year-1) — no val-period data in the bundle.
    """
    raw_train = raw[
        (raw["Tranc_Year"] >= TRAIN_YEAR_START) & (raw["Tranc_Year"] < val_year)
    ].reset_index(drop=True)
    raw_val   = raw[raw["Tranc_Year"] == val_year].reset_index(drop=True)

    if len(raw_train) < MIN_ROWS_TRAIN or len(raw_val) < MIN_ROWS_VAL:
        logger.warning(
            "Skip fold val=%d (train=%d val=%d)", val_year, len(raw_train), len(raw_val)
        )
        return None

    mall_med  = raw_train["Mall_Nearest_Distance"].median()

    def _fe(df):
        df = engineer_features(df, mall_med)
        df = add_official_rpi(df, RPI_PATH)
        return add_macro_interaction_features(df)

    train_df = _fe(raw_train)
    val_df   = _fe(raw_val)

    # Spatial
    empty = train_df.iloc[:0].copy()
    tr_sp, va_sp, _ = compute_spatial_features(train_df, val_df, empty)
    for feat, vals in tr_sp.items():
        train_df[feat] = vals
    for feat, vals in va_sp.items():
        val_df[feat] = vals

    # ARIMA — fitted on historical data strictly before val_year
    raw_arima = raw[
        (raw["Tranc_Year"] >= HIST_YEAR_START) & (raw["Tranc_Year"] < val_year)
    ].reset_index(drop=True)
    bundle = ARIMABundle().fit(raw_arima, rpi_df)

    arima_tr = bundle.get_arima_features(train_df)
    arima_va = bundle.get_arima_features(val_df)
    for name, arr in arima_tr.items():
        train_df[name] = arr
    for name, arr in arima_va.items():
        val_df[name] = arr

    feature_cols, cat_cols = get_feature_cols(list(train_df.columns))
    X_tr = prepare_X(train_df, feature_cols, cat_cols)
    X_va = prepare_X(val_df,   feature_cols, cat_cols)
    y_tr = train_df[TARGET]
    y_va = val_df[TARGET]

    tr_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
    va_pool = Pool(X_va, y_va, cat_features=cat_cols)

    logger.info(
        "Fold val=%d | train=%d | val=%d | features=%d (incl. %d ARIMA)",
        val_year, len(X_tr), len(X_va), len(feature_cols), len(ARIMA_FEATURES),
    )
    return tr_pool, va_pool, y_va


def prepare_rolling_folds(fold_val_years: tuple[int, ...]) -> None:
    global _FOLD_POOLS
    _FOLD_POOLS.clear()

    logger.info("Loading %s", DATA_PATH)
    raw    = pd.read_csv(DATA_PATH, low_memory=False)
    rpi_df = pd.read_csv(RPI_PATH)

    for val_year in fold_val_years:
        logger.info("Building fold val=%d (includes ARIMA fitting)…", val_year)
        t0     = time.perf_counter()
        result = _build_one_fold(raw, rpi_df, val_year)
        logger.info("  Fold val=%d ready in %.1fs", val_year, time.perf_counter() - t0)
        if result is not None:
            _FOLD_POOLS[val_year] = result

    if not _FOLD_POOLS:
        raise RuntimeError("No valid rolling folds — check years and MIN_ROWS_* thresholds")


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_objective(seeds: tuple[int, ...]):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "depth":               trial.suggest_int("depth", 4, 10),
            "learning_rate":       trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 1.0, 100.0, log=True),
            "random_strength":     trial.suggest_float("random_strength", 0.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
            "border_count":        trial.suggest_int("border_count", 32, 255),
            "min_data_in_leaf":    trial.suggest_int("min_data_in_leaf", 1, 50),
        }

        fold_scores: list[float] = []
        for val_year, (tr_pool, va_pool, y_va) in sorted(_FOLD_POOLS.items()):
            seed_maes: list[float] = []
            for seed in seeds:
                m = CatBoostRegressor(
                    iterations=HPO_ITERATIONS,
                    loss_function="MAE",
                    eval_metric="MAE",
                    early_stopping_rounds=EARLY_STOP,
                    random_seed=seed,
                    thread_count=-1,
                    verbose=0,
                    **params,
                )
                m.fit(tr_pool, eval_set=va_pool, use_best_model=True)
                pred_log = m.predict(va_pool)
                seed_maes.append(float(mean_absolute_error(np.asarray(y_va), pred_log)))
            fold_scores.append(float(np.median(seed_maes)))
            trial.set_user_attr(f"fold_{val_year}_medseed_mae_log", fold_scores[-1])

        score = float(np.median(fold_scores))
        trial.set_user_attr("median_fold_median_seed_mae_log", score)
        return score

    return objective


def _mlflow_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    mlflow.log_metrics(
        {"trial_val_med_mae_log": trial.value, "best_so_far": study.best_value},
        step=trial.number,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main(
    n_trials: int,
    fold_val_years: tuple[int, ...],
    seeds: tuple[int, ...],
    study_seed: int,
    fresh: bool,
) -> dict:
    LOCAL_ARTIFACTS.mkdir(parents=True, exist_ok=True)

    logger.info("Pre-computing %d rolling folds (ARIMA + features)…", len(fold_val_years))
    t_folds = time.perf_counter()
    prepare_rolling_folds(fold_val_years)
    logger.info("All folds ready in %.1fs — starting Optuna", time.perf_counter() - t_folds)

    if not fresh and STUDY_PATH.exists():
        study = joblib.load(STUDY_PATH)
        n_done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        logger.info("Resuming study — %d completed trials", n_done)
    else:
        study = optuna.create_study(
            direction="minimize",
            study_name="catboost_arima_v4_hpo",
            sampler=optuna.samplers.TPESampler(seed=study_seed),
        )
        logger.info("Created new study")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    folds_str = ",".join(str(y) for y in sorted(_FOLD_POOLS.keys()))

    t0 = time.perf_counter()
    with mlflow.start_run(run_name="catboost_arima_v4_hpo"):
        mlflow.set_tags({"model": "catboost-arima-v4", "phase": "hpo"})
        mlflow.log_params({
            "n_trials":     n_trials,
            "fold_years":   folds_str,
            "seeds":        str(seeds),
            "arima_features": str(ARIMA_FEATURES),
        })

        study.optimize(
            make_objective(seeds),
            n_trials=n_trials,
            callbacks=[_mlflow_cb],
            show_progress_bar=True,
        )

        elapsed = time.perf_counter() - t0
        best    = study.best_params
        mlflow.log_metrics({
            "best_median_fold_med_seed_MAE_log": study.best_value,
            "t_hpo_total_s": elapsed,
        })
        mlflow.log_params({f"best_{k}": v for k, v in best.items()})

    joblib.dump(study, STUDY_PATH)
    logger.info("Study saved → %s", STUDY_PATH)

    logger.info("\n%s", "=" * 60)
    logger.info("HPO complete — best trial #%d | MAE(log)=%.6f", study.best_trial.number, study.best_value)
    logger.info("\nPaste into train_v4.py:")
    for k, v in best.items():
        fmt = f"{v:.6g}" if isinstance(v, float) else str(v)
        logger.info("  %-24s = %s,", k.upper(), fmt)
    logger.info("%s\n", "=" * 60)
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",      type=int,   default=80)
    parser.add_argument(
        "--fold-years", type=int, nargs="+",
        default=list(DEFAULT_FOLD_VAL_YEARS),
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        default=list(DEFAULT_HPO_SEEDS),
    )
    parser.add_argument("--study-seed",  type=int,   default=42)
    parser.add_argument("--fresh",       action="store_true")
    args = parser.parse_args()
    main(
        n_trials=args.trials,
        fold_val_years=tuple(args.fold_years),
        seeds=tuple(args.seeds),
        study_seed=args.study_seed,
        fresh=args.fresh,
    )
