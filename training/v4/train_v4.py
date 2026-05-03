"""
train_v4.py — HDB resale price training pipeline: ARIMA market baseline + CatBoost (v4).

Strategy
--------
  1. ARIMA fits on 2017-2024 monthly log-price series: a **global** model, optional
     **per-(town, flat_type)** models (fallback to global when sparse), and an
     **RPI** series model inside the bundle for inference hooks.  The two CatBoost
     inputs from this bundle are **relative/static** only (see ``ARIMA_FEATURES`` in
     ``arima_v4.py``): never raw resale price levels.

  2. CatBoost is trained on **2020-2024** with the shared v2-style engineered +
     spatial + RPI features, plus **two** ARIMA columns
     (``arima_seg_vs_global``, ``arima_seg_series_std``).  It blends macro drift /
     segment positioning from ARIMA with unit-specific signals (floor, area,
     lease, amenities, location).

  3. At inference the backend loads ``ARIMABundle`` and calls
     ``get_arima_features(query_df)`` for those columns, then CatBoost predicts
     **log-price** and ``expm1`` yields SGD.

Leakage controls
----------------
  • ARIMA trained on 2017-2024 raw transactions only (not val / test).
  • KDTree spatial encoding: past-only trees for training rows; full training tree
    for val / test (unchanged from v2).
  • CatBoost early stopping on val MAE (2025).

Artifacts saved to training/v4/artifacts/
  model_v4.cbm              CatBoost model
  arima_bundle_v4.pkl       Pickled ``ARIMABundle`` (statsmodels + pandas)
  metrics_v4.json           Metrics + feature list + hyperparameters
  feature_importance_v4.csv
  spatial_inference.pkl     KDTree bundle (same format as v2, used by backend)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

_REPO_ROOT   = Path(__file__).resolve().parents[2]
_BACKEND_DIR = _REPO_ROOT / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
sys.path.insert(0, str(Path(__file__).parent))

from app.inference_features import (  # noqa: E402
    SPATIAL_FEATS,
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
    TRAIN_YEAR_END,
    VAL_YEAR,
    TEST_YEAR_START,
    TEST_YEAR_END,
    get_feature_cols,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH       = _REPO_ROOT / "data" / "hdb_resale_complete.csv"
RPI_PATH        = _REPO_ROOT / "data" / "hdb_rpi.csv"
LOCAL_ARTIFACTS = Path(__file__).parent / "artifacts"
GCS_BUCKET      = os.environ.get("GCS_BUCKET", "hdb-resale-artifacts")
MLFLOW_URI      = "http://127.0.0.1:5005/"
EXPERIMENT_NAME = "HDB Resale Telegram Bot"
MODEL_VERSION   = "catboost-arima-v4"
TARGET          = "log_resale_price"
RANDOM_SEED     = 42

# ── CatBoost hyperparameters (tuned by hpo_v4.py, best trial #62) ────────────
DEPTH               = 4
LEARNING_RATE       = 0.208241
L2_LEAF_REG         = 1.66016
RANDOM_STRENGTH     = 0.873233
BAGGING_TEMPERATURE = 0.213883
BORDER_COUNT        = 204
MIN_DATA_IN_LEAF    = 5


# ── Data helpers ──────────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame):
    train = df[
        (df["Tranc_Year"] >= TRAIN_YEAR_START) & (df["Tranc_Year"] <= TRAIN_YEAR_END)
    ].reset_index(drop=True)
    val  = df[df["Tranc_Year"] == VAL_YEAR].reset_index(drop=True)
    test = df[
        (df["Tranc_Year"] >= TEST_YEAR_START) & (df["Tranc_Year"] <= TEST_YEAR_END)
    ].reset_index(drop=True)
    logger.info(
        "Split — train %d-%d: %d | val %d: %d | test %d-%d: %d",
        TRAIN_YEAR_START, TRAIN_YEAR_END, len(train),
        VAL_YEAR, len(val),
        TEST_YEAR_START, TEST_YEAR_END, len(test),
    )
    return train, val, test


def _mse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def _r2_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def _add_arima_cols(df: pd.DataFrame, bundle: ARIMABundle) -> pd.DataFrame:
    feats = bundle.get_arima_features(df)
    df = df.copy()
    for name, arr in feats.items():
        df[name] = arr
    return df


def evaluate(name: str, model: CatBoostRegressor, pool: Pool, y_log: pd.Series) -> dict:
    pred_log = np.asarray(model.predict(pool), dtype=np.float64)
    y_log_np = np.asarray(y_log.to_numpy(), dtype=np.float64)
    pred     = np.expm1(pred_log)
    true     = np.expm1(y_log_np)
    mae      = float(np.mean(np.abs(true - pred)))
    rmse     = float(np.sqrt(_mse_np(true, pred)))
    mape     = float(np.mean(np.abs((true - pred) / true)) * 100)
    r2       = float(_r2_np(true, pred))
    log_rmse = float(np.sqrt(_mse_np(y_log_np, pred_log)))
    logger.info(
        "%s: MAE=%s | RMSE=%s | MAPE=%.2f%% | R²=%.4f | log_RMSE=%.6f",
        name, f"{mae:,.0f}", f"{rmse:,.0f}", mape, r2, log_rmse,
    )
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2, "log_RMSE": log_rmse}


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    LOCAL_ARTIFACTS.mkdir(exist_ok=True)
    t_start = time.perf_counter()

    # ── 1. Load + split ───────────────────────────────────────────────────────
    logger.info("Loading %s", DATA_PATH)
    raw = pd.read_csv(DATA_PATH, low_memory=False)
    logger.info("Loaded %d rows", len(raw))
    raw_train, raw_val, raw_test = split_data(raw)

    mall_dist_median = raw_train["Mall_Nearest_Distance"].median()

    # ── 2. Property feature engineering ──────────────────────────────────────
    logger.info("Engineering features…")
    t0 = time.perf_counter()

    def _fe(df):
        df = engineer_features(df, mall_dist_median)
        df = add_official_rpi(df, RPI_PATH)
        return add_macro_interaction_features(df)

    train_df = _fe(raw_train)
    val_df   = _fe(raw_val)
    test_df  = _fe(raw_test)
    t_feat   = time.perf_counter() - t0
    logger.info("Feature engineering: %.2fs", t_feat)

    # ── 3. Spatial KDTree features ────────────────────────────────────────────
    logger.info("Building KDTree spatial features…")
    t0 = time.perf_counter()
    bundle_spatial  = build_spatial_bundle_dict(train_df)
    spatial_path    = LOCAL_ARTIFACTS / "spatial_inference.pkl"
    with open(spatial_path, "wb") as _sf:
        pickle.dump(bundle_spatial, _sf, protocol=pickle.HIGHEST_PROTOCOL)
    tr_sp, val_sp, te_sp = compute_spatial_features(train_df, val_df, test_df)
    for feat, vals in tr_sp.items():
        train_df[feat] = vals
    for feat, vals in val_sp.items():
        val_df[feat] = vals
    for feat, vals in te_sp.items():
        test_df[feat] = vals
    t_spatial = time.perf_counter() - t0
    logger.info("Spatial features: %.2fs | %s", t_spatial, SPATIAL_FEATS)

    # ── 4. Fit ARIMA bundle on 2017-2024 (historical window, no val/test) ─────
    logger.info("Fitting ARIMA bundle on %d-%d data…", HIST_YEAR_START, TRAIN_YEAR_END)
    t0 = time.perf_counter()
    rpi_df = pd.read_csv(RPI_PATH)
    raw_for_arima = raw[
        (raw["Tranc_Year"] >= HIST_YEAR_START) & (raw["Tranc_Year"] <= TRAIN_YEAR_END)
    ].reset_index(drop=True)
    arima_bundle = ARIMABundle().fit(raw_for_arima, rpi_df)
    arima_path   = LOCAL_ARTIFACTS / "arima_bundle_v4.pkl"
    arima_bundle.save(arima_path)
    t_arima = time.perf_counter() - t0
    logger.info("ARIMA fitting: %.2fs", t_arima)

    # ── 5. Inject ARIMA features ──────────────────────────────────────────────
    logger.info("Injecting ARIMA features (%s)…", ARIMA_FEATURES)
    train_df = _add_arima_cols(train_df, arima_bundle)
    val_df   = _add_arima_cols(val_df,   arima_bundle)
    test_df  = _add_arima_cols(test_df,  arima_bundle)

    # ── 6. Build CatBoost feature set ─────────────────────────────────────────
    feature_cols, cat_cols = get_feature_cols(list(train_df.columns))
    logger.info(
        "Total features: %d (v2 base + %d ARIMA) | Categorical: %d",
        len(feature_cols), len(ARIMA_FEATURES), len(cat_cols),
    )
    logger.info("Features: %s", feature_cols)

    X_train = prepare_X(train_df, feature_cols, cat_cols)
    X_val   = prepare_X(val_df,   feature_cols, cat_cols)
    X_test  = prepare_X(test_df,  feature_cols, cat_cols)
    y_train, y_val, y_test = train_df[TARGET], val_df[TARGET], test_df[TARGET]

    train_pool = Pool(X_train, y_train, cat_features=cat_cols)
    val_pool   = Pool(X_val,   y_val,   cat_features=cat_cols)
    test_pool  = Pool(X_test,  y_test,  cat_features=cat_cols)

    # ── 7. Train CatBoost ─────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="catboost_arima_v4"):
        mlflow.set_tags({
            "model":       "catboost",
            "version":     MODEL_VERSION,
            "arima":       "global+segment+rpi",
            "spatial_enc": "kdtree_temporal",
        })
        mlflow.log_params({
            "train_years":        f"{TRAIN_YEAR_START}-{TRAIN_YEAR_END}",
            "arima_hist_years":   f"{HIST_YEAR_START}-{TRAIN_YEAR_END}",
            "val_year":           VAL_YEAR,
            "test_years":         f"{TEST_YEAR_START}-{TEST_YEAR_END}",
            "n_train":            len(X_train),
            "n_val":              len(X_val),
            "n_test":             len(X_test),
            "n_features":         len(feature_cols),
            "n_arima_features":   len(ARIMA_FEATURES),
            "depth":              DEPTH,
            "learning_rate":      LEARNING_RATE,
            "l2_leaf_reg":        L2_LEAF_REG,
            "random_strength":    RANDOM_STRENGTH,
            "bagging_temperature": BAGGING_TEMPERATURE,
            "border_count":       BORDER_COUNT,
            "min_data_in_leaf":   MIN_DATA_IN_LEAF,
            "random_seed":        RANDOM_SEED,
        })
        mlflow.log_metrics({
            "t_feature_engineering_s": t_feat,
            "t_kdtree_s":              t_spatial,
            "t_arima_fitting_s":       t_arima,
        })

        model = CatBoostRegressor(
            iterations=3000,
            loss_function="MAE",
            eval_metric="MAE",
            early_stopping_rounds=100,
            random_seed=RANDOM_SEED,
            thread_count=1,
            verbose=500,
            depth=DEPTH,
            learning_rate=LEARNING_RATE,
            l2_leaf_reg=L2_LEAF_REG,
            random_strength=RANDOM_STRENGTH,
            bagging_temperature=BAGGING_TEMPERATURE,
            border_count=BORDER_COUNT,
            min_data_in_leaf=MIN_DATA_IN_LEAF,
        )
        logger.info("Training CatBoost v4 (MAE on log target)…")
        t0 = time.perf_counter()
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        t_train = time.perf_counter() - t0
        logger.info("Training done: %.2fs | best iter=%d", t_train, model.get_best_iteration())

        mlflow.log_metric("t_training_s",  t_train)
        mlflow.log_param("best_iteration", model.get_best_iteration())

        train_metrics = evaluate("Train",      model, train_pool, y_train)
        val_metrics   = evaluate("Validation", model, val_pool,   y_val)
        test_metrics  = evaluate("Test",       model, test_pool,  y_test)

        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"val_{k}":   v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}":  v for k, v in test_metrics.items()})

        # ── 8. Feature importance ─────────────────────────────────────────────
        fi_df = (
            pd.DataFrame({"feature": feature_cols, "importance": model.get_feature_importance()})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        logger.info("Top 15 features:")
        for _, row in fi_df.head(15).iterrows():
            marker = " ◄ ARIMA" if row["feature"] in ARIMA_FEATURES else ""
            logger.info("  %-45s %.4f%s", row["feature"], row["importance"], marker)

        fi_path = LOCAL_ARTIFACTS / "feature_importance_v4.csv"
        fi_df.to_csv(fi_path, index=False)

        # ── 9. Save artifacts ─────────────────────────────────────────────────
        model_path   = LOCAL_ARTIFACTS / "model_v4.cbm"
        metrics_path = LOCAL_ARTIFACTS / "metrics_v4.json"
        model.save_model(str(model_path))

        metrics_out = {
            "trained_at":    datetime.now(timezone.utc).isoformat(),
            "model":         MODEL_VERSION,
            "split": {
                "train_years":    [TRAIN_YEAR_START, TRAIN_YEAR_END],
                "arima_hist_years": [HIST_YEAR_START, TRAIN_YEAR_END],
                "val_year":       VAL_YEAR,
                "test_years":     [TEST_YEAR_START, TEST_YEAR_END],
                "n_train":        len(X_train),
                "n_val":          len(X_val),
                "n_test":         len(X_test),
            },
            "catboost_hparams": {
                "loss_function":     "MAE",
                "iterations":        3000,
                "depth":             DEPTH,
                "learning_rate":     LEARNING_RATE,
                "l2_leaf_reg":       L2_LEAF_REG,
                "random_strength":   RANDOM_STRENGTH,
                "bagging_temperature": BAGGING_TEMPERATURE,
                "border_count":      BORDER_COUNT,
                "min_data_in_leaf":  MIN_DATA_IN_LEAF,
                "random_seed":       RANDOM_SEED,
                "early_stopping":    100,
                "best_iteration":    int(model.get_best_iteration()),
            },
            "arima": {
                "n_segment_models": sum(
                    1 for m, _, _ in arima_bundle.segment_data.values() if m is not None
                ),
                "min_series_len":   18,
                "train_end_period": arima_bundle.train_end_period,
            },
            "train":      train_metrics,
            "validation": val_metrics,
            "test":       test_metrics,
            "n_features":  len(feature_cols),
            "features":    feature_cols,
            "arima_features": ARIMA_FEATURES,
            "cat_features":   cat_cols,
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics_out, f, indent=2)

        for p in [model_path, metrics_path, fi_path, arima_path, spatial_path]:
            mlflow.log_artifact(str(p))

        try:
            from google.cloud import storage  # noqa: PLC0415
            gcs    = storage.Client()
            bucket = gcs.bucket(GCS_BUCKET)
            for fname in ["model_v4.cbm", "arima_bundle_v4.pkl",
                          "metrics_v4.json", "spatial_inference.pkl"]:
                fp = LOCAL_ARTIFACTS / fname
                if fp.exists():
                    blob_name = f"models/{fname}"
                    bucket.blob(blob_name).upload_from_filename(str(fp))
                    logger.info("Uploaded gs://%s/%s", GCS_BUCKET, blob_name)
        except Exception as e:
            logger.warning("GCS upload skipped: %s", e)

        t_total = time.perf_counter() - t_start
        mlflow.log_metric("t_total_s", t_total)
        logger.info("Total pipeline: %.2fs", t_total)
        logger.info("Training complete ✅")


if __name__ == "__main__":
    main()
