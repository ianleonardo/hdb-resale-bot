"""
train_catboost_v2.py — CatBoost training pipeline for HDB Resale Price prediction.

Uses tuned CatBoost hyperparameters with MAE loss on log_resale_price (early stopping on MAE).
Feature engineering and KDTree spatial encoding follow hdb_ml_pipeline_v20.py.
High-cardinality location identifiers (block, street_name, postal) are replaced
by smoothed spatial radius features built from a KDTree on training coordinates.

Shared feature logic lives in backend/app/inference_features.py (sys.path bootstrap below).

Leakage controls:
  - split_data() is called on raw data before any statistics are fit
  - mall_dist_median is computed from train only, then applied to val/test
  - KDTree spatial encoding uses a per-month past-only tree for training rows;
    val/test rows query the full training tree (train years strictly before val/test)

Data split (option 1): train 2020–2024, val 2025, test 2026 — latest market year in train before forward val/test.

Macro interactions (option 3): after official HDB RPI join, adds rpi_x_year, rpi_x_tranc_period,
rpi_x_floor_area_sqm (shared via inference_features.add_macro_interaction_features).

Rows outside train/val/test years are excluded from modelling splits (logged).
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from google.cloud import storage
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = _REPO_ROOT / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.inference_features import (  # noqa: E402
    SPATIAL_FEATS,
    add_macro_interaction_features,
    add_official_rpi,
    build_spatial_bundle_dict,
    compute_spatial_features,
    engineer_features,
    prepare_X,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH       = _REPO_ROOT / "data" / "hdb_resale_complete.csv"
RPI_PATH        = _REPO_ROOT / "data" / "hdb_rpi.csv"
GCS_BUCKET      = "hdb-resale-artifacts"
LOCAL_ARTIFACTS = Path(__file__).parent / "artifacts"
MLFLOW_URI      = "http://127.0.0.1:5005/"
EXPERIMENT_NAME = "HDB Resale Telegram Bot"
MODEL_VERSION   = "catboost-v2"
TARGET          = "log_resale_price"
RANDOM_SEED     = 42

# Time-based splits (forward-chronological)
TRAIN_YEAR_START = 2020
TRAIN_YEAR_END   = 2024
VAL_YEAR         = 2025
TEST_YEAR_START  = 2026
TEST_YEAR_END    = 2026  # single held-out year; bump TEST_YEAR_END when adding future tests

# ── CatBoost hyperparameters (Optuna / manual tuned) ────────────────────────────
# HPO snapshot 2026-05-02
DEPTH                 = 5
LEARNING_RATE         = 0.294152
L2_LEAF_REG           = 12.2505
RANDOM_STRENGTH       = 0.183718
BAGGING_TEMPERATURE   = 0.986251
BORDER_COUNT          = 203
MIN_DATA_IN_LEAF      = 45

# ── Feature drop lists (following hdb_ml_pipeline_v20 DROP_COLS logic) ─────────
REDUNDANT_COLS = [
    "mrt_nearest_distance", "Mall_Nearest_Distance", "Hawker_Nearest_Distance",
    "bus_stop_nearest_distance", "pri_sch_nearest_distance", "sec_sch_nearest_dist",
]

LOW_IMP_COLS = [
    "1room_rental", "2room_rental", "3room_rental", "other_room_rental",
    "1room_sold", "studio_apartment_sold", "multigen_sold",
    "residential", "commercial", "market_hawker",
    "multistorey_carpark", "precinct_pavilion",
    "mrt_interchange", "bus_interchange",
    "Tranc_Month",
    "affiliation", "pri_sch_affiliation",
    "hawker_market_stalls", "hawker_food_stalls",
    "school_quality", "cutoff_point",
    "Latitude", "Longitude",   # used only for KDTree, not as direct features
]

LOCATION_ID_COLS = [
    "block", "street_name", "postal",
]

IDENTIFIER_COLS = [
    "resale_price",          # raw target
    "Tranc_YearMonth",       # redundant with Tranc_Year + tranc_period
    "storey_range",          # replaced by mid_storey
    "lease_commence_date",   # replaced by lease_remaining_years
    "bus_stop_name", "bus_stop_latitude", "bus_stop_longitude",
    "mrt_latitude", "mrt_longitude",
    "pri_sch_latitude", "pri_sch_longitude",
    "sec_sch_latitude", "sec_sch_longitude",
    "planning_area",         # entirely null in hdb_resale_complete.csv
]

DROP_COLS = set(
    [TARGET]
    + IDENTIFIER_COLS
    + LOCATION_ID_COLS
    + REDUNDANT_COLS
    + LOW_IMP_COLS
)

CAT_FEATURES = [
    "flat_type", "flat_model", "town",
    "mrt_name", "pri_sch_name", "sec_sch_name",
]


def split_data(df: pd.DataFrame):
    train = df[
        (df["Tranc_Year"] >= TRAIN_YEAR_START) & (df["Tranc_Year"] <= TRAIN_YEAR_END)
    ].reset_index(drop=True)
    val = df[df["Tranc_Year"] == VAL_YEAR].reset_index(drop=True)
    test = df[
        (df["Tranc_Year"] >= TEST_YEAR_START) & (df["Tranc_Year"] <= TEST_YEAR_END)
    ].reset_index(drop=True)
    used = len(train) + len(val) + len(test)
    if used < len(df):
        logger.info(
            "Excluded from splits (outside train/val/test years): %s rows",
            f"{len(df) - used:,}",
        )
    logger.info(
        "Split — train %s–%s: %s | val %s: %s | test %s–%s: %s",
        TRAIN_YEAR_START,
        TRAIN_YEAR_END,
        f"{len(train):,}",
        VAL_YEAR,
        f"{len(val):,}",
        TEST_YEAR_START,
        TEST_YEAR_END,
        f"{len(test):,}",
    )
    return train, val, test


def evaluate(name: str, model: CatBoostRegressor, pool: Pool, y_log: pd.Series) -> dict:
    pred_log = model.predict(pool)
    pred     = np.expm1(pred_log)
    true     = np.expm1(y_log.to_numpy())

    mae      = float(mean_absolute_error(true, pred))
    rmse     = float(np.sqrt(mean_squared_error(true, pred)))
    mape     = float(np.mean(np.abs((true - pred) / true)) * 100)
    r2       = float(r2_score(true, pred))
    log_rmse = float(np.sqrt(mean_squared_error(y_log.to_numpy(), pred_log)))

    logger.info(
        f"{name}: MAE={mae:,.0f} | RMSE={rmse:,.0f} | MAPE={mape:.2f}% "
        f"| R²={r2:.4f} | log_RMSE={log_rmse:.6f}"
    )
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2, "log_RMSE": log_rmse}


def upload_artifacts(gcs: storage.Client):
    bucket = gcs.bucket(GCS_BUCKET)
    for fname in [
        "model_catboost_v2.cbm",
        "metrics_catboost_v2.json",
        "feature_importance_v2.csv",
        "spatial_inference.pkl",
        "block_lookup.parquet",
    ]:
        path = LOCAL_ARTIFACTS / fname
        if path.exists():
            blob_name = f"models/{fname}"
            bucket.blob(blob_name).upload_from_filename(str(path))
            logger.info(f"Uploaded gs://{GCS_BUCKET}/{blob_name}")


def main():
    LOCAL_ARTIFACTS.mkdir(exist_ok=True)

    t_pipeline_start = time.perf_counter()

    logger.info(f"Loading data from {DATA_PATH}")
    raw = pd.read_csv(DATA_PATH, low_memory=False)
    logger.info(f"Loaded {len(raw):,} rows, {len(raw.columns)} columns")

    raw_train, raw_val, raw_test = split_data(raw)

    mall_dist_median = raw_train["Mall_Nearest_Distance"].median()

    logger.info("Engineering features (following hdb_ml_pipeline_v20)...")
    t0 = time.perf_counter()
    train_df = engineer_features(raw_train, mall_dist_median)
    val_df   = engineer_features(raw_val,   mall_dist_median)
    test_df  = engineer_features(raw_test,  mall_dist_median)
    t_feature_engineering_s = time.perf_counter() - t0
    logger.info(f"Feature engineering done in {t_feature_engineering_s:.2f}s")

    logger.info(f"Joining official HDB RPI from {RPI_PATH.name}...")
    t0 = time.perf_counter()
    train_df = add_official_rpi(train_df, RPI_PATH)
    val_df   = add_official_rpi(val_df, RPI_PATH)
    test_df  = add_official_rpi(test_df, RPI_PATH)
    train_df = add_macro_interaction_features(train_df)
    val_df   = add_macro_interaction_features(val_df)
    test_df  = add_macro_interaction_features(test_df)
    t_rpi_join_s = time.perf_counter() - t0
    logger.info(f"RPI join + macro interaction features done in {t_rpi_join_s:.2f}s")

    spatial_path = LOCAL_ARTIFACTS / "spatial_inference.pkl"
    bundle = build_spatial_bundle_dict(train_df)
    joblib.dump(bundle, spatial_path)
    logger.info(f"Saved spatial inference bundle -> {spatial_path}")

    logger.info(
        f"Computing KDTree spatial encodings "
        f"(radii={bundle['radii_m']}m, {len(train_df):,} training coords)..."
    )
    t0 = time.perf_counter()
    tr_spatial, val_spatial, te_spatial = compute_spatial_features(train_df, val_df, test_df)
    t_kdtree_s = time.perf_counter() - t0
    logger.info(f"KDTree spatial encoding done in {t_kdtree_s:.2f}s")

    for feat, vals in tr_spatial.items():
        train_df[feat] = vals
    for feat, vals in val_spatial.items():
        val_df[feat]   = vals
    for feat, vals in te_spatial.items():
        test_df[feat]  = vals
    logger.info(f"Spatial features added: {SPATIAL_FEATS}")

    feature_cols = [c for c in train_df.columns if c not in DROP_COLS]
    cat_cols     = [c for c in CAT_FEATURES if c in feature_cols]

    logger.info(f"Total features: {len(feature_cols)} | Categorical: {len(cat_cols)}")
    logger.info(f"Features: {feature_cols}")

    X_train = prepare_X(train_df, feature_cols, cat_cols)
    X_val   = prepare_X(val_df,   feature_cols, cat_cols)
    X_test  = prepare_X(test_df,  feature_cols, cat_cols)
    y_train, y_val, y_test = train_df[TARGET], val_df[TARGET], test_df[TARGET]

    train_pool = Pool(X_train, y_train, cat_features=cat_cols)
    val_pool   = Pool(X_val,   y_val,   cat_features=cat_cols)
    test_pool  = Pool(X_test,  y_test,  cat_features=cat_cols)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="catboost_v2_mae_kdtree_rsi"):
        mlflow.set_tags({
            "model":        "catboost",
            "version":      MODEL_VERSION,
            "loss":         "MAE",
            "target":       TARGET,
            "hyperparams":  "tuned_mae_log_optuna_rolling_medseed",
            "spatial_enc":  "kdtree_temporal_train_full_tree_valtest",
            "hdb_rpi":      "official_lag1q",
            "val_split":    str(VAL_YEAR),
            "test_split":   f"{TEST_YEAR_START}-{TEST_YEAR_END}",
            "data_source":  DATA_PATH.name,
        })
        mlflow.log_params({
            "train_year_start": TRAIN_YEAR_START,
            "train_year_end":   TRAIN_YEAR_END,
            "val_year":         VAL_YEAR,
            "test_year_start":  TEST_YEAR_START,
            "test_year_end":    TEST_YEAR_END,
            "n_train":          len(X_train),
            "n_val":            len(X_val),
            "n_test":           len(X_test),
            "n_features":       len(feature_cols),
            "n_cat_features":   len(cat_cols),
            "spatial_radii_m":  str(bundle["radii_m"]),
            "random_seed":      RANDOM_SEED,
            "loss_function":        "MAE",
            "early_stopping":       100,
            "depth":                DEPTH,
            "learning_rate":      LEARNING_RATE,
            "l2_leaf_reg":          L2_LEAF_REG,
            "random_strength":      RANDOM_STRENGTH,
            "bagging_temperature": BAGGING_TEMPERATURE,
            "border_count":       BORDER_COUNT,
            "min_data_in_leaf":   MIN_DATA_IN_LEAF,
        })
        mlflow.log_metrics({
            "t_feature_engineering_s": t_feature_engineering_s,
            "t_rpi_join_s":            t_rpi_join_s,
            "t_kdtree_s":              t_kdtree_s,
        })

        model = CatBoostRegressor(
            loss_function="MAE",
            eval_metric="MAE",
            early_stopping_rounds=100,
            random_seed=RANDOM_SEED,
            verbose=500,
            depth=DEPTH,
            learning_rate=LEARNING_RATE,
            l2_leaf_reg=L2_LEAF_REG,
            random_strength=RANDOM_STRENGTH,
            bagging_temperature=BAGGING_TEMPERATURE,
            border_count=BORDER_COUNT,
            min_data_in_leaf=MIN_DATA_IN_LEAF,
        )
        logger.info(
            "Training CatBoost (MAE loss on log target; depth=%s lr=%s)...",
            DEPTH,
            LEARNING_RATE,
        )
        t0 = time.perf_counter()
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        t_training_s = time.perf_counter() - t0
        logger.info(f"Training done in {t_training_s:.2f}s")

        best_iter = model.get_best_iteration()
        logger.info(f"Best iteration: {best_iter}")
        mlflow.log_param("best_iteration", best_iter)
        mlflow.log_metric("t_training_s", t_training_s)

        train_metrics = evaluate("Train",      model, train_pool, y_train)
        val_metrics   = evaluate("Validation", model, val_pool,   y_val)
        test_metrics  = evaluate("Test",       model, test_pool,  y_test)

        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"val_{k}":   v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}":  v for k, v in test_metrics.items()})

        fi_df = (
            pd.DataFrame({"feature": feature_cols, "importance": model.get_feature_importance()})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        logger.info("Top 20 features by importance:")
        for _, row in fi_df.head(20).iterrows():
            logger.info(f"  {row['feature']:<45} {row['importance']:.4f}")

        fi_path = LOCAL_ARTIFACTS / "feature_importance_v2.csv"
        fi_df.to_csv(fi_path, index=False)
        mlflow.log_artifact(str(fi_path))

        model_path   = LOCAL_ARTIFACTS / "model_catboost_v2.cbm"
        metrics_path = LOCAL_ARTIFACTS / "metrics_catboost_v2.json"
        model.save_model(str(model_path))

        metrics_out = {
            "trained_at":       datetime.now(timezone.utc).isoformat(),
            "model":            MODEL_VERSION,
            "split": {
                "train_years": [TRAIN_YEAR_START, TRAIN_YEAR_END],
                "val_year":    VAL_YEAR,
                "test_years":  [TEST_YEAR_START, TEST_YEAR_END],
                "n_train":     len(X_train),
                "n_val":       len(X_val),
                "n_test":      len(X_test),
            },
            "catboost_hparams": {
                "loss_function": "MAE",
                "eval_metric": "MAE",
                "depth": DEPTH,
                "learning_rate": LEARNING_RATE,
                "l2_leaf_reg": L2_LEAF_REG,
                "random_strength": RANDOM_STRENGTH,
                "bagging_temperature": BAGGING_TEMPERATURE,
                "border_count": BORDER_COUNT,
                "min_data_in_leaf": MIN_DATA_IN_LEAF,
                "random_seed": RANDOM_SEED,
                "early_stopping_rounds": 100,
            },
            "best_iteration":   best_iter,
            "train":            train_metrics,
            "validation":       val_metrics,
            "test":             test_metrics,
            "n_features":       len(feature_cols),
            "features":         feature_cols,
            "cat_features":     cat_cols,
            "spatial_feats":    list(SPATIAL_FEATS),
            "mall_dist_median": float(mall_dist_median),
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics_out, f, indent=2)

        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(spatial_path))

        try:
            gcs = storage.Client()
            upload_artifacts(gcs)
        except Exception as e:
            logger.warning(f"GCS upload skipped: {e}")

        t_total_s = time.perf_counter() - t_pipeline_start
        mlflow.log_metric("t_total_s", t_total_s)
        logger.info(f"Total pipeline time: {t_total_s:.2f}s")
        logger.info("Training complete ✅")


if __name__ == "__main__":
    main()
