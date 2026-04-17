"""
train_catboost.py — CatBoost training pipeline with recent-comps features.

Key differences from train_v2.py (LightGBM):
  - No ColumnTransformer / sklearn preprocessor — CatBoost encodes categoricals natively.
  - loss_function='MAE' on log-price directly aligns training loss with the evaluation metric.
  - Saves model_catboost.cbm (CatBoost native format) + comp_lookup.pkl.
"""

import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from google.cloud import storage
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing_catboost import (
    ALL_FEATURES, CAT_FEATURES, TARGET,
    engineer_features, compute_comp_features, build_comp_lookup, prepare_X,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GCS_BUCKET      = "hdb-resale-artifacts"
DATA_BLOB       = "training/Resaleflatpricesbase.csv"
LOCAL_ARTIFACTS = Path("artifacts/")
LOCAL_ARTIFACTS.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    logger.info(f"Loading data from gs://{GCS_BUCKET}/{DATA_BLOB}")
    gcs = storage.Client()
    csv_bytes = gcs.bucket(GCS_BUCKET).blob(DATA_BLOB).download_as_bytes()
    df = pd.read_csv(io.BytesIO(csv_bytes))
    logger.info(f"Loaded {len(df)} rows | date range: {df['month'].min()} – {df['month'].max()}")
    return df


def split_data(df: pd.DataFrame):
    """
    Train : 2017–2024  (~197k rows, 86%)
    Val   : 2025 Jan–Sep (~20k rows,  9%) — early stopping only
    Test  : 2025 Oct–2026  (~11k rows,  5%) — fully held-out
    """
    train = df[(df["transaction_year"] >= 2017) & (df["transaction_year"] <= 2024)]
    val   = df[(df["transaction_year"] == 2025) & (df["transaction_month"] <= 9)]
    test  = df[
        ((df["transaction_year"] == 2025) & (df["transaction_month"] >= 10)) |
        (df["transaction_year"] == 2026)
    ]
    logger.info(f"Split — train: {len(train)} | val: {len(val)} | test: {len(test)}")
    return train, val, test


def evaluate(name: str, model: CatBoostRegressor, pool: Pool, y_log: pd.Series) -> dict:
    pred = np.expm1(model.predict(pool))
    true = np.expm1(y_log)
    mae  = mean_absolute_error(true, pred)
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    mape = float(np.mean(np.abs((true - pred) / true)) * 100)
    r2   = r2_score(true, pred)
    logger.info(f"{name}: MAE={mae:,.0f} | RMSE={rmse:,.0f} | MAPE={mape:.2f}% | R²={r2:.4f}")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def upload_artifacts(gcs: storage.Client):
    bucket = gcs.bucket(GCS_BUCKET)
    for fname in ["model_catboost.cbm", "comp_lookup.pkl", "metrics_catboost.json"]:
        path = LOCAL_ARTIFACTS / fname
        if path.exists():
            bucket.blob(f"models/{fname}").upload_from_filename(str(path))
            logger.info(f"Uploaded gs://{GCS_BUCKET}/models/{fname}")


def main():
    raw = load_data()

    logger.info("Engineering base features...")
    df = engineer_features(raw)

    logger.info("Computing comp features on full dataset (leakage-safe)...")
    df = compute_comp_features(df)
    comp_coverage = (df["comp_count"] > 0).mean()
    logger.info(f"Comp coverage: {comp_coverage:.1%} of all rows have at least one comp")

    train_df, val_df, test_df = split_data(df)

    comp_lookup = build_comp_lookup(train_df)
    joblib.dump(comp_lookup, LOCAL_ARTIFACTS / "comp_lookup.pkl")
    logger.info(f"Saved comp_lookup with {len(comp_lookup)} streets")

    X_train = prepare_X(train_df)
    X_val   = prepare_X(val_df)
    X_test  = prepare_X(test_df)
    y_train, y_val, y_test = train_df[TARGET], val_df[TARGET], test_df[TARGET]

    train_pool = Pool(X_train, y_train, cat_features=CAT_FEATURES)
    val_pool   = Pool(X_val,   y_val,   cat_features=CAT_FEATURES)
    test_pool  = Pool(X_test,  y_test,  cat_features=CAT_FEATURES)

    model = CatBoostRegressor(
        iterations=5000,
        loss_function="MAE",          # directly optimises MAE on log-price
        eval_metric="MAE",
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3.0,
        random_strength=1.0,
        bagging_temperature=1.0,
        border_count=128,
        early_stopping_rounds=200,
        random_seed=42,
        thread_count=-1,
        verbose=500,
    )
    model.fit(train_pool, eval_set=val_pool)

    metrics = {
        "trained_at":        datetime.now(timezone.utc).isoformat(),
        "model":             "catboost-with-comps",
        "validation":        evaluate("Validation", model, val_pool,  y_val),
        "test":              evaluate("Test",        model, test_pool, y_test),
        "comp_coverage_pct": round(comp_coverage * 100, 1),
    }

    model.save_model(str(LOCAL_ARTIFACTS / "model_catboost.cbm"))
    json.dump(metrics, open(LOCAL_ARTIFACTS / "metrics_catboost.json", "w"), indent=2)

    gcs = storage.Client()
    upload_artifacts(gcs)
    logger.info("Training complete ✅")


if __name__ == "__main__":
    main()
