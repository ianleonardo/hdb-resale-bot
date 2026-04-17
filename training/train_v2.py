"""
train_v2.py — training pipeline with 6-month recent-comps features.

Key difference from train.py:
  1. compute_comp_features() runs on the FULL dataset before the split so that
     val/test rows can see comps from earlier training data (no leakage).
  2. Saves comp_lookup.pkl alongside model.pkl for use at inference time.
"""

import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing_v2 import (
    ALL_FEATURES, TARGET, build_preprocessor,
    engineer_features, compute_comp_features, build_comp_lookup,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GCS_BUCKET      = "hdb-resale-artifacts"
DATA_BLOB       = "training/Resaleflatpricesbase.csv"
LOCAL_DATA      = Path("../data")
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
    Time-based split — comp features must already be present in df.
      Train : 2017–2024  (~197k rows, 86%)
      Val   : 2025 Jan–Sep (~20k rows,  9%) — early stopping only
      Test  : 2025 Oct–2026 (~11k rows,  5%) — fully held-out evaluation
    """
    train = df[(df["transaction_year"] >= 2017) & (df["transaction_year"] <= 2024)]
    val   = df[(df["transaction_year"] == 2025) & (df["transaction_month"] <= 9)]
    test  = df[
        ((df["transaction_year"] == 2025) & (df["transaction_month"] >= 10)) |
        (df["transaction_year"] == 2026)
    ]
    logger.info(f"Split — train: {len(train)} | val: {len(val)} | test: {len(test)}")
    return train, val, test


def evaluate(name: str, model, X, y_log: pd.Series) -> dict:
    pred = np.expm1(model.predict(X))
    true = np.expm1(y_log)
    mae  = mean_absolute_error(true, pred)
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    mape = float(np.mean(np.abs((true - pred) / true)) * 100)
    r2   = r2_score(true, pred)
    logger.info(f"{name}: MAE={mae:,.0f} | RMSE={rmse:,.0f} | MAPE={mape:.2f}% | R²={r2:.4f}")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def upload_artifacts(gcs: storage.Client):
    bucket = gcs.bucket(GCS_BUCKET)
    for fname in ["model_v2.pkl", "preprocessor_v2.pkl", "comp_lookup.pkl", "metrics_v2.json"]:
        path = LOCAL_ARTIFACTS / fname
        if path.exists():
            bucket.blob(f"models/{fname}").upload_from_filename(str(path))
            logger.info(f"Uploaded gs://{GCS_BUCKET}/models/{fname}")


def main():
    raw = load_data()

    # Feature engineering then comp computation on FULL dataset before split
    logger.info("Engineering base features...")
    df = engineer_features(raw)

    logger.info("Computing comp features on full dataset (leakage-safe)...")
    df = compute_comp_features(df)
    comp_coverage = (df["comp_count"] > 0).mean()
    logger.info(f"Comp coverage: {comp_coverage:.1%} of all rows have at least one comp")

    train_df, val_df, test_df = split_data(df)

    # Save comp lookup built from training data only (used at inference)
    comp_lookup = build_comp_lookup(train_df)
    joblib.dump(comp_lookup, LOCAL_ARTIFACTS / "comp_lookup.pkl")
    logger.info(f"Saved comp_lookup with {len(comp_lookup)} streets")

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(train_df[ALL_FEATURES], train_df[TARGET])
    X_val   = preprocessor.transform(val_df[ALL_FEATURES])
    X_test  = preprocessor.transform(test_df[ALL_FEATURES])
    y_train, y_val, y_test = train_df[TARGET], val_df[TARGET], test_df[TARGET]

    model = lgb.LGBMRegressor(
        n_estimators=5000, metric="mae",
        # HPO best params from hpo.py run (trial 85) — re-tune with hpo_v2.py for best results
        num_leaves=181, learning_rate=0.050, min_child_samples=82,
        subsample=0.738, colsample_bytree=0.537,
        reg_alpha=0.031, reg_lambda=0.021,
        random_state=42, n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(500)],
    )

    metrics = {
        "trained_at":  datetime.now(timezone.utc).isoformat(),
        "model":       "v2-with-comps",
        "validation":  evaluate("Validation", model, X_val,  y_val),
        "test":        evaluate("Test",        model, X_test, y_test),
        "comp_coverage_pct": round(comp_coverage * 100, 1),
    }

    joblib.dump(model,        LOCAL_ARTIFACTS / "model_v2.pkl")
    joblib.dump(preprocessor, LOCAL_ARTIFACTS / "preprocessor_v2.pkl")
    json.dump(metrics, open(LOCAL_ARTIFACTS / "metrics_v2.json", "w"), indent=2)

    gcs = storage.Client()
    upload_artifacts(gcs)
    logger.info("Training complete ✅")


if __name__ == "__main__":
    main()
