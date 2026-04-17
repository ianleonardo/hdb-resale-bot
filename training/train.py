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

from preprocessing import build_preprocessor, engineer_features, ALL_FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GCS_BUCKET      = "hdb-resale-artifacts"
DATA_BLOB       = "training/Resaleflatpricesbase.csv"
LOCAL_DATA      = Path("../data")
LOCAL_ARTIFACTS = Path("artifacts/")
LOCAL_ARTIFACTS.mkdir(exist_ok=True)

TARGET = "resale_price"


def load_data() -> pd.DataFrame:
    logger.info(f"Loading data from gs://{GCS_BUCKET}/{DATA_BLOB}")
    gcs = storage.Client()
    csv_bytes = gcs.bucket(GCS_BUCKET).blob(DATA_BLOB).download_as_bytes()
    df = pd.read_csv(io.BytesIO(csv_bytes))
    logger.info(f"Loaded {len(df)} rows | date range: {df['month'].min()} – {df['month'].max()}")
    return engineer_features(df)


def split_data(df: pd.DataFrame):
    """
    Time-based split to prevent data leakage:
      Train : 2017-01 – 2024-12  (~85%)
      Val   : 2025-01 – 2025-09  (~10%)
      Test  : 2025-10 – 2026-03  (~5%)
    """
    train = df[(df["transaction_year"] >= 2017) & (df["transaction_year"] <= 2024)]
    val   = df[(df["transaction_year"] == 2025) & (df["transaction_month"] <= 9)]
    test  = df[
        ((df["transaction_year"] == 2025) & (df["transaction_month"] >= 10)) |
        ((df["transaction_year"] == 2026) & (df["transaction_month"] <= 3))
    ]
    logger.info(f"Split — train: {len(train)} | val: {len(val)} | test: {len(test)}")
    return train, val, test


def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    """Save split datasets to local data/ directory and upload to GCS."""
    splits = {"train": train, "val": val, "test": test}

    # Save locally
    LOCAL_DATA.mkdir(exist_ok=True)
    for name, df in splits.items():
        path = LOCAL_DATA / f"{name}.csv"
        df.to_csv(path, index=False)
        logger.info(f"Saved {path} ({len(df)} rows)")

    # Upload to GCS
    gcs = storage.Client()
    bucket = gcs.bucket(GCS_BUCKET)
    for name, df in splits.items():
        blob_path = f"training/splits/{name}.csv"
        bucket.blob(blob_path).upload_from_string(
            df.to_csv(index=False), content_type="text/csv"
        )
        logger.info(f"Uploaded gs://{GCS_BUCKET}/{blob_path}")


def evaluate(name: str, model, X, y: pd.Series) -> dict:
    pred = model.predict(X)
    mae  = mean_absolute_error(y, pred)
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    mape = float(np.mean(np.abs((y - pred) / y)) * 100)
    r2   = r2_score(y, pred)
    logger.info(f"{name}: MAE={mae:,.0f} | RMSE={rmse:,.0f} | MAPE={mape:.2f}% | R²={r2:.4f}")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def upload_artifacts(gcs: storage.Client):
    bucket = gcs.bucket(GCS_BUCKET)
    for fname in ["model.pkl", "preprocessor.pkl", "metrics.json"]:
        bucket.blob(f"models/{fname}").upload_from_filename(str(LOCAL_ARTIFACTS / fname))
        logger.info(f"Uploaded gs://{GCS_BUCKET}/models/{fname}")


def main():
    df = load_data()
    train_df, val_df, test_df = split_data(df)
    save_splits(train_df, val_df, test_df)

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(train_df[ALL_FEATURES], train_df[TARGET])
    X_val   = preprocessor.transform(val_df[ALL_FEATURES])
    X_test  = preprocessor.transform(test_df[ALL_FEATURES])
    y_train, y_val, y_test = train_df[TARGET], val_df[TARGET], test_df[TARGET]

    model = lgb.LGBMRegressor(
        n_estimators=5000, objective="regression_l1", metric="mae",
        # HPO best params (trial 85/100, val MAE=46,103 — re-tune after objective change)
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
        "validation":  evaluate("Validation", model, X_val,  y_val),
        "test":        evaluate("Test",        model, X_test, y_test),
    }

    joblib.dump(model,        LOCAL_ARTIFACTS / "model.pkl")
    joblib.dump(preprocessor, LOCAL_ARTIFACTS / "preprocessor.pkl")
    json.dump(metrics, open(LOCAL_ARTIFACTS / "metrics.json", "w"), indent=2)

    gcs = storage.Client()
    upload_artifacts(gcs)
    logger.info("Training complete ✅")


if __name__ == "__main__":
    main()
