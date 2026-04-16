import io
import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from training.preprocessing import build_preprocessor, engineer_features, ALL_FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GCS_BUCKET      = "hdb-resale-artifacts"
DATA_BLOB       = "training/hdb_resale_2017_2026.csv"
LOCAL_ARTIFACTS = Path("artifacts/")
LOCAL_ARTIFACTS.mkdir(exist_ok=True)

TARGET = "log_resale_price"


def load_data() -> pd.DataFrame:
    logger.info(f"Loading data from gs://{GCS_BUCKET}/{DATA_BLOB}")
    gcs = storage.Client()
    csv_bytes = gcs.bucket(GCS_BUCKET).blob(DATA_BLOB).download_as_bytes()
    df = pd.read_csv(io.BytesIO(csv_bytes))
    return engineer_features(df)


def split_data(df: pd.DataFrame):
    train = df[df["transaction_year"] <= 2024]
    val   = df[(df["transaction_year"] == 2025) & (df["transaction_month"] <= 9)]
    test  = df[df["transaction_year"] >= 2026]
    logger.info(f"Split — train: {len(train)}, val: {len(val)}, test: {len(test)}")
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
    for fname in ["model.pkl", "preprocessor.pkl", "metrics.json"]:
        bucket.blob(f"models/{fname}").upload_from_filename(str(LOCAL_ARTIFACTS / fname))
        logger.info(f"Uploaded {fname} → gs://{GCS_BUCKET}/models/{fname}")


def main():
    df = load_data()
    train_df, val_df, test_df = split_data(df)

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(train_df[ALL_FEATURES], train_df[TARGET])
    X_val   = preprocessor.transform(val_df[ALL_FEATURES])
    X_test  = preprocessor.transform(test_df[ALL_FEATURES])
    y_train, y_val, y_test = train_df[TARGET], val_df[TARGET], test_df[TARGET]

    model = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.03, num_leaves=127,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )

    metrics = {
        "trained_at":  datetime.utcnow().isoformat(),
        "validation":  evaluate("Validation", model, X_val,  y_val),
        "test":        evaluate("Test",        model, X_test, y_test),
    }

    joblib.dump(model,        LOCAL_ARTIFACTS / "model.pkl")
    joblib.dump(preprocessor, LOCAL_ARTIFACTS / "preprocessor.pkl")
    json.dump(metrics,        open(LOCAL_ARTIFACTS / "metrics.json", "w"), indent=2)

    gcs = storage.Client()
    upload_artifacts(gcs)
    logger.info("Training complete ✅")


if __name__ == "__main__":
    main()
