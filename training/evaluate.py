import io
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.cloud import storage
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import engineer_features, ALL_FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GCS_BUCKET = "hdb-resale-artifacts"


def load_artifacts_from_gcs():
    gcs = storage.Client()
    bucket = gcs.bucket(GCS_BUCKET)

    def load_pkl(blob_path: str):
        buf = io.BytesIO()
        bucket.blob(blob_path).download_to_file(buf)
        buf.seek(0)
        return joblib.load(buf)

    return load_pkl("models/model.pkl"), load_pkl("models/preprocessor.pkl")


def compute_metrics(y_true_log: pd.Series, y_pred_log: np.ndarray) -> dict:
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return {
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "R2":   r2_score(y_true, y_pred),
    }


def plot_residuals(y_true_log: pd.Series, y_pred_log: np.ndarray, title: str = "Residuals"):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=5)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted Price (SGD)")
    axes[0].set_ylabel("Residual (SGD)")
    axes[0].set_title(f"{title} — Residual Plot")

    axes[1].hist(residuals, bins=50, edgecolor="black")
    axes[1].set_xlabel("Residual (SGD)")
    axes[1].set_title(f"{title} — Residual Distribution")

    plt.tight_layout()
    return fig


def run_evaluation(data_path: str):
    """Run full evaluation on a CSV file. Expects raw HDB resale columns."""
    df = engineer_features(pd.read_csv(data_path))
    model, preprocessor = load_artifacts_from_gcs()

    X = preprocessor.transform(df[ALL_FEATURES])
    y_pred_log = model.predict(X)
    metrics = compute_metrics(df["log_resale_price"], y_pred_log)

    logger.info(
        f"MAE={metrics['MAE']:,.0f} | RMSE={metrics['RMSE']:,.0f} | "
        f"MAPE={metrics['MAPE']:.2f}% | R²={metrics['R2']:.4f}"
    )
    return metrics


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/hdb_resale_after_2024.csv"
    run_evaluation(path)
