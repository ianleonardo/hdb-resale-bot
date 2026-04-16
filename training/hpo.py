import io
import logging

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from google.cloud import storage
from sklearn.metrics import mean_absolute_error

from training.preprocessing import build_preprocessor, engineer_features, ALL_FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GCS_BUCKET = "hdb-resale-artifacts"
TARGET     = "log_resale_price"

# Populated by prepare_data()
X_train = X_val = y_train = y_val = None


def prepare_data():
    global X_train, X_val, y_train, y_val
    gcs = storage.Client()
    csv_bytes = gcs.bucket(GCS_BUCKET).blob("training/hdb_resale_2017_2026.csv").download_as_bytes()
    df = engineer_features(pd.read_csv(io.BytesIO(csv_bytes)))

    train = df[df["transaction_year"] <= 2024]
    val   = df[(df["transaction_year"] == 2025) & (df["transaction_month"] <= 9)]

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(train[ALL_FEATURES], train[TARGET])
    X_val   = preprocessor.transform(val[ALL_FEATURES])
    y_train, y_val = train[TARGET], val[TARGET]
    logger.info(f"Data ready — train: {len(train)}, val: {len(val)}")


def objective(trial: optuna.Trial) -> float:
    params = {
        "num_leaves":        trial.suggest_int("num_leaves", 31, 255),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }
    model = lgb.LGBMRegressor(n_estimators=500, **params, random_state=42, n_jobs=-1)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    return mean_absolute_error(np.expm1(y_val), np.expm1(model.predict(X_val)))


def main(n_trials: int = 100):
    prepare_data()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    logger.info(f"Best MAE: {study.best_value:,.0f}")
    logger.info(f"Best params: {study.best_params}")
    joblib.dump(study, "artifacts/hpo_study.pkl")
    return study.best_params


if __name__ == "__main__":
    main()
