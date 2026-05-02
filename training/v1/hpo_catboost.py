"""
hpo_catboost.py — Optuna hyperparameter search for the CatBoost model.

Paste best_params into train_catboost.py after the run completes.
"""

import io
import logging

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor, Pool
from google.cloud import storage
from sklearn.metrics import mean_absolute_error

from preprocessing_catboost import (
    ALL_FEATURES, CAT_FEATURES, TARGET,
    engineer_features, compute_comp_features, prepare_X,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GCS_BUCKET = "hdb-resale-artifacts"
DATA_BLOB  = "training/Resaleflatpricesbase.csv"

# Module-level — populated once by prepare_data()
train_pool = val_pool = y_val = None


def prepare_data():
    global train_pool, val_pool, y_val

    gcs = storage.Client()
    csv_bytes = gcs.bucket(GCS_BUCKET).blob(DATA_BLOB).download_as_bytes()
    raw = pd.read_csv(io.BytesIO(csv_bytes))

    logger.info("Engineering features + computing comps on full dataset...")
    df = engineer_features(raw)
    df = compute_comp_features(df)

    train = df[(df["transaction_year"] >= 2017) & (df["transaction_year"] <= 2024)]
    val   = df[(df["transaction_year"] == 2025) & (df["transaction_month"] <= 9)]

    X_train, y_train = prepare_X(train), train[TARGET]
    X_val,   y_val_s = prepare_X(val),   val[TARGET]

    train_pool = Pool(X_train, y_train, cat_features=CAT_FEATURES)
    val_pool   = Pool(X_val,   y_val_s, cat_features=CAT_FEATURES)
    y_val      = y_val_s
    logger.info(f"Data ready — train: {len(train)}, val: {len(val)}")


def objective(trial: optuna.Trial) -> float:
    params = {
        "depth":              trial.suggest_int("depth", 4, 10),
        "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg":        trial.suggest_float("l2_leaf_reg", 1.0, 100.0, log=True),
        "random_strength":    trial.suggest_float("random_strength", 0.0, 10.0),
        "bagging_temperature":trial.suggest_float("bagging_temperature", 0.0, 2.0),
        "border_count":       trial.suggest_int("border_count", 32, 255),
    }
    model = CatBoostRegressor(
        iterations=5000,
        loss_function="MAE",
        eval_metric="MAE",
        early_stopping_rounds=100,
        random_seed=42,
        thread_count=-1,
        verbose=0,
        **params,
    )
    model.fit(train_pool, eval_set=val_pool)
    return mean_absolute_error(np.expm1(y_val), np.expm1(model.predict(val_pool)))


def main(n_trials: int = 100):
    prepare_data()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    logger.info(f"Best MAE: {study.best_value:,.0f}")
    logger.info(f"Best params: {study.best_params}")
    joblib.dump(study, "artifacts/hpo_catboost_study.pkl")
    return study.best_params


if __name__ == "__main__":
    main()
