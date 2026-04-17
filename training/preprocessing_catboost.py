"""
preprocessing_catboost.py — feature definitions for the CatBoost pipeline.

CatBoost handles all categorical columns natively (ordered target statistics),
so there is no ColumnTransformer. Only base feature engineering and comp
computation are needed — both imported from preprocessing_v2.
"""

import numpy as np
import pandas as pd

# Re-use the proven feature engineering + comp computation from v2
from preprocessing_v2 import (
    engineer_features,
    compute_comp_features,
    build_comp_lookup,
    get_comp_features_inference,
)

__all__ = [
    "ALL_FEATURES", "CAT_FEATURES", "NUMERIC_FEATURES", "TARGET",
    "engineer_features", "compute_comp_features",
    "build_comp_lookup", "get_comp_features_inference",
    "prepare_X",
]

TARGET = "log_resale_price"

NUMERIC_FEATURES = [
    "floor_area_sqm", "storey_midpoint", "remaining_lease_years",
    "transaction_year", "month_sin", "month_cos",
    # 6-month comp features
    "comp_count", "comp_median_price", "comp_median_ppqm",
    "comp_p25_price", "comp_p75_price",
]

# CatBoost encodes these internally — no manual encoding required.
# Includes all three location levels so the model can learn the hierarchy.
CAT_FEATURES = [
    "flat_type", "town", "flat_model",
    "street_name", "block", "block_street",
]

ALL_FEATURES = NUMERIC_FEATURES + CAT_FEATURES


def prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with the correct column types for CatBoost.
    Categorical columns must be str (not NaN); numeric NaNs are left as-is
    (CatBoost handles them natively).
    """
    X = df[ALL_FEATURES].copy()
    for col in CAT_FEATURES:
        X[col] = X[col].astype(str).str.strip()
    return X
