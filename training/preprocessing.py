import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from category_encoders import TargetEncoder

ORDINAL_FLAT_TYPE = [
    "1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM",
    "5 ROOM", "EXECUTIVE", "MULTI-GENERATION",
]

NUMERIC_FEATURES = [
    "floor_area_sqm", "storey_midpoint", "remaining_lease_years",
    "lease_commence_date", "transaction_year", "transaction_month",
]
ORDINAL_FEATURES = ["flat_type"]
NOMINAL_FEATURES = ["town", "flat_model"]
HIGH_CARD        = ["street_name", "block"]

ALL_FEATURES = NUMERIC_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES + HIGH_CARD


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(),                                  NUMERIC_FEATURES),
        ("ord", OrdinalEncoder(categories=[ORDINAL_FLAT_TYPE]),    ORDINAL_FEATURES),
        ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), NOMINAL_FEATURES),
        ("hc",  TargetEncoder(smoothing=10),                       HIGH_CARD),
    ])


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["transaction_year"]  = pd.to_datetime(df["month"]).dt.year
    df["transaction_month"] = pd.to_datetime(df["month"]).dt.month
    df["storey_midpoint"]   = (
        df["storey_range"]
        .str.extract(r"(\d+) TO (\d+)")
        .astype(float)
        .mean(axis=1)
    )
    years  = df["remaining_lease"].str.extract(r"(\d+)\s*year").fillna(0).astype(float)[0]
    months = df["remaining_lease"].str.extract(r"(\d+)\s*month").fillna(0).astype(float)[0]
    df["remaining_lease_years"] = years + months / 12
    df["log_resale_price"]      = np.log1p(df["resale_price"])
    return df
