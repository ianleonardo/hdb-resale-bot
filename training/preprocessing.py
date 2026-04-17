import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from category_encoders import TargetEncoder

ORDINAL_FLAT_TYPE = [
    "1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM",
    "5 ROOM", "EXECUTIVE", "MULTI-GENERATION",
]

# lease_commence_date dropped (collinear with remaining_lease_years)
# transaction_month replaced with cyclical sin/cos encoding
# block + street_name combined into block_street
NUMERIC_FEATURES  = [
    "floor_area_sqm", "storey_midpoint", "remaining_lease_years",
    "transaction_year", "month_sin", "month_cos",
]
ORDINAL_FEATURES  = ["flat_type"]
NOMINAL_FEATURES  = ["town", "flat_model"]
# All three location features kept: street_name and block capture coarser signals,
# block_street captures the exact building — higher smoothing prevents rare combos from overfitting
HIGH_CARD_STREET = ["street_name"]
HIGH_CARD_BLOCK  = ["block"]
HIGH_CARD_COMBO  = ["block_street"]

ALL_FEATURES = (
    NUMERIC_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES
    + HIGH_CARD_STREET + HIGH_CARD_BLOCK + HIGH_CARD_COMBO
)


def build_preprocessor() -> ColumnTransformer:
    # No StandardScaler — LightGBM is scale-invariant
    return ColumnTransformer(transformers=[
        ("num",       "passthrough",                                          NUMERIC_FEATURES),
        ("ord",       OrdinalEncoder(categories=[ORDINAL_FLAT_TYPE]),         ORDINAL_FEATURES),
        ("nom",       OneHotEncoder(handle_unknown="ignore", sparse_output=False), NOMINAL_FEATURES),
        ("hc_street", TargetEncoder(smoothing=10),                            HIGH_CARD_STREET),
        ("hc_block",  TargetEncoder(smoothing=10),                            HIGH_CARD_BLOCK),
        ("hc_combo",  TargetEncoder(smoothing=50),                            HIGH_CARD_COMBO),
    ])


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["transaction_year"]  = pd.to_datetime(df["month"]).dt.year
    df["transaction_month"] = pd.to_datetime(df["month"]).dt.month

    # Cyclical month encoding — captures Dec/Jan adjacency
    df["month_sin"] = np.sin(2 * np.pi * df["transaction_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["transaction_month"] / 12)

    df["storey_midpoint"] = (
        df["storey_range"]
        .str.extract(r"(\d+) TO (\d+)")
        .astype(float)
        .mean(axis=1)
    )

    years  = df["remaining_lease"].str.extract(r"(\d+)\s*year").fillna(0).astype(float)[0]
    months = df["remaining_lease"].str.extract(r"(\d+)\s*month").fillna(0).astype(float)[0]
    df["remaining_lease_years"] = years + months / 12

    # Combine block + street for a single high-cardinality location feature
    df["block_street"] = df["block"].astype(str).str.strip() + " " + df["street_name"].astype(str).str.strip()

    return df
