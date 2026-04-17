"""
preprocessing_v2.py — adds 6-month recent-comps features to the baseline pipeline.

Key design rule: compute_comp_features() must be called on the FULL dataset
(train + val + test combined) BEFORE the time-based split. Each row only reads
sales strictly before its own transaction month, so there is no leakage.
"""

from bisect import bisect_left

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

ORDINAL_FLAT_TYPE = [
    "1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM",
    "5 ROOM", "EXECUTIVE", "MULTI-GENERATION",
]

NUMERIC_BASE = [
    "floor_area_sqm", "storey_midpoint", "remaining_lease_years",
    "transaction_year", "month_sin", "month_cos",
]
COMP_FEATURES = [
    "comp_count", "comp_median_price", "comp_median_ppqm",
    "comp_p25_price", "comp_p75_price",
]
NUMERIC_FEATURES = NUMERIC_BASE + COMP_FEATURES
ORDINAL_FEATURES = ["flat_type"]
NOMINAL_FEATURES = ["town", "flat_model"]
HIGH_CARD_STREET = ["street_name"]
HIGH_CARD_BLOCK  = ["block"]
HIGH_CARD_COMBO  = ["block_street"]

ALL_FEATURES = (
    NUMERIC_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES
    + HIGH_CARD_STREET + HIGH_CARD_BLOCK + HIGH_CARD_COMBO
)

TARGET = "log_resale_price"


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num",       "passthrough",                                               NUMERIC_FEATURES),
        ("ord",       OrdinalEncoder(categories=[ORDINAL_FLAT_TYPE]),              ORDINAL_FEATURES),
        ("nom",       OneHotEncoder(handle_unknown="ignore", sparse_output=False), NOMINAL_FEATURES),
        ("hc_street", TargetEncoder(smoothing=10),                                 HIGH_CARD_STREET),
        ("hc_block",  TargetEncoder(smoothing=10),                                 HIGH_CARD_BLOCK),
        ("hc_combo",  TargetEncoder(smoothing=50),                                 HIGH_CARD_COMBO),
    ])


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["transaction_year"]  = pd.to_datetime(df["month"]).dt.year
    df["transaction_month"] = pd.to_datetime(df["month"]).dt.month
    df["ym"]                = df["transaction_year"] * 12 + df["transaction_month"]

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

    df["block_street"]     = df["block"].astype(str).str.strip() + " " + df["street_name"].astype(str).str.strip()
    df["log_resale_price"] = np.log1p(df["resale_price"])
    return df


def compute_comp_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 6-month lookback comparable-sales features for every row.

    For a transaction in month T, comps = all sales on the same street_name
    where month is in [T-6, T) — strictly before T, no same-month leakage.

    Call this on the FULL dataset (all splits combined) before splitting,
    then split afterward. The time boundary guarantees no leakage.
    """
    df = df.copy()
    df["price_per_sqm"] = df["resale_price"] / df["floor_area_sqm"]

    for col in COMP_FEATURES:
        df[col] = np.nan
    df["comp_count"] = 0

    for street_name, street_df in df.groupby("street_name"):
        sorted_df = street_df.sort_values("ym")
        yms    = sorted_df["ym"].tolist()
        prices = sorted_df["resale_price"].tolist()
        ppqms  = sorted_df["price_per_sqm"].tolist()

        # Compute window stats once per unique ym for this street
        ym_stats: dict[int, dict] = {}
        for ym in sorted(set(yms)):
            lo = bisect_left(yms, ym - 6)   # first month in [ym-6, ym)
            hi = bisect_left(yms, ym)        # exclusive of current month
            pw = prices[lo:hi]
            qw = ppqms[lo:hi]
            n  = len(pw)
            ym_stats[ym] = {
                "comp_count":        n,
                "comp_median_price": float(np.median(pw))         if n else np.nan,
                "comp_median_ppqm":  float(np.median(qw))         if n else np.nan,
                "comp_p25_price":    float(np.percentile(pw, 25)) if n else np.nan,
                "comp_p75_price":    float(np.percentile(pw, 75)) if n else np.nan,
            }

        for col in ["comp_count", "comp_median_price", "comp_median_ppqm",
                    "comp_p25_price", "comp_p75_price"]:
            df.loc[street_df.index, col] = (
                street_df["ym"].map({ym: s[col] for ym, s in ym_stats.items()})
            )

    return df


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def build_comp_lookup(df: pd.DataFrame) -> dict:
    """
    Build a per-street price history dict for use at inference time.
    Pass the full training dataset (or the most recent N months if storage is a concern).
    Save the result as comp_lookup.pkl alongside model.pkl.
    """
    df = df.copy()
    df["price_per_sqm"] = df["resale_price"] / df["floor_area_sqm"]

    lookup: dict = {}
    for street_name, group in df.groupby("street_name"):
        sg = group.sort_values("ym")
        lookup[street_name] = {
            "yms":    sg["ym"].tolist(),
            "prices": sg["resale_price"].tolist(),
            "ppqms":  sg["price_per_sqm"].tolist(),
        }
    return lookup


def get_comp_features_inference(street_name: str, ym: int, comp_lookup: dict) -> dict:
    """
    Return comp features for a single inference request.
    ym = transaction_year * 12 + transaction_month (use current date at inference).
    Falls back to comp_count=0 / NaN for unknown streets.
    """
    null = {
        "comp_count": 0, "comp_median_price": np.nan,
        "comp_median_ppqm": np.nan, "comp_p25_price": np.nan, "comp_p75_price": np.nan,
    }
    data = comp_lookup.get(street_name)
    if not data:
        return null

    yms = data["yms"]
    lo  = bisect_left(yms, ym - 6)
    hi  = bisect_left(yms, ym)
    pw  = data["prices"][lo:hi]
    qw  = data["ppqms"][lo:hi]
    n   = len(pw)

    if n == 0:
        return null
    return {
        "comp_count":        n,
        "comp_median_price": float(np.median(pw)),
        "comp_median_ppqm":  float(np.median(qw)),
        "comp_p25_price":    float(np.percentile(pw, 25)),
        "comp_p75_price":    float(np.percentile(pw, 75)),
    }
