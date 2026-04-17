from bisect import bisect_left
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Feature schema — must match preprocessing_catboost.py exactly
NUMERIC_FEATURES = [
    "floor_area_sqm", "storey_midpoint", "remaining_lease_years",
    "transaction_year", "month_sin", "month_cos",
    "comp_count", "comp_median_price", "comp_median_ppqm",
    "comp_p25_price", "comp_p75_price",
]
CAT_FEATURES = ["flat_type", "town", "flat_model", "street_name", "block", "block_street"]
ALL_FEATURES = NUMERIC_FEATURES + CAT_FEATURES

DEFAULT_REMAINING_LEASE_YEARS = 70.0
DEFAULT_STREET_NAME           = "UNKNOWN"
DEFAULT_BLOCK                 = "UNKNOWN"


def _get_comp_features(street_name: str, ym: int, comp_lookup: dict) -> dict:
    """
    Return 6-month lookback comp stats for a single inference row.
    ym = transaction_year * 12 + transaction_month.
    Falls back to comp_count=0 / NaN for streets not in the lookup.
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


def build_inference_dataframe(req, comp_lookup: dict) -> pd.DataFrame:
    """Build a single-row DataFrame ready for CatBoost prediction."""
    parts      = req.storey_range.split(" TO ")
    storey_mid = (int(parts[0]) + int(parts[1])) / 2
    now        = datetime.now(timezone.utc)
    month      = now.month

    remaining    = req.remaining_lease_years if req.remaining_lease_years is not None else DEFAULT_REMAINING_LEASE_YEARS
    street_name  = req.street_name.strip() if req.street_name else DEFAULT_STREET_NAME
    block        = req.block.strip()        if req.block        else DEFAULT_BLOCK
    block_street = f"{block} {street_name}"

    ym    = now.year * 12 + month
    comps = _get_comp_features(street_name, ym, comp_lookup)

    row = {
        # Numeric
        "floor_area_sqm":        req.floor_area_sqm,
        "storey_midpoint":       storey_mid,
        "remaining_lease_years": remaining,
        "transaction_year":      now.year,
        "month_sin":             np.sin(2 * np.pi * month / 12),
        "month_cos":             np.cos(2 * np.pi * month / 12),
        **comps,
        # Categorical — must be str for CatBoost
        "flat_type":   str(req.flat_type),
        "town":        str(req.town),
        "flat_model":  str(req.flat_model),
        "street_name": street_name,
        "block":       block,
        "block_street": block_street,
    }
    return pd.DataFrame([row])[ALL_FEATURES]
