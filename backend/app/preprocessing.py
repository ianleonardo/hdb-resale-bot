from datetime import datetime, timezone
import numpy as np
import pandas as pd

# Defaults used when optional fields are not provided
DEFAULT_REMAINING_LEASE_YEARS = 70.0   # ~median for Singapore HDB stock
DEFAULT_STREET_NAME = "UNKNOWN"
DEFAULT_BLOCK = "UNKNOWN"


def build_inference_dataframe(req) -> pd.DataFrame:
    """Convert a PredictRequest into a feature DataFrame ready for preprocessor.transform()."""
    parts = req.storey_range.split(" TO ")
    storey_mid = (int(parts[0]) + int(parts[1])) / 2
    now = datetime.now(timezone.utc)

    remaining = req.remaining_lease_years if req.remaining_lease_years is not None else DEFAULT_REMAINING_LEASE_YEARS
    street = req.street_name.strip() if req.street_name else DEFAULT_STREET_NAME
    block = req.block.strip() if req.block else DEFAULT_BLOCK
    block_street = f"{block} {street}"

    month = now.month
    return pd.DataFrame([{
        "town":                   req.town,
        "flat_type":              req.flat_type,
        "flat_model":             req.flat_model,
        "storey_midpoint":        storey_mid,
        "floor_area_sqm":         req.floor_area_sqm,
        "remaining_lease_years":  remaining,
        "street_name":            street,
        "block":                  block,
        "block_street":           block_street,
        "transaction_year":       now.year,
        "month_sin":              np.sin(2 * np.pi * month / 12),
        "month_cos":              np.cos(2 * np.pi * month / 12),
    }])
