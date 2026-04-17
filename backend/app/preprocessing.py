from datetime import datetime, timezone
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
    lease_commence = round(now.year - (99 - remaining))

    return pd.DataFrame([{
        "town":                   req.town,
        "flat_type":              req.flat_type,
        "flat_model":             req.flat_model,
        "storey_midpoint":        storey_mid,
        "floor_area_sqm":         req.floor_area_sqm,
        "remaining_lease_years":  remaining,
        "lease_commence_date":    lease_commence,
        "street_name":            req.street_name if req.street_name else DEFAULT_STREET_NAME,
        "block":                  req.block if req.block else DEFAULT_BLOCK,
        "transaction_year":       now.year,
        "transaction_month":      now.month,
    }])
