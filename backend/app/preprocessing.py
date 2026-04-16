from datetime import datetime, timezone
import pandas as pd


def build_inference_dataframe(req) -> pd.DataFrame:
    """Convert a PredictRequest into a feature DataFrame ready for preprocessor.transform()."""
    parts = req.storey_range.split(" TO ")
    storey_mid = (int(parts[0]) + int(parts[1])) / 2
    now = datetime.now(timezone.utc)

    return pd.DataFrame([{
        "town":                   req.town,
        "flat_type":              req.flat_type,
        "flat_model":             req.flat_model,
        "storey_midpoint":        storey_mid,
        "floor_area_sqm":         req.floor_area_sqm,
        "remaining_lease_years":  req.remaining_lease_years,
        "street_name":            req.street_name,
        "block":                  req.block,
        "transaction_year":       now.year,
        "transaction_month":      now.month,
    }])
