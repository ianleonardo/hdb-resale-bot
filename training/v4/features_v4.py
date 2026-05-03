"""
features_v4.py — Feature definitions and drop lists for HDB v4 (ARIMA + CatBoost).

v4 extends v2's 49-feature CatBoost set with 2 ARIMA market-baseline features.
Both are relative or static quantities so they present equally valid signals to
CatBoost for in-sample training rows and out-of-sample val/test rows.

  arima_seg_vs_global   segment level[M] − global level[M]: encodes whether this
                        (town, flat_type) cell is trending above or below the
                        overall market.  Even if both levels are off for future
                        periods, their difference is a stable cross-sectional signal.
  arima_seg_series_std  historical log-price volatility per (town, flat_type)
                        computed from the training series.  Gives CatBoost a
                        per-segment uncertainty/risk signal that is symmetric
                        across all splits (it is a training-era constant).

The v2 `hdb_rpi` (actual lagged RPI) is kept as the primary RPI feature.
`arima_rpi_forecast` is available on ARIMABundle for backend inference-time
feature switching when the official RPI series is stale (future-date queries).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT   = Path(__file__).resolve().parents[2]
_BACKEND_DIR = _REPO_ROOT / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from arima_v4 import ARIMA_FEATURES  # noqa: E402

# ── Split config (mirrors v2) ────────────────────────────────────────────────
HIST_YEAR_START  = 2017   # earliest data loaded for ARIMA fitting
TRAIN_YEAR_START = 2020   # first year used in CatBoost loss / gradient updates
TRAIN_YEAR_END   = 2024
VAL_YEAR         = 2025
TEST_YEAR_START  = 2026
TEST_YEAR_END    = 2026

# ── v2 drop lists (unchanged) ────────────────────────────────────────────────
REDUNDANT_COLS = [
    "mrt_nearest_distance", "Mall_Nearest_Distance", "Hawker_Nearest_Distance",
    "bus_stop_nearest_distance", "pri_sch_nearest_distance", "sec_sch_nearest_dist",
]
LOW_IMP_COLS = [
    "1room_rental", "2room_rental", "3room_rental", "other_room_rental",
    "1room_sold", "studio_apartment_sold", "multigen_sold",
    "residential", "commercial", "market_hawker",
    "multistorey_carpark", "precinct_pavilion",
    "mrt_interchange", "bus_interchange",
    "Tranc_Month",
    "affiliation", "pri_sch_affiliation",
    "hawker_market_stalls", "hawker_food_stalls",
    "school_quality", "cutoff_point",
    "Latitude", "Longitude",
]
LOCATION_ID_COLS  = ["block", "street_name", "postal"]
IDENTIFIER_COLS   = [
    "resale_price", "Tranc_YearMonth", "storey_range", "lease_commence_date",
    "bus_stop_name", "bus_stop_latitude", "bus_stop_longitude",
    "mrt_latitude", "mrt_longitude",
    "pri_sch_latitude", "pri_sch_longitude",
    "sec_sch_latitude", "sec_sch_longitude",
    "planning_area",
]

DROP_COLS = set(
    ["log_resale_price"]       # target
    + IDENTIFIER_COLS
    + LOCATION_ID_COLS
    + REDUNDANT_COLS
    + LOW_IMP_COLS
)

CAT_FEATURES = [
    "flat_type", "flat_model", "town",
    "mrt_name", "pri_sch_name", "sec_sch_name",
]


def get_feature_cols(df_columns: list[str]) -> tuple[list[str], list[str]]:
    """
    Return (feature_cols, cat_cols) for a DataFrame that already has both
    v2 engineered columns and ARIMA feature columns injected.

    ARIMA features are numeric — they are NOT in DROP_COLS so they will be
    automatically included when we subtract DROP_COLS from the column set.
    """
    feature_cols = [c for c in df_columns if c not in DROP_COLS]
    cat_cols     = [c for c in CAT_FEATURES if c in feature_cols]
    return feature_cols, cat_cols
