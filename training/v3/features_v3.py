"""
features_v3.py — Feature engineering and temporal sequence building for HDB v3 LSTM.

Reuses v2 property-level feature engineering from backend/app/inference_features.py.
Adds monthly market aggregate computation and LSTM sequence construction.

Key additions vs v2:
  - Sequence history loads 2017-2024 for monthly_agg; model fitting uses 2020-2024.
  - `build_monthly_agg`: per-(town, flat_type, year, month) market statistics.
  - `build_sequences`: for each transaction, look up the last SEQ_LEN months of segment
    stats strictly before the transaction month (no temporal leakage).
  - `fit_preprocessors` / `apply_preprocessors`: StandardScaler for static and sequence
    features, OrdinalEncoder for categoricals (embedding-ready).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = _REPO_ROOT / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.inference_features import (  # noqa: E402
    add_macro_interaction_features,
    add_official_rpi,
    engineer_features,
)

logger = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────────────────
DATA_PATH = _REPO_ROOT / "data" / "hdb_resale_complete.csv"
RPI_PATH  = _REPO_ROOT / "data" / "hdb_rpi.csv"

# ── Split config ────────────────────────────────────────────────────────────────
# TRAIN_YEAR_START matches v2 (2020-2024).  Broader historical data (back to 2017)
# is loaded separately in the training script solely for monthly_agg sequence context.
HIST_YEAR_START  = 2017   # earliest year loaded for sequence history (not model fitting)
TRAIN_YEAR_START = 2020   # first year included in model loss / gradient updates
TRAIN_YEAR_END   = 2024
VAL_YEAR         = 2025
TEST_YEAR_START  = 2026
TEST_YEAR_END    = 2026

# ── Sequence config ─────────────────────────────────────────────────────────────
SEQ_LEN = 12  # months of lookback for LSTM

# 7 per-month market features; each row covers one (town, flat_type, year, month) cell.
SEQ_FEATURES = [
    "log_mean_price",   # mean log1p(resale_price) for this segment-month
    "log_std_price",    # std of log prices (market volatility; 0 for single-txn months)
    "log_volume",       # log1p(transaction count)
    "log_mean_area",    # log1p(mean floor_area_sqm)
    "hdb_rpi",          # official HDB RPI lagged 1 quarter
    "month_sin",        # sin(2π·month/12) — seasonal encoding
    "month_cos",        # cos(2π·month/12)
]
N_SEQ_FEATURES = len(SEQ_FEATURES)

# ── Feature lists ────────────────────────────────────────────────────────────────
CAT_FEATURES = ["flat_type", "flat_model", "town", "mrt_name", "pri_sch_name", "sec_sch_name"]

STATIC_NUM_FEATURES = [
    "Tranc_Year", "floor_area_sqm", "mid_storey", "max_floor_lvl",
    "year_completed", "total_dwelling_units",
    "2room_sold", "3room_sold", "4room_sold", "5room_sold", "exec_sold",
    "Mall_Within_500m", "Mall_Within_1km", "Mall_Within_2km",
    "Hawker_Within_500m", "Hawker_Within_1km", "Hawker_Within_2km",
    "lease_remaining_years", "lease_remaining_pct", "tranc_period",
    "storey_ratio", "is_high_floor", "floor_band",
    "log_mrt_dist", "log_mall_dist", "log_hawker_dist", "log_bus_dist",
    "log_pri_sch_dist", "log_sec_sch_dist", "accessibility_score",
    "pri_school_quality",
    "area_x_storey", "area_x_lease_rem", "storey_x_lease_rem",
    "year_completed_x_floor_area",
    "hdb_rpi", "rpi_x_year", "rpi_x_tranc_period", "rpi_x_floor_area_sqm",
    "spatial_500m_te", "spatial_2000m_te", "spatial_500m_psm", "spatial_2000m_psm",
]
N_STATIC = len(STATIC_NUM_FEATURES)  # 43

TARGET = "log_resale_price"


# ── Data split ───────────────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[
        (df["Tranc_Year"] >= TRAIN_YEAR_START) & (df["Tranc_Year"] <= TRAIN_YEAR_END)
    ].reset_index(drop=True)
    val  = df[df["Tranc_Year"] == VAL_YEAR].reset_index(drop=True)
    test = df[
        (df["Tranc_Year"] >= TEST_YEAR_START) & (df["Tranc_Year"] <= TEST_YEAR_END)
    ].reset_index(drop=True)
    logger.info(
        "Split — train %d–%d: %d | val %d: %d | test %d–%d: %d",
        TRAIN_YEAR_START, TRAIN_YEAR_END, len(train),
        VAL_YEAR, len(val),
        TEST_YEAR_START, TEST_YEAR_END, len(test),
    )
    return train, val, test


# ── Sequence builder ──────────────────────────────────────────────────────────────

def build_monthly_agg(df_all: pd.DataFrame, rpi_path: Path) -> pd.DataFrame:
    """
    Compute per-(town, flat_type, year, month) market stats from all transactions.

    The resulting table is used as the sequence lookup; each transaction only sees
    months strictly before its own (enforced in build_sequences), so including
    val/test rows here does not cause leakage.
    """
    df = df_all[["Tranc_Year", "Tranc_Month", "town", "flat_type",
                 "resale_price", "floor_area_sqm"]].copy()
    df["log_price"] = np.log1p(df["resale_price"])

    agg = (
        df.groupby(["Tranc_Year", "Tranc_Month", "town", "flat_type"])
        .agg(
            log_mean_price=("log_price", "mean"),
            log_std_price=(
                "log_price",
                lambda x: float(x.std(ddof=0)) if len(x) > 1 else 0.0,
            ),
            volume=("log_price", "count"),
            mean_floor_area=("floor_area_sqm", "mean"),
        )
        .reset_index()
    )
    agg["log_volume"]    = np.log1p(agg["volume"])
    agg["log_mean_area"] = np.log1p(agg["mean_floor_area"])
    agg["log_std_price"] = agg["log_std_price"].fillna(0.0)
    agg["tranc_period"]  = agg["Tranc_Year"] * 12 + agg["Tranc_Month"]

    # RPI with 1-quarter lag (same logic as v2's add_official_rpi)
    rpi_df  = pd.read_csv(rpi_path)[["year", "quarter", "rpi"]]
    rpi_map = rpi_df.set_index(["year", "quarter"])["rpi"].to_dict()
    last_rpi = float(rpi_df["rpi"].iloc[-1])

    def _lagged(year: int, month: int) -> float:
        q = (month - 1) // 3 + 1
        ly, lq = (year, q - 1) if q > 1 else (year - 1, 4)
        return rpi_map.get((ly, lq), float("nan"))

    agg["hdb_rpi"] = [_lagged(y, m) for y, m in zip(agg["Tranc_Year"], agg["Tranc_Month"])]
    agg["hdb_rpi"] = agg["hdb_rpi"].fillna(last_rpi)

    agg["month_sin"] = np.sin(2 * np.pi * agg["Tranc_Month"] / 12)
    agg["month_cos"] = np.cos(2 * np.pi * agg["Tranc_Month"] / 12)

    return agg.sort_values("tranc_period").reset_index(drop=True)


def build_sequences(
    df: pd.DataFrame,
    monthly_agg: pd.DataFrame,
    seq_len: int = SEQ_LEN,
) -> np.ndarray:
    """
    Build a [N, seq_len, N_SEQ_FEATURES] float32 array.

    For each transaction, collects the last `seq_len` (town, flat_type) segment-months
    strictly before the transaction month, right-aligned.  Earlier positions are
    zero-padded (cold start / insufficient history).
    """
    sequences = np.zeros((len(df), seq_len, N_SEQ_FEATURES), dtype=np.float32)

    # Pre-index by (town, flat_type) → sorted periods + feature arrays
    lookup: dict[tuple, dict] = {}
    for (town, ft), grp in monthly_agg.groupby(["town", "flat_type"]):
        grp_s = grp.sort_values("tranc_period")
        lookup[(town, ft)] = {
            "periods":  grp_s["tranc_period"].values,
            "features": grp_s[SEQ_FEATURES].values.astype(np.float32),
        }

    towns      = df["town"].values
    flat_types = df["flat_type"].values
    periods    = (df["Tranc_Year"] * 12 + df["Tranc_Month"]).values.astype(int)

    cold_start = 0
    for i in range(len(df)):
        key    = (towns[i], flat_types[i])
        curr_p = periods[i]
        if key not in lookup:
            cold_start += 1
            continue
        all_p = lookup[key]["periods"]
        all_f = lookup[key]["features"]
        mask  = all_p < curr_p
        if not mask.any():
            cold_start += 1
            continue
        past = all_f[mask]
        n    = min(len(past), seq_len)
        sequences[i, seq_len - n:] = past[-n:]  # right-aligned; most recent at seq_len-1

    if cold_start:
        logger.warning("Cold-start (zero-padded) rows: %d / %d", cold_start, len(df))
    return sequences


# ── Preprocessors ────────────────────────────────────────────────────────────────

def fit_preprocessors(
    train_df: pd.DataFrame,
    train_sequences: np.ndarray,
) -> dict:
    """
    Fit scalers and encoder on training data.  Returns a dict of fitted objects
    plus metadata needed for inference.
    """
    # Static numericals — fill NaN with train median then scale
    X_tr = train_df[STATIC_NUM_FEATURES].values.astype(float)
    static_medians = np.nanmedian(X_tr, axis=0)
    nan_mask = np.isnan(X_tr)
    X_tr[nan_mask] = np.take(static_medians, np.where(nan_mask)[1])

    static_scaler = StandardScaler()
    static_scaler.fit(X_tr)

    # Sequence features — fit only on non-zero (non-padded) timesteps
    flat_seq = train_sequences.reshape(-1, N_SEQ_FEATURES)
    nonzero  = flat_seq.any(axis=1)
    seq_scaler = StandardScaler()
    seq_scaler.fit(flat_seq[nonzero] if nonzero.sum() > 100 else flat_seq)

    # Categorical → integer indices; unknown → -1 (shifted to 0 at apply time)
    cat_enc = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.int64,
    )
    cat_enc.fit(train_df[CAT_FEATURES].astype(str))
    cat_vocab_sizes = {
        name: len(cats)
        for name, cats in zip(CAT_FEATURES, cat_enc.categories_)
    }

    return {
        "static_scaler":  static_scaler,
        "static_medians": static_medians,
        "seq_scaler":     seq_scaler,
        "cat_enc":        cat_enc,
        "cat_vocab_sizes": cat_vocab_sizes,
    }


def apply_preprocessors(
    df: pd.DataFrame,
    sequences: np.ndarray,
    prep: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Apply fitted preprocessors. Returns (static_num, scaled_seq, cat_dict).

    static_num : float32 [N, N_STATIC]
    scaled_seq : float32 [N, seq_len, N_SEQ_FEATURES] — padding positions kept as 0
    cat_dict   : {feature_name: int64 [N]} — indices 1..vocab_size, 0 = unknown
    """
    # Static
    X = df[STATIC_NUM_FEATURES].values.astype(float)
    medians  = prep["static_medians"]
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(medians, np.where(nan_mask)[1])
    static_num = prep["static_scaler"].transform(X).astype(np.float32)

    # Sequences — scale then restore zero-padding
    N, T, F = sequences.shape
    flat        = sequences.reshape(-1, F)
    is_padding  = ~flat.any(axis=1)          # True for all-zero (padded) timesteps
    scaled_flat = prep["seq_scaler"].transform(flat).astype(np.float32)
    scaled_flat[is_padding] = 0.0            # re-zero padding after scaling
    scaled_seq  = scaled_flat.reshape(N, T, F)

    # Categoricals: OrdinalEncoder outputs -1 for unknown; shift +1 → 0 = unknown token
    cat_raw = prep["cat_enc"].transform(df[CAT_FEATURES].astype(str))  # [N, n_cats]
    cat_dict = {
        name: (cat_raw[:, i] + 1).astype(np.int64)
        for i, name in enumerate(CAT_FEATURES)
    }

    return static_num, scaled_seq, cat_dict


# ── Full feature prep helper (used in train + HPO) ───────────────────────────────

def prepare_split(
    raw_df: pd.DataFrame,
    mall_dist_median: float,
    monthly_agg: pd.DataFrame,
    prep: dict,
    spatial_feats: dict,
    seq_len: int = SEQ_LEN,
) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    """
    End-to-end feature prep for one split.  Returns (static_num, scaled_seq, cat_dict, y).
    Expects spatial_feats dict with keys matching SPATIAL_FEATS already computed.
    """
    df = engineer_features(raw_df, mall_dist_median)
    df = add_official_rpi(df, RPI_PATH)
    df = add_macro_interaction_features(df)
    for feat, vals in spatial_feats.items():
        df[feat] = vals

    seqs = build_sequences(df, monthly_agg, seq_len)
    static_num, scaled_seq, cat_dict = apply_preprocessors(df, seqs, prep)
    y = df[TARGET].values.astype(np.float32)
    return static_num, scaled_seq, cat_dict, y
