"""
Shared HDB v2 feature engineering + spatial encoding for training and inference.
Training imports from backend/app via sys.path; backend uses directly.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

LAT_M = 111_000.0
LON_M = 110_970.0
RADII_M = [500, 2000]
SPATIAL_FEATS = (
    [f"spatial_{r}m_te" for r in RADII_M]
    + [f"spatial_{r}m_psm" for r in RADII_M]
)


def smooth(count: int, grp_mean: float, global_mean: float, k: int = 10) -> float:
    s = 1.0 / (1.0 + np.exp(-(count - k) / k))
    return global_mean * (1 - s) + grp_mean * s


def latlon_to_m_df(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        df["Latitude"].to_numpy(dtype=float) * LAT_M,
        df["Longitude"].to_numpy(dtype=float) * LON_M,
    ])


def query_coords_from_lat_lon(lat: float, lon: float) -> np.ndarray:
    """Shape (1, 2) query points in metres."""
    return np.array([[lat * LAT_M, lon * LON_M]], dtype=np.float64)


def build_spatial_bundle_dict(train_df: pd.DataFrame, smooth_k: int = 10) -> dict:
    coords_tr = latlon_to_m_df(train_df).astype(np.float32)
    prices_tr = train_df["resale_price"].to_numpy(dtype=np.float64)
    psm_tr = (train_df["resale_price"] / train_df["floor_area_sqm"]).to_numpy(dtype=np.float64)
    return {
        "coords_train_m": coords_tr,
        "prices_train": prices_tr,
        "psm_train": psm_tr,
        "global_price": float(prices_tr.mean()),
        "global_psm": float(psm_tr.mean()),
        "radii_m": list(RADII_M),
        "smooth_k": smooth_k,
        "lat_m": LAT_M,
        "lon_m": LON_M,
    }


def encode_queries_spatial(query_coords_m: np.ndarray, bundle: dict) -> dict[str, np.ndarray]:
    """Encode spatial features for query points using full training KDTree (val/test parity)."""
    tree = cKDTree(bundle["coords_train_m"])
    prices_tr = bundle["prices_train"]
    psm_tr = bundle["psm_train"]
    gp = bundle["global_price"]
    gpsm = bundle["global_psm"]
    k_smooth = int(bundle.get("smooth_k", 10))

    def _encode_radius(
        qc: np.ndarray,
        tree_: cKDTree,
        values: np.ndarray,
        global_val: float,
        radius: float,
    ) -> np.ndarray:
        nbrs = tree_.query_ball_point(qc, r=radius, workers=-1)
        out = np.empty(len(qc))
        for i, nbr in enumerate(nbrs):
            out[i] = (
                global_val if len(nbr) == 0
                else smooth(len(nbr), float(values[nbr].mean()), global_val, k_smooth)
            )
        return out

    out: dict[str, np.ndarray] = {}
    for r in bundle["radii_m"]:
        out[f"spatial_{r}m_te"] = _encode_radius(query_coords_m, tree, prices_tr, gp, r)
        out[f"spatial_{r}m_psm"] = _encode_radius(query_coords_m, tree, psm_tr, gpsm, r)
    return out


def compute_spatial_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    k: int = 10,
) -> tuple[dict, dict, dict]:
    """Training pipeline spatial encoding (temporal trees on train; full tree on val/test)."""
    prices_tr = df_train["resale_price"].to_numpy()
    psm_tr = (df_train["resale_price"] / df_train["floor_area_sqm"]).to_numpy()
    global_price = prices_tr.mean()
    global_psm = psm_tr.mean()

    coords_tr = latlon_to_m_df(df_train)
    coords_val = latlon_to_m_df(df_val)
    coords_te = latlon_to_m_df(df_test)

    def _encode_radius(
        query_coords: np.ndarray,
        tree: cKDTree,
        values: np.ndarray,
        global_val: float,
        radius: float,
    ) -> np.ndarray:
        nbrs = tree.query_ball_point(query_coords, r=radius, workers=-1)
        out = np.empty(len(query_coords))
        for i, nbr in enumerate(nbrs):
            out[i] = (
                global_val if len(nbr) == 0
                else smooth(len(nbr), values[nbr].mean(), global_val, k)
            )
        return out

    sort_key = (df_train["Tranc_Year"] * 100 + df_train["Tranc_Month"]).to_numpy()
    months = sorted(np.unique(sort_key))

    tr_enc = {feat: np.empty(len(df_train)) for feat in SPATIAL_FEATS}

    for month in months:
        past_mask = sort_key < month
        focal_mask = sort_key == month
        focal_coords = coords_tr[focal_mask]

        if past_mask.sum() == 0:
            for r in RADII_M:
                tr_enc[f"spatial_{r}m_te"][focal_mask] = global_price
                tr_enc[f"spatial_{r}m_psm"][focal_mask] = global_psm
            continue

        past_tree = cKDTree(coords_tr[past_mask])
        past_prices = prices_tr[past_mask]
        past_psm = psm_tr[past_mask]

        for r in RADII_M:
            tr_enc[f"spatial_{r}m_te"][focal_mask] = _encode_radius(
                focal_coords, past_tree, past_prices, global_price, r)
            tr_enc[f"spatial_{r}m_psm"][focal_mask] = _encode_radius(
                focal_coords, past_tree, past_psm, global_psm, r)

    full_tree = cKDTree(coords_tr)
    val_enc, te_enc = {}, {}
    for r in RADII_M:
        val_enc[f"spatial_{r}m_te"] = _encode_radius(coords_val, full_tree, prices_tr, global_price, r)
        val_enc[f"spatial_{r}m_psm"] = _encode_radius(coords_val, full_tree, psm_tr, global_psm, r)
        te_enc[f"spatial_{r}m_te"] = _encode_radius(coords_te, full_tree, prices_tr, global_price, r)
        te_enc[f"spatial_{r}m_psm"] = _encode_radius(coords_te, full_tree, psm_tr, global_psm, r)

    return tr_enc, val_enc, te_enc


def engineer_features(df: pd.DataFrame, mall_dist_median: float) -> pd.DataFrame:
    df = df.copy()

    df["lease_remaining_years"] = 99 - (df["Tranc_Year"] - df["lease_commence_date"])
    df["lease_remaining_pct"] = df["lease_remaining_years"] / 99.0
    df["tranc_period"] = df["Tranc_Year"] * 12 + df["Tranc_Month"]
    df["storey_ratio"] = df["mid_storey"] / df["max_floor_lvl"].replace(0, np.nan)
    df["is_high_floor"] = (df["mid_storey"] >= 20).astype(int)
    df["floor_band"] = pd.cut(
        df["mid_storey"],
        bins=[0, 5, 10, 15, 20, 30, 50, 999],
        labels=[1, 2, 3, 4, 5, 6, 7],
    ).astype(float)

    df["log_mrt_dist"] = np.log1p(df["mrt_nearest_distance"])
    df["log_mall_dist"] = np.log1p(df["Mall_Nearest_Distance"].fillna(mall_dist_median))
    df["log_hawker_dist"] = np.log1p(df["Hawker_Nearest_Distance"])
    df["log_bus_dist"] = np.log1p(df["bus_stop_nearest_distance"])
    df["log_pri_sch_dist"] = np.log1p(df["pri_sch_nearest_distance"])
    df["log_sec_sch_dist"] = np.log1p(df["sec_sch_nearest_dist"])
    df["accessibility_score"] = (
        df["log_mrt_dist"] * 0.4
        + df["log_mall_dist"] * 0.2
        + df["log_hawker_dist"] * 0.2
    )

    for col in ["Mall_Within_500m", "Mall_Within_1km", "Mall_Within_2km",
                "Hawker_Within_500m", "Hawker_Within_1km", "Hawker_Within_2km"]:
        df[col] = df[col].fillna(0)

    df["pri_school_quality"] = (
        df["pri_sch_affiliation"].fillna(0) * 10
        + 1.0 / (df["pri_sch_nearest_distance"] + 1)
    )

    df["area_x_storey"] = df["floor_area_sqm"] * df["mid_storey"]
    df["area_x_lease_rem"] = df["floor_area_sqm"] * df["lease_remaining_years"]
    df["storey_x_lease_rem"] = df["mid_storey"] * df["lease_remaining_years"]
    df["year_completed_x_floor_area"] = df["year_completed"] * df["floor_area_sqm"]

    df["log_resale_price"] = np.log1p(df["resale_price"])

    return df


def add_official_rpi(df: pd.DataFrame, rpi_source: Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(rpi_source, pd.DataFrame):
        rpi = rpi_source[["year", "quarter", "rpi"]].copy()
    else:
        rpi = pd.read_csv(rpi_source)[["year", "quarter", "rpi"]]
    rpi_map = rpi.set_index(["year", "quarter"])["rpi"].to_dict()

    def _lagged_rpi(year: int, month: int) -> float:
        q = (month - 1) // 3 + 1
        lag_year = year if q > 1 else year - 1
        lag_quarter = q - 1 if q > 1 else 4
        return rpi_map.get((lag_year, lag_quarter), float("nan"))

    df = df.copy()
    df["hdb_rpi"] = [_lagged_rpi(y, m) for y, m in zip(df["Tranc_Year"], df["Tranc_Month"])]

    n_null = df["hdb_rpi"].isna().sum()
    if n_null:
        last_rpi = rpi["rpi"].iloc[-1]
        df["hdb_rpi"] = df["hdb_rpi"].fillna(last_rpi)
        logger.warning(
            "hdb_rpi: %s rows beyond published series — filled with latest (%s)",
            n_null, last_rpi,
        )

    return df


def add_macro_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Richer macro/time signal for forward extrapolation (after add_official_rpi).

    Requires: hdb_rpi, Tranc_Year, tranc_period, floor_area_sqm.
    """
    df = df.copy()
    rpi = df["hdb_rpi"].astype(np.float64)
    df["rpi_x_year"] = rpi * df["Tranc_Year"].astype(np.float64)
    df["rpi_x_tranc_period"] = rpi * df["tranc_period"].astype(np.float64)
    df["rpi_x_floor_area_sqm"] = rpi * df["floor_area_sqm"].astype(np.float64)
    return df


def prepare_X(df: pd.DataFrame, feature_cols: list, cat_cols: list) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for col in cat_cols:
        X[col] = X[col].astype(str).str.strip()
    return X


def load_inference_metrics(path: Path | None, raw_json: str | None = None) -> dict:
    if raw_json is not None:
        return json.loads(raw_json)
    with open(path, encoding="utf-8") as f:
        return json.load(f)
