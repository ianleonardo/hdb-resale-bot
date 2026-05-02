"""
Build CatBoost Pool rows matching training/v2 feature schema from minimal PredictRequest.
"""

from __future__ import annotations

import difflib
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from catboost import Pool

from app.constants import DEFAULT_FLAT_MODEL, DEFAULT_FLAT_TYPE
from app.inference_features import (
    SPATIAL_FEATS,
    add_macro_interaction_features,
    add_official_rpi,
    encode_queries_spatial,
    engineer_features,
    prepare_X,
    query_coords_from_lat_lon,
)
from app.model_loader import (
    load_block_lookup,
    load_inference_metrics,
    load_rpi_quarters_df,
    load_spatial_bundle,
)
from app.schemas import PredictRequest

logger = logging.getLogger(__name__)

TZ_SG = ZoneInfo("Asia/Singapore")

# When block_lookup misses — plausible central defaults (numeric columns used before engineer_features).
_FALLBACK_TEMPLATE = {
    "lease_commence_date":      1988.0,
    "flat_type":                DEFAULT_FLAT_TYPE,
    "flat_model":               DEFAULT_FLAT_MODEL,
    "street_name":              "UNKNOWN",
    "max_floor_lvl":            16.0,
    "year_completed":           1988.0,
    "total_dwelling_units":     120.0,
    "2room_sold":               0.0,
    "3room_sold":               0.0,
    "4room_sold":               80.0,
    "5room_sold":               0.0,
    "exec_sold":                0.0,
    "mrt_nearest_distance":     650.0,
    "Mall_Nearest_Distance":    1100.0,
    "Hawker_Nearest_Distance":  450.0,
    "bus_stop_nearest_distance": 220.0,
    "pri_sch_nearest_distance": 750.0,
    "sec_sch_nearest_dist":     1100.0,
    "Mall_Within_500m":         0.0,
    "Mall_Within_1km":          1.0,
    "Mall_Within_2km":          4.0,
    "Hawker_Within_500m":       1.0,
    "Hawker_Within_1km":        2.0,
    "Hawker_Within_2km":        5.0,
    "pri_sch_affiliation":      0.0,
    "mrt_name":                 "UNKNOWN",
    "pri_sch_name":             "UNKNOWN",
    "sec_sch_name":             "UNKNOWN",
    "Latitude":                 1.352,
    "Longitude":                103.8198,
}


def _normalize_street_label(s: object) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    t = str(s).strip().upper()
    if t in ("", "NAN", "NONE"):
        return ""
    return " ".join(t.split())


def _fuzzy_pick_normalized(query: str, normalized_candidates: list[str], cutoff: float = 0.55) -> str | None:
    """Return normalized lookup key closest to query (difflib — resilient to typos / spacing)."""
    q = _normalize_street_label(query)
    uniq = list(dict.fromkeys([c for c in normalized_candidates if c]))
    if not uniq:
        return None
    if q in uniq:
        return q
    m = difflib.get_close_matches(q, uniq, n=1, cutoff=cutoff)
    return m[0] if m else None


def _lookup_row_to_ctx(row: pd.Series) -> dict:
    out: dict = {}
    for k, v in row.items():
        ks = str(k)
        if ks.startswith("_"):
            continue
        if k in ("town", "block"):
            continue
        if pd.isna(v):
            continue
        if isinstance(v, str):
            out[k] = v.strip()
        elif isinstance(v, (bool, np.bool_)):
            out[k] = bool(v)
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        elif isinstance(v, (np.floating, float)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def _lookup_context(town: str, block: str, street_hint: str | None = None) -> dict:
    """Resolve property row: optional street_hint fuzzy-matches among rows sharing town+block."""
    df = load_block_lookup()
    t, b = town.strip().upper(), block.strip().upper()
    cand = df[(df["town"] == t) & (df["block"] == b)]
    if cand.empty:
        return {}

    dwell = cand.copy()
    if "total_dwelling_units" in dwell.columns:
        dwell["_dw"] = pd.to_numeric(dwell["total_dwelling_units"], errors="coerce").fillna(0)
        dwell = dwell.sort_values("_dw", ascending=False)
    dwell["_sn"] = dwell["street_name"].astype(str).map(_normalize_street_label)
    dwell = dwell[dwell["_sn"].str.len() > 0]
    if dwell.empty:
        return {}

    hint = street_hint.strip() if street_hint and str(street_hint).strip() else ""
    if hint:
        hit_norm = _fuzzy_pick_normalized(hint, dwell["_sn"].tolist())
        if hit_norm is not None:
            row = dwell[dwell["_sn"] == hit_norm].iloc[0]
            logger.debug("street hint %r matched canonical %s", hint, row["street_name"])
        else:
            row = dwell.iloc[0]
            logger.info(
                "street hint %r did not match any row for %s blk %s — using primary street %s",
                hint,
                t,
                b,
                row.get("street_name"),
            )
    else:
        row = dwell.iloc[0]

    return _lookup_row_to_ctx(row)


def _mid_storey(storey_range: str) -> float:
    parts = storey_range.upper().split(" TO ")
    return (int(parts[0]) + int(parts[1])) / 2


def _as_float(val, fallback: float) -> float:
    try:
        x = float(pd.to_numeric(val, errors="coerce"))
        if np.isnan(x):
            return fallback
        return x
    except (TypeError, ValueError):
        return fallback


def build_inference_pool(req: PredictRequest) -> Pool:
    metrics      = load_inference_metrics()
    bundle_sp    = load_spatial_bundle()
    feature_cols = metrics["features"]
    cat_cols     = metrics["cat_features"]
    mall_med     = float(metrics.get("mall_dist_median", _FALLBACK_TEMPLATE["Mall_Nearest_Distance"]))

    now = datetime.now(TZ_SG)
    ty, tm = now.year, now.month

    ctx = dict(_FALLBACK_TEMPLATE)
    ctx.update(_lookup_context(req.town, req.block, req.street_name))

    if req.flat_type:
        ctx["flat_type"] = req.flat_type
    if req.flat_model:
        ctx["flat_model"] = req.flat_model
    # Lookup supplies canonical street_name; only override if lookup missed entirely.
    if req.street_name and ctx.get("street_name") in (None, "", "UNKNOWN"):
        ctx["street_name"] = req.street_name.strip().upper()

    ms = _mid_storey(req.storey_range)

    lease_commence = _as_float(ctx["lease_commence_date"], _FALLBACK_TEMPLATE["lease_commence_date"])
    if req.remaining_lease_years is not None:
        lease_commence = float(ty - (99.0 - req.remaining_lease_years))

    lat = _as_float(ctx["Latitude"], _FALLBACK_TEMPLATE["Latitude"])
    lon = _as_float(ctx["Longitude"], _FALLBACK_TEMPLATE["Longitude"])

    raw = {
        "Tranc_Year":               ty,
        "Tranc_Month":              tm,
        "town":                     req.town.strip().upper(),
        "flat_type":                ctx["flat_type"],
        "flat_model":               ctx["flat_model"],
        "block":                    req.block.strip().upper(),
        "street_name":              ctx["street_name"],
        "floor_area_sqm":           float(req.floor_area_sqm),
        "mid_storey":               ms,
        "lease_commence_date":      lease_commence,
        "resale_price":             0.0,
        "max_floor_lvl":            _as_float(ctx["max_floor_lvl"], _FALLBACK_TEMPLATE["max_floor_lvl"]),
        "year_completed":           _as_float(ctx["year_completed"], _FALLBACK_TEMPLATE["year_completed"]),
        "total_dwelling_units":     _as_float(ctx["total_dwelling_units"], _FALLBACK_TEMPLATE["total_dwelling_units"]),
        "2room_sold":               _as_float(ctx["2room_sold"], _FALLBACK_TEMPLATE["2room_sold"]),
        "3room_sold":               _as_float(ctx["3room_sold"], _FALLBACK_TEMPLATE["3room_sold"]),
        "4room_sold":               _as_float(ctx["4room_sold"], _FALLBACK_TEMPLATE["4room_sold"]),
        "5room_sold":               _as_float(ctx["5room_sold"], _FALLBACK_TEMPLATE["5room_sold"]),
        "exec_sold":                _as_float(ctx["exec_sold"], _FALLBACK_TEMPLATE["exec_sold"]),
        "mrt_nearest_distance":     _as_float(ctx["mrt_nearest_distance"], _FALLBACK_TEMPLATE["mrt_nearest_distance"]),
        "Mall_Nearest_Distance":    _as_float(ctx["Mall_Nearest_Distance"], _FALLBACK_TEMPLATE["Mall_Nearest_Distance"]),
        "Hawker_Nearest_Distance":  _as_float(ctx["Hawker_Nearest_Distance"], _FALLBACK_TEMPLATE["Hawker_Nearest_Distance"]),
        "bus_stop_nearest_distance": _as_float(ctx["bus_stop_nearest_distance"], _FALLBACK_TEMPLATE["bus_stop_nearest_distance"]),
        "pri_sch_nearest_distance": _as_float(ctx["pri_sch_nearest_distance"], _FALLBACK_TEMPLATE["pri_sch_nearest_distance"]),
        "sec_sch_nearest_dist":     _as_float(ctx["sec_sch_nearest_dist"], _FALLBACK_TEMPLATE["sec_sch_nearest_dist"]),
        "Mall_Within_500m":         _as_float(ctx["Mall_Within_500m"], _FALLBACK_TEMPLATE["Mall_Within_500m"]),
        "Mall_Within_1km":          _as_float(ctx["Mall_Within_1km"], _FALLBACK_TEMPLATE["Mall_Within_1km"]),
        "Mall_Within_2km":          _as_float(ctx["Mall_Within_2km"], _FALLBACK_TEMPLATE["Mall_Within_2km"]),
        "Hawker_Within_500m":       _as_float(ctx["Hawker_Within_500m"], _FALLBACK_TEMPLATE["Hawker_Within_500m"]),
        "Hawker_Within_1km":        _as_float(ctx["Hawker_Within_1km"], _FALLBACK_TEMPLATE["Hawker_Within_1km"]),
        "Hawker_Within_2km":        _as_float(ctx["Hawker_Within_2km"], _FALLBACK_TEMPLATE["Hawker_Within_2km"]),
        "pri_sch_affiliation":      _as_float(ctx["pri_sch_affiliation"], _FALLBACK_TEMPLATE["pri_sch_affiliation"]),
        "mrt_name":                 str(ctx["mrt_name"]),
        "pri_sch_name":             str(ctx["pri_sch_name"]),
        "sec_sch_name":             str(ctx["sec_sch_name"]),
        "Latitude":                 lat,
        "Longitude":                lon,
    }

    df = pd.DataFrame([raw])
    df = engineer_features(df, mall_med)

    df = add_official_rpi(df, load_rpi_quarters_df())

    df = add_macro_interaction_features(df)

    qc = query_coords_from_lat_lon(lat, lon)
    spat = encode_queries_spatial(qc, bundle_sp)
    for feat in SPATIAL_FEATS:
        df[feat] = spat[feat][0]

    X = prepare_X(df, feature_cols, cat_cols)
    return Pool(X, cat_features=cat_cols)
