"""
Build block_lookup.parquet for backend v2 inference.

Joins data/raw/hdb_property_info_geocoded.csv with resale-derived (town, block) aggregates
from data/hdb_resale_complete.csv. One row per (town, block); single read at backend startup.

Usage:
  python scripts/build_block_inference_lookup.py
  # writes training/v2/artifacts/block_lookup.parquet (and copies to backend/artifacts if dir exists)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RAW_PROP = BASE / "data" / "raw" / "hdb_property_info_geocoded.csv"
RESALE_PATH = BASE / "data" / "hdb_resale_complete.csv"
OUT_DIR = BASE / "training" / "v2" / "artifacts"
OUT_PATH = OUT_DIR / "block_lookup.parquet"


def _mode_first(s: pd.Series):
    s = s.dropna()
    if s.empty:
        return np.nan
    m = s.mode()
    return m.iloc[0] if len(m) else s.iloc[0]


def _norm_block_street(df: pd.DataFrame, blk: str, st: str) -> pd.DataFrame:
    out = df.copy()
    out[blk] = out[blk].astype(str).str.strip().str.upper()
    out[st] = out[st].astype(str).str.strip().str.upper()
    return out[(out[blk] != "") & (out[st] != "")]


def main() -> None:
    if not RAW_PROP.exists():
        raise FileNotFoundError(f"Missing {RAW_PROP} (run scripts/2_download_hd_property_info.py)")
    if not RESALE_PATH.exists():
        raise FileNotFoundError(f"Missing {RESALE_PATH}")

    prop = pd.read_csv(RAW_PROP, low_memory=False)
    prop = prop.rename(columns={"blk_no": "block", "street": "street_name"})
    prop = _norm_block_street(prop, "block", "street_name")

    resale = pd.read_csv(RESALE_PATH, low_memory=False)
    resale = _norm_block_street(resale, "block", "street_name")

    # Canonical town per (block, street): most frequent in resale history
    tc = resale.groupby(["block", "street_name", "town"], as_index=False).size()
    tc = tc.sort_values("size", ascending=False).drop_duplicates(["block", "street_name"])
    town_map = tc[["block", "street_name", "town"]].copy()
    town_map["town"] = town_map["town"].astype(str).str.strip().str.upper()

    prop = prop.merge(town_map, on=["block", "street_name"], how="inner")
    prop["town"] = prop["town"].astype(str).str.strip().str.upper()

    # Coordinates: drop invalid
    for col in ["Latitude", "Longitude"]:
        prop[col] = pd.to_numeric(prop[col], errors="coerce")

    numeric_prop = [
        "max_floor_lvl", "year_completed", "total_dwelling_units",
        "2room_sold", "3room_sold", "4room_sold", "5room_sold", "exec_sold",
    ]
    for c in numeric_prop:
        if c in prop.columns:
            prop[c] = pd.to_numeric(prop[c], errors="coerce")

    prop = prop.sort_values("total_dwelling_units", ascending=False, na_position="last")
    prop_agg = prop.groupby(["town", "block"], as_index=False).first()

    resale["town"] = resale["town"].astype(str).str.strip().str.upper()

    median_cols = [
        "lease_commence_date",
        "max_floor_lvl",
        "year_completed",
        "total_dwelling_units",
        "2room_sold", "3room_sold", "4room_sold", "5room_sold", "exec_sold",
        "mrt_nearest_distance",
        "Mall_Nearest_Distance",
        "Hawker_Nearest_Distance",
        "bus_stop_nearest_distance",
        "pri_sch_nearest_distance",
        "sec_sch_nearest_dist",
        "Mall_Within_500m", "Mall_Within_1km", "Mall_Within_2km",
        "Hawker_Within_500m", "Hawker_Within_1km", "Hawker_Within_2km",
        "pri_sch_affiliation",
        "Latitude",
        "Longitude",
    ]
    mode_cols = [
        "flat_type",
        "flat_model",
        "street_name",
        "mrt_name",
        "pri_sch_name",
        "sec_sch_name",
    ]

    agg_kw = {c: "median" for c in median_cols if c in resale.columns}
    for c in mode_cols:
        if c in resale.columns:
            agg_kw[c] = _mode_first

    resale_gb = resale.groupby(["town", "block"]).agg(agg_kw).reset_index()

    merged = resale_gb.merge(prop_agg, on=["town", "block"], how="outer", suffixes=("", "_prop"))

    # Prefer resale aggregates; fill structure fields from property file
    prop_suffix_cols = [c for c in merged.columns if c.endswith("_prop")]
    for c in prop_suffix_cols:
        base = c.replace("_prop", "")
        if base in merged.columns:
            merged[base] = merged[base].fillna(merged[c])
        else:
            merged[base] = merged[c]
        merged.drop(columns=[c], inplace=True)

    merged["town"] = merged["town"].astype(str).str.strip().str.upper()
    merged["block"] = merged["block"].astype(str).str.strip().str.upper()

    merged = merged.drop_duplicates(["town", "block"]).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(merged):,} rows -> {OUT_PATH}")

    backend_art = BASE / "backend" / "artifacts"
    if backend_art.parent.exists():
        backend_art.mkdir(exist_ok=True)
        backend_out = backend_art / "block_lookup.parquet"
        merged.to_parquet(backend_out, index=False)
        print(f"Copied -> {backend_out}")


if __name__ == "__main__":
    main()
