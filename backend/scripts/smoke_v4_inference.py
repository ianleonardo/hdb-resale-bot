#!/usr/bin/env python3
"""Sanity-check v4 inference row construction (CatBoost–ARIMA Pool shape vs metrics_v4.json).

Run from repo root:
  python backend/scripts/smoke_v4_inference.py

Uses ``training/v4/artifacts`` by default. If ``block_lookup.parquet`` is absent there,
files are staged into a temp dir with ``block_lookup`` copied from ``training/v2/artifacts``.

If pip fails building CatBoost on Python 3.13, use Python 3.12 (matches Dockerfile).
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))


def _artifact_dir_for_smoke() -> Path:
    v4 = REPO_ROOT / "training" / "v4" / "artifacts"
    required_v4 = [
        "metrics_v4.json",
        "spatial_inference.pkl",
        "arima_bundle_v4.pkl",
    ]
    for name in required_v4:
        if not (v4 / name).exists():
            raise SystemExit(f"Missing {v4 / name}")

    bl_v4 = v4 / "block_lookup.parquet"
    if bl_v4.exists():
        return v4

    bl_fb = REPO_ROOT / "training" / "v2" / "artifacts" / "block_lookup.parquet"
    if not bl_fb.exists():
        raise SystemExit(
            f"No block_lookup at {bl_v4}; place block_lookup.parquet in v4 artifacts "
            f"or ensure fallback exists at {bl_fb}"
        )

    d = Path(tempfile.mkdtemp(prefix="hdb_smoke_v4_"))
    for name in required_v4:
        shutil.copy2(v4 / name, d / name)
    shutil.copy2(bl_fb, d / "block_lookup.parquet")
    return d


def main() -> None:
    import json

    art_dir = _artifact_dir_for_smoke()
    os.environ["BACKEND_ARTIFACT_DIR"] = str(art_dir)
    os.environ.setdefault("HDB_RPI_PATH", str(REPO_ROOT / "data" / "hdb_rpi.csv"))

    from app.preprocessing import build_inference_pool
    from app.schemas import PredictRequest

    metrics_path = art_dir / "metrics_v4.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    expected = len(metrics["features"])

    req = PredictRequest(
        town="TAMPINES",
        block="406",
        storey_range="07 TO 09",
        floor_area_sqm=93.0,
    )
    pool = build_inference_pool(req)
    nfeat = pool.num_col()
    nrow = pool.num_row()
    assert nrow == 1, nrow
    assert nfeat == expected, (nfeat, expected)
    print(f"OK CatBoost Pool rows={nrow} cols={nfeat} (matches metrics_v4.json)")


if __name__ == "__main__":
    main()
