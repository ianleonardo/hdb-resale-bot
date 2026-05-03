#!/usr/bin/env python3
"""Sanity-check v2 inference row construction (no trained .cbm required).

Run from repo root:
  BACKEND_ARTIFACT_DIR=training/v2/artifacts python backend/scripts/smoke_v2_inference.py

Artifacts expected in that directory:
  metrics_catboost_v2.json, spatial_inference.pkl, block_lookup.parquet

If pip fails building CatBoost on Python 3.13, use Python 3.12 (matches Dockerfile) so a binary wheel installs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

os.environ.setdefault(
    "BACKEND_ARTIFACT_DIR",
    str(REPO_ROOT / "training" / "v2" / "artifacts"),
)
# Align blob basenames with files in v2 artifact dir (defaults in model_loader target v4).
os.environ.setdefault("MODEL_BLOB", "models/model_catboost_v2.cbm")
os.environ.setdefault("METRICS_BLOB", "models/metrics_catboost_v2.json")
os.environ.setdefault("HDB_RPI_PATH", str(REPO_ROOT / "data" / "hdb_rpi.csv"))


def main() -> None:
    import json

    # Import after env — model_loader reads BACKEND_ARTIFACT_DIR at call time (lru_cache clear not needed first run)
    from app.preprocessing import build_inference_pool
    from app.schemas import PredictRequest

    art_dir = Path(os.environ["BACKEND_ARTIFACT_DIR"])
    metrics_path = art_dir / "metrics_catboost_v2.json"
    if not metrics_path.exists():
        raise SystemExit(f"Missing {metrics_path}")

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
    print(f"OK CatBoost Pool rows={nrow} cols={nfeat} (matches metrics_catboost_v2.json)")


if __name__ == "__main__":
    main()
