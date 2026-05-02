import io
import json
import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostRegressor
from google.cloud import storage

logger = logging.getLogger(__name__)

GCS_BUCKET = os.environ.get("GCS_BUCKET", "hdb-resale-artifacts")

# Blob paths under bucket (single prefix models/)
MODEL_BLOB           = os.environ.get("MODEL_BLOB", "models/model_catboost_v2.cbm")
METRICS_BLOB         = os.environ.get("METRICS_BLOB", "models/metrics_catboost_v2.json")
SPATIAL_BLOB         = os.environ.get("SPATIAL_BLOB", "models/spatial_inference.pkl")
BLOCK_LOOKUP_BLOB    = os.environ.get("BLOCK_LOOKUP_BLOB", "models/block_lookup.parquet")
RPI_BLOB             = os.environ.get("RPI_BLOB", "hdb_rpi.csv")

# Local dir with artifact files (development); when set, skips GCS for matching basenames.
BACKEND_ARTIFACT_DIR = os.environ.get("BACKEND_ARTIFACT_DIR", "").strip()


def _local_path(filename: str) -> Path | None:
    if not BACKEND_ARTIFACT_DIR:
        return None
    p = Path(BACKEND_ARTIFACT_DIR) / filename
    return p if p.exists() else None


def _download_blob_to_tmp(blob_path: str, suffix: str) -> str:
    client = storage.Client()
    blob = client.bucket(GCS_BUCKET).blob(blob_path)
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    blob.download_to_filename(tmp_path)
    return tmp_path


@lru_cache(maxsize=1)
def load_model() -> CatBoostRegressor:
    logger.info("Loading CatBoost v2 model...")
    lp = _local_path(MODEL_BLOB.rsplit("/", 1)[-1])
    if lp:
        tmp_path = str(lp)
        delete_after = False
    else:
        tmp_path = _download_blob_to_tmp(MODEL_BLOB, ".cbm")
        delete_after = True
    model = CatBoostRegressor()
    model.load_model(tmp_path)
    if delete_after:
        os.unlink(tmp_path)
    logger.info("model_catboost_v2 loaded ✅")
    return model


@lru_cache(maxsize=1)
def load_inference_metrics() -> dict:
    fname = METRICS_BLOB.rsplit("/", 1)[-1]
    lp = _local_path(fname)
    if lp:
        data = json.loads(lp.read_text(encoding="utf-8"))
    else:
        client = storage.Client()
        raw = client.bucket(GCS_BUCKET).blob(METRICS_BLOB).download_as_bytes()
        data = json.loads(raw.decode("utf-8"))
    logger.info("metrics_catboost_v2 loaded (%s features)", len(data.get("features", [])))
    return data


@lru_cache(maxsize=1)
def load_spatial_bundle() -> dict:
    fname = SPATIAL_BLOB.rsplit("/", 1)[-1]
    lp = _local_path(fname)
    if lp:
        bundle = joblib.load(lp)
    else:
        client = storage.Client()
        buf = io.BytesIO()
        client.bucket(GCS_BUCKET).blob(SPATIAL_BLOB).download_to_file(buf)
        buf.seek(0)
        bundle = joblib.load(buf)
    logger.info("spatial_inference bundle loaded ✅")
    return bundle


@lru_cache(maxsize=1)
def load_block_lookup() -> pd.DataFrame:
    fname = BLOCK_LOOKUP_BLOB.rsplit("/", 1)[-1]
    lp = _local_path(fname)
    if lp:
        df = pd.read_parquet(lp)
    else:
        client = storage.Client()
        buf = io.BytesIO()
        client.bucket(GCS_BUCKET).blob(BLOCK_LOOKUP_BLOB).download_to_file(buf)
        buf.seek(0)
        df = pd.read_parquet(buf)
    df["town"] = df["town"].astype(str).str.strip().str.upper()
    df["block"] = df["block"].astype(str).str.strip().str.upper()
    df = df.drop_duplicates(["town", "block"]).set_index(["town", "block"], verify_integrity=False)
    logger.info("block_lookup loaded — %s rows ✅", len(df))
    return df


@lru_cache(maxsize=1)
def load_rpi_quarters_df() -> pd.DataFrame:
    """Official HDB RPI quarters — same resolution path as other artifacts (local env → BACKEND_ARTIFACT_DIR → GCS)."""
    explicit = os.environ.get("HDB_RPI_PATH", "").strip()
    if explicit:
        ep = Path(explicit)
        if ep.is_file():
            df = pd.read_csv(ep)
            df = df[["year", "quarter", "rpi"]].copy()
            logger.info("hdb_rpi loaded from HDB_RPI_PATH (%s quarters)", len(df))
            return df

    fname = RPI_BLOB.rsplit("/", 1)[-1]
    lp = _local_path(fname)
    if lp:
        df = pd.read_csv(lp)
        df = df[["year", "quarter", "rpi"]].copy()
        logger.info("hdb_rpi loaded from BACKEND_ARTIFACT_DIR (%s quarters)", len(df))
        return df

    client = storage.Client()
    raw = client.bucket(GCS_BUCKET).blob(RPI_BLOB).download_as_bytes()
    df = pd.read_csv(io.BytesIO(raw))[["year", "quarter", "rpi"]].copy()
    logger.info("hdb_rpi loaded from gs://%s/%s (%s quarters)", GCS_BUCKET, RPI_BLOB, len(df))
    return df
