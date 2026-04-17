import io
import joblib
import logging
import os
import tempfile
from functools import lru_cache

from catboost import CatBoostRegressor
from google.cloud import storage

logger = logging.getLogger(__name__)
GCS_BUCKET = "hdb-resale-artifacts"


@lru_cache(maxsize=1)
def load_model() -> CatBoostRegressor:
    """Load CatBoost model from GCS. Cached for process lifetime."""
    logger.info("Loading CatBoost model from GCS...")
    client = storage.Client()
    blob   = client.bucket(GCS_BUCKET).blob("models/model_catboost.cbm")
    # CatBoost requires a file path, not a buffer
    with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as f:
        blob.download_to_file(f)
        tmp_path = f.name
    model = CatBoostRegressor()
    model.load_model(tmp_path)
    os.unlink(tmp_path)
    logger.info("CatBoost model loaded ✅")
    return model


@lru_cache(maxsize=1)
def load_comp_lookup() -> dict:
    """Load recent-comps lookup table from GCS. Cached for process lifetime."""
    logger.info("Loading comp lookup from GCS...")
    client = storage.Client()
    buf    = io.BytesIO()
    client.bucket(GCS_BUCKET).blob("models/comp_lookup.pkl").download_to_file(buf)
    buf.seek(0)
    lookup = joblib.load(buf)
    logger.info(f"Comp lookup loaded — {len(lookup)} streets ✅")
    return lookup
