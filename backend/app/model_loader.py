import io
import logging
from functools import lru_cache
import joblib
from google.cloud import storage

logger = logging.getLogger(__name__)
GCS_BUCKET = "hdb-resale-artifacts"


@lru_cache(maxsize=1)
def load_model():
    """Load LightGBM model from GCS. Cached for process lifetime."""
    logger.info("Loading model from GCS...")
    return _load_pkl_from_gcs("models/model.pkl")


@lru_cache(maxsize=1)
def load_preprocessor():
    """Load sklearn preprocessor from GCS. Cached for process lifetime."""
    logger.info("Loading preprocessor from GCS...")
    return _load_pkl_from_gcs("models/preprocessor.pkl")


def _load_pkl_from_gcs(blob_path: str):
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob   = bucket.blob(blob_path)
    buf    = io.BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    return joblib.load(buf)
