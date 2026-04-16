import io
import json
import joblib
from functools import lru_cache
from google.cloud import storage

GCS_BUCKET = "hdb-resale-artifacts"


def upload_pkl(obj, blob_path: str) -> None:
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    buf.seek(0)
    _bucket().blob(blob_path).upload_from_file(buf, content_type="application/octet-stream")


def download_pkl(blob_path: str):
    buf = io.BytesIO()
    _bucket().blob(blob_path).download_to_file(buf)
    buf.seek(0)
    return joblib.load(buf)


def upload_json(data: dict, blob_path: str) -> None:
    _bucket().blob(blob_path).upload_from_string(
        json.dumps(data, indent=2), content_type="application/json"
    )


@lru_cache(maxsize=1)
def _bucket():
    return storage.Client().bucket(GCS_BUCKET)
