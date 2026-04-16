import os
import logging
import httpx

logger = logging.getLogger(__name__)
BACKEND_URL = os.environ.get("BACKEND_URL", "")


async def call_predict(params: dict) -> dict:
    """POST collected params to hdb-backend /predict. Returns parsed response dict."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(f"{BACKEND_URL}/predict", json=params)
        resp.raise_for_status()
        return resp.json()
