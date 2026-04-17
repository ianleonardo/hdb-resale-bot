import os
import logging
import httpx
import google.auth.transport.requests
import google.oauth2.id_token

logger = logging.getLogger(__name__)
BACKEND_URL = os.environ.get("BACKEND_URL", "")


def _get_identity_token(audience: str) -> str:
    """Fetch a Google OIDC identity token for Cloud Run to Cloud Run auth."""
    auth_req = google.auth.transport.requests.Request()
    return google.oauth2.id_token.fetch_id_token(auth_req, audience)


async def call_predict(params: dict) -> dict:
    """POST collected params to hdb-backend /predict with identity token."""
    token = _get_identity_token(BACKEND_URL)
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(f"{BACKEND_URL}/predict", json=params, headers=headers)
        resp.raise_for_status()
        return resp.json()
