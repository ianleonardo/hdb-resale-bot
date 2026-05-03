import os
import logging
import httpx
import google.auth.transport.requests
import google.oauth2.id_token

logger = logging.getLogger(__name__)
BACKEND_URL = os.environ.get("BACKEND_URL", "")
# Cold-start / heavy inference can exceed a short client timeout (Telegram webhook tolerates ~60s total).
BACKEND_TIMEOUT_S = float(os.environ.get("BACKEND_TIMEOUT_SECONDS", "45"))


def _get_identity_token(audience: str) -> str:
    """Fetch a Google OIDC identity token for Cloud Run to Cloud Run auth."""
    auth_req = google.auth.transport.requests.Request()
    return google.oauth2.id_token.fetch_id_token(auth_req, audience)


async def call_predict(params: dict) -> dict:
    """POST collected params to hdb-backend /predict with identity token."""
    token = _get_identity_token(BACKEND_URL)
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=BACKEND_TIMEOUT_S) as client:
        payload = {k: v for k, v in params.items() if v is not None}
        resp = await client.post(f"{BACKEND_URL}/predict", json=payload, headers=headers)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 422:
                logger.error(
                    "Backend 422 Unprocessable: body=%s detail=%s",
                    payload,
                    e.response.text[:2000],
                )
            raise
        return resp.json()
