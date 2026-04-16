import asyncio
import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
from fastapi import FastAPI, HTTPException
from google.cloud import storage

from app.model_loader import load_model, load_preprocessor
from app.preprocessing import build_inference_dataframe
from app.schemas import PredictRequest, PredictResponse

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(title="HDB Resale Price Estimator", version="3.0.0")
GCS_BUCKET = os.environ.get("GCS_BUCKET", "hdb-resale-artifacts")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "3.0.0")


@app.on_event("startup")
async def startup():
    """Attempt to warm up model cache on startup. Non-fatal if artifacts not yet in GCS."""
    try:
        load_model()
        load_preprocessor()
        logger.info("Model and preprocessor loaded from GCS ✅")
    except Exception as exc:
        logger.warning(f"Model not loaded at startup (will retry on first request): {exc}")


@app.get("/health")
async def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.get("/meta")
async def meta():
    from app.constants import VALID_TOWNS, VALID_FLAT_TYPES, VALID_FLAT_MODELS
    return {
        "towns":       VALID_TOWNS,
        "flat_types":  VALID_FLAT_TYPES,
        "flat_models": VALID_FLAT_MODELS,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        model        = load_model()
        preprocessor = load_preprocessor()
    except Exception as exc:
        logger.error(f"Model unavailable: {exc}")
        raise HTTPException(status_code=503, detail="Model not yet available. Run training pipeline first.")

    df        = build_inference_dataframe(req)
    features  = preprocessor.transform(df)
    log_pred  = model.predict(features)[0]
    price     = float(np.expm1(log_pred))
    margin    = price * 0.05

    result = PredictResponse(
        predicted_price=round(price, -3),
        price_range={
            "low":  round(price - margin, -3),
            "high": round(price + margin, -3),
        },
        confidence="medium",
        model_version=MODEL_VERSION,
        input_echo=req.model_dump(),
    )

    asyncio.create_task(_log_prediction_to_gcs(req, result))
    return result


async def _log_prediction_to_gcs(req: PredictRequest, result: PredictResponse):
    """Append prediction as JSON to GCS — async, non-blocking."""
    try:
        now = datetime.now(timezone.utc)
        record = {
            "timestamp":       now.isoformat(),
            "input":           req.model_dump(),
            "predicted_price": result.predicted_price,
            "price_range":     result.price_range,
        }
        path = (
            f"logs/predictions/{now.year}/{now.month:02d}/{now.day:02d}"
            f"/pred_{now.timestamp():.0f}.json"
        )
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        bucket.blob(path).upload_from_string(
            json.dumps(record) + "\n", content_type="application/json"
        )
    except Exception as exc:
        logger.warning(f"GCS log write failed (non-critical): {exc}")
