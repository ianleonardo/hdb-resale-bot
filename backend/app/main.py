import asyncio
import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from google.cloud import storage

from app.model_loader import (
    load_block_lookup,
    load_inference_metrics,
    load_model,
    load_rpi_quarters_df,
    load_spatial_bundle,
)
from app.preprocessing import build_inference_pool
from app.schemas import PredictRequest, PredictResponse

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(title="HDB Resale Price Estimator", version="5.0.0")
GCS_BUCKET    = os.environ.get("GCS_BUCKET", "hdb-resale-artifacts")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "catboost-v2")


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Log body shape issues — ERROR so Cloud Logging shows up next to uvicorn access lines."""
    detail = exc.errors()
    try:
        payload = json.dumps(detail, default=str)
    except TypeError:
        payload = str(detail)
    logger.error("POST /predict validation failed: %s", payload)
    return JSONResponse(status_code=422, content={"detail": detail})


@app.on_event("startup")
async def startup():
    """Warm model + v2 inference artifacts (metrics, spatial bundle, block lookup, RPI)."""
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, load_model)
        await loop.run_in_executor(None, load_inference_metrics)
        await loop.run_in_executor(None, load_spatial_bundle)
        await loop.run_in_executor(None, load_block_lookup)
        await loop.run_in_executor(None, load_rpi_quarters_df)
        logger.info("CatBoost v2 artifacts warmed ✅")
    except Exception as exc:
        logger.warning("Artifacts not loaded at startup (will retry on first request): %s", exc)


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
        model = load_model()
        pool  = build_inference_pool(req)
    except Exception as exc:
        logger.error("Prediction pipeline failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Model or inference artifacts unavailable. Run training pipeline and upload v2 artifacts.",
        ) from exc

    # Conservative haircut vs raw model output (business calibration).
    price = float(np.expm1(model.predict(pool)[0])) * 0.95
    margin = price * 0.05

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
        now    = datetime.now(timezone.utc)
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
        client.bucket(GCS_BUCKET).blob(path).upload_from_string(
            json.dumps(record) + "\n", content_type="application/json"
        )
    except Exception as exc:
        logger.warning(f"GCS log write failed (non-critical): {exc}")
