# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Telegram bot that estimates HDB resale prices in Singapore using a LightGBM ML model and Google Gemini for Singlish-speaking conversational UI. The full technical specification is in `HDB_Resale_Price_Bot_Technical_Guidelines_v3.md`.

**Status**: Implementation phase — source code does not yet exist; only the technical spec and training data are present.

## Planned Architecture

Two Cloud Run services communicating over HTTPS:

```
Telegram → hdb-bot (Cloud Run) → hdb-backend (Cloud Run) → GCS (model artifacts)
                ↓
           Gemini 2.0 Flash (LLM, parameter extraction + Singlish persona)
```

- **`bot/`** — Python 3.12, `python-telegram-bot` v21, webhook receiver + Gemini conversation engine. Single instance (min=max=1) because session state lives in an in-process `cachetools.TTLCache`.
- **`backend/`** — FastAPI + LightGBM inference service. Stateless, scales 1–3 instances. Loads model artifacts from GCS on startup via `functools.lru_cache`.
- **`training/`** — Offline training pipeline (LightGBM, Optuna HPO, scikit-learn preprocessing). Run locally or on Colab/Vertex AI.
- **`data/`** — HDB resale CSV datasets (time-split: train 2017–2024, val 2025-01–09, test 2025-10+).

## Key Design Decisions

### LLM Conversation Flow
The bot collects **8 required parameters** via Gemini before calling `/predict`. Gemini returns strict JSON on every turn:
```json
{ "reply": "...", "extracted_params": {...}, "ready_to_predict": false, "off_topic": false }
```
The system prompt is rebuilt each turn, injecting already-collected params and listing what's still missing.

**Topic guardrail** uses two layers: fast keyword pre-filter (`quick_topic_check()` returns `'hdb' | 'off_topic' | 'ambiguous'`), then Gemini handles ambiguous cases via the system prompt.

### ML Model
- **Algorithm**: LightGBM with scikit-learn `ColumnTransformer` (OrdinalEncoder for `flat_type`, OneHotEncoder for `town`/`flat_model`, TargetEncoder for `street_name`/`block`, StandardScaler for numerics)
- **Target**: `log1p(resale_price)` — apply `expm1` at inference
- **`storey_range`** input (e.g. `"07 TO 09"`) must be converted to numeric midpoint before model inference
- **`remaining_lease_years`** is decimal years (e.g. `61.33`), not integer months
- Target metrics: MAE < SGD 25k, MAPE < 5%, R² > 0.96

### `/predict` API Contract
```
POST /predict
{
  "town": "TAMPINES", "flat_type": "4 ROOM", "flat_model": "Model A",
  "storey_range": "07 TO 09", "floor_area_sqm": 93.5,
  "remaining_lease_years": 61.0, "street_name": "TAMPINES ST 42", "block": "456B"
}
→ { "predicted_price": 650000, "price_range": {"low": 617000, "high": 683000},
    "confidence": "medium", "model_version": "3.0.0", "input_echo": {...} }
```

## Infrastructure

- **GCP project region**: `asia-southeast1` (Singapore)
- **GCS bucket**: `hdb-resale-artifacts` — stores model `.pkl` files under `models/`, prediction logs under `logs/predictions/YYYY/MM/DD/`
- **Secrets** (GCP Secret Manager): `TELEGRAM_BOT_TOKEN`, `GEMINI_API_KEY`, `WEBHOOK_SECRET`
- **CI/CD**: GitHub Actions (`deploy.yml`) — build → push to GCR → deploy backend first, then bot

## Development Commands

No source code exists yet. Once implemented, expected commands will be:

```bash
# Install dependencies (per service)
pip install -r bot/requirements.txt
pip install -r backend/requirements.txt

# Run backend locally
cd backend && uvicorn app.main:app --reload --port 8080

# Run bot locally (requires env vars)
cd bot && python main.py

# Train model
cd training && python train.py

# Deploy via gcloud
gcloud run deploy hdb-backend --source backend/ --region asia-southeast1
gcloud run deploy hdb-bot --source bot/ --region asia-southeast1
```

## Valid Domain Values

26 HDB towns, 7 flat types (`1 ROOM` through `5 ROOM`, `EXECUTIVE`, `MULTI-GENERATION`), and 21 flat models are defined in `constants.py` (to be created). See the technical guidelines for the full canonical lists.
