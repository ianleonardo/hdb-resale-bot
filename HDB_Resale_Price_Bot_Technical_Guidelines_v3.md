# Technical Implementation Guidelines

# HDB SG Resale Price Estimation — Telegram Bot (v3.0 — GCP Native)

> **Version:** 3.0  
> **Date:** April 2026  
> **Scope:** End-to-end design for a Telegram chatbot with Gemini-powered natural Singlish conversation, topic guardrails, ML-powered price estimation — fully deployed on **Google Cloud Run** with **Google Cloud Storage** for artifact persistence and **Python in-process cache** for session state.
>
> **What changed from v2.0**:
>
> - 🔄 **LLM**: Anthropic Claude → **Google Gemini** (`gemini-2.0-flash`)
> - 🔄 **Session cache**: Redis → **Python in-process TTL cache** (`cachetools`)
> - 🔄 **Storage**: PostgreSQL + Docker volumes → **Google Cloud Storage (GCS)** bucket
> - 🔄 **Deployment**: Docker Compose / VPS → **Google Cloud Run** (fully managed, serverless)
> - ❌ **Removed**: Redis, PostgreSQL, Kubernetes, Docker Compose

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Design](#2-architecture-design)
3. [LLM Conversation Engine — Gemini](#3-llm-conversation-engine--gemini)
4. [Singlish Persona & Tone Design](#4-singlish-persona--tone-design)
5. [Topic Guardrail Design](#5-topic-guardrail-design)
6. [Telegram Bot — Integration Layer](#6-telegram-bot--integration-layer)
7. [Python In-Process Session Cache](#7-python-in-process-session-cache)
8. [Backend API Service](#8-backend-api-service)
9. [Machine Learning Model](#9-machine-learning-model)
10. [Data Preprocessing Pipeline](#10-data-preprocessing-pipeline)
11. [ML Algorithm Proposals](#11-ml-algorithm-proposals)
12. [Model Training & Evaluation](#12-model-training--evaluation)
13. [Feature Engineering](#13-feature-engineering)
14. [Google Cloud Storage — Artifact & Log Management](#14-google-cloud-storage--artifact--log-management)
15. [Google Cloud Run — Deployment](#15-google-cloud-run--deployment)
16. [API Contract](#16-api-contract)
17. [Environment & Configuration](#17-environment--configuration)
18. [Tech Stack Summary](#18-tech-stack-summary)
19. [Project Structure](#19-project-structure)
20. [Development Roadmap](#20-development-roadmap)

---

## 1. System Overview

### 1.1 Architecture Overview

```
[Telegram User — types anything, Singlish or English]
      │
      ▼
[Cloud Run: Telegram Bot Service]     ← python-telegram-bot v21 + webhook
      │  session lookup from Python TTL Cache (in-process, cachetools)
      ▼
[Cloud Run: LLM Conversation Engine]  ← Google Gemini 2.0 Flash (google-genai SDK)
      │  - Uncle HDB persona, Singlish style
      │  - Extracts 8 flat parameters from free-form chat
      │  - Enforces HDB resale topic guardrail
      │  - Returns structured JSON + Singlish reply
      ▼
[Cloud Run: Backend FastAPI Service]  ← FastAPI + Pydantic v2
      │  - Feature engineering
      │  - ML model inference (model loaded from GCS on startup)
      ▼
[ML Inference Engine]                 ← LightGBM / CatBoost (model.pkl from GCS)
      │
      ▼
[LLM formats result in Singlish → back to user via Telegram]
```

### 1.2 Deployment Topology

All three services are deployed as **independent Cloud Run services**:

```
GCP Project
│
├── Cloud Run: hdb-bot          (Telegram webhook handler + LLM engine)
├── Cloud Run: hdb-backend      (FastAPI ML inference service)
│
└── Cloud Storage: hdb-resale-artifacts/
        ├── models/
        │   ├── model.pkl
        │   └── preprocessor.pkl
        ├── logs/
        │   └── predictions/YYYY/MM/DD/predictions_*.jsonl
        └── training/
            └── hdb_resale_2017_2026.csv
```

### 1.3 Design Principles (v3.0)

- **Serverless-first**: Cloud Run scales to zero; no idle compute cost.
- **Stateless services**: All session state is held in Python in-process TTL cache. Cloud Run instances are sticky per chat via Telegram webhook routing (single instance per bot service recommended for MVP; see Section 7.4 for scaling notes).
- **No database**: Prediction logs are streamed as JSONL files to GCS bucket.
- **No Redis**: `cachetools.TTLCache` handles conversation sessions in memory.
- **Single artifact store**: GCS bucket is the single source of truth for model files and logs.

---

## 2. Architecture Design

### 2.1 Detailed Component Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            TELEGRAM PLATFORM                             │
│  User ──► Telegram Servers ──► HTTPS Webhook POST to Cloud Run           │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │ message.text + chat_id
┌──────────────────────────────────▼───────────────────────────────────────┐
│              Cloud Run: hdb-bot  (min-instances: 1)                      │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  Telegram Message Router                                           │  │
│  │  ① Lookup ConversationState from TTLCache[chat_id]                 │  │
│  │  ② Quick topic pre-filter (keyword check)                          │  │
│  │  ③ Call Gemini LLM Engine with history + collected_params          │  │
│  │  ④ Parse LLM JSON response                                         │  │
│  │  ⑤ Merge new params into TTLCache session                          │  │
│  │  ⑥ If ready_to_predict → call hdb-backend /predict                 │  │
│  │  ⑦ LLM formats result → send to Telegram                           │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  Python TTLCache  (cachetools.TTLCache, maxsize=500, ttl=3600)     │  │
│  │  Key: chat_id  │  Value: { history[], collected_params{} }         │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │ POST /predict (internal HTTPS)
┌──────────────────────────────────▼───────────────────────────────────────┐
│              Cloud Run: hdb-backend  (min-instances: 1)                  │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  FastAPI  /predict                                                  │  │
│  │  ① Pydantic validation                                              │  │
│  │  ② Feature engineering                                              │  │
│  │  ③ Model inference (model loaded from GCS at cold start)            │  │
│  │  ④ lru_cache: model/preprocessor cached in-process after load      │  │
│  │  ⑤ Stream prediction log → GCS bucket (async, fire-and-forget)     │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────────────┐
│        GCS Bucket: hdb-resale-artifacts                                  │
│   models/model.pkl  ·  models/preprocessor.pkl  ·  logs/*.jsonl          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Communication Flow

```
1.  Telegram sends webhook POST → hdb-bot Cloud Run
2.  Bot looks up session from TTLCache (chat_id key)
3.  Quick keyword pre-filter: off-topic? → LLM handles redirect
4.  Bot calls Gemini with: system_prompt + conversation history + user message
5.  Gemini returns JSON: { reply, extracted_params, ready_to_predict, off_topic }
6.  Bot merges extracted_params into TTLCache session (nulls don't overwrite)
7.  Bot appends turn to conversation history in TTLCache
8.  Bot sends Gemini reply text to Telegram user
9.  If ready_to_predict=true:
      a. Bot calls POST https://hdb-backend.run.app/predict
      b. Backend runs inference, streams log to GCS
      c. Bot calls Gemini again to format result in Singlish
      d. Bot sends formatted result to user, clears session from cache
```

---

## 3. LLM Conversation Engine — Gemini

### 3.1 LLM Choice


| Attribute         | Choice                                           |
| ----------------- | ------------------------------------------------ |
| Provider          | Google AI                                        |
| Model             | `gemini-2.0-flash`                               |
| SDK               | `google-genai` (Python)                          |
| Mode              | Multi-turn chat with injected system instruction |
| Output format     | Structured JSON + natural Singlish reply         |
| Max output tokens | 1024 per turn                                    |
| Temperature       | 0.4                                              |


> **Why Gemini 2.0 Flash?** Fast, cost-effective, strong instruction following, excellent JSON output reliability, and native Google Cloud integration (Vertex AI option available for enterprise). Flash tier keeps per-session LLM cost under $0.001.

### 3.2 Gemini SDK Setup

```python
# bot/llm/gemini_client.py
import os
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def get_gemini_model() -> genai.GenerativeModel:
    """Create Gemini model with safety settings tuned for HDB chat."""
    return genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=genai.GenerationConfig(
            temperature=0.4,
            max_output_tokens=1024,
            response_mime_type="application/json",  # Force JSON output mode
        ),
        safety_settings=[
            {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
    )
```

### 3.3 System Prompt Design

```python
# bot/llm/system_prompt.py
from bot.constants import VALID_TOWNS, VALID_FLAT_TYPES, VALID_FLAT_MODELS

def build_system_prompt(collected_params: dict) -> str:
    missing = [k for k, v in collected_params.items() if v is None]
    collected_display = "\n".join(
        f"  - {k}: {v}" for k, v in collected_params.items() if v is not None
    ) or "  (nothing collected yet)"

    return f"""
You are Uncle HDB — a friendly, warm Singaporean housing kakak/uncle chatbot on Telegram.
Your ONLY job is to help people estimate HDB resale flat prices in Singapore.
You speak natural Singlish: casual, warm, local — like a helpful kampung neighbour, not a government officer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎭 PERSONA & TONE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Speak natural Singlish. Sprinkle (don't spam): lah, lor, leh, ah, wah, aiyo, aiyah,
  sia, can, liddat, steady, walao, shiok, blur, confirm plus chop, one (at end of sentence).
- Short punchy sentences. 2–3 lines per paragraph. Use newlines liberally.
- Acknowledge what user said BEFORE asking next question.
- Warm reactions to context: "Wah Bishan ah! Quite atas area leh!"
- If user corrects themselves: "Ohhh ok ok, I update for you! 👍"
- Never say "I am an AI" or "as a language model". You are Uncle HDB.
- Max 2 emojis per message. 1 emoji on follow-up turns.
- Never robotic. Not "Input received: TAMPINES". Instead: "Ah Tampines, noted!"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚫 TOPIC GUARDRAIL — STRICT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You ONLY discuss HDB resale flat prices in Singapore. Nothing else.
Topics: HDB flat valuation, resale price estimation, town comparison, flat types, storey, lease, floor area.

If user asks ANYTHING off-topic (food, weather, politics, coding, BTO, condo, private property, crypto, etc.),
redirect warmly but firmly. NEVER answer off-topic content, even partially.

Redirect examples:
- "Alamak, that one outside my lane leh 😄 I only know HDB resale prices one. Which flat you want to check ah?"
- "Aiyoh food question I blur lah! I'm only expert in HDB lor. So back to your flat — which town?"
- "Wah condo ah? That one different story leh. I only do HDB resale. Got HDB flat to check anot?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 8 PARAMETERS TO COLLECT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You need ALL 8 before estimating:

1. town               Valid: {', '.join(VALID_TOWNS)}
2. flat_type          Valid: {', '.join(VALID_FLAT_TYPES)}
3. flat_model         Valid: {', '.join(VALID_FLAT_MODELS)}
4. storey_range       Format "NN TO NN". Infer from natural language:
                      "around 8th floor" → "07 TO 09", "high floor ~20" → "19 TO 21"
5. floor_area_sqm     Float, 20–300. Parse "~93sqm", "about 90 square meters" → float
6. remaining_lease_years  Float (decimal years). Parse:
                      "61 years 4 months" → 61.33, "about 60 years" → 60.0, "60 over years" → 60.5
7. street_name        Free text. e.g. "TAMPINES ST 42"
8. block              Alphanumeric. e.g. "456B"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📦 COLLECTED SO FAR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{collected_display}

Still missing: {', '.join(missing) if missing else '✅ ALL COLLECTED — ready to predict!'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧠 EXTRACTION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Extract ALL params mentioned in ONE message — user may give several facts at once.
- Validate against valid lists. If ambiguous (e.g. "bukit" → multiple towns), ask to clarify.
- flat_type: "4-room","4room","four room","4RM" → "4 ROOM"
- town: "tampines","TPE area","near tampines MRT" → "TAMPINES"
- street_name given → try to infer town if not stated.
- NEVER assume or guess values you are not confident about. Ask instead.
- If user corrects a param, update it; don't re-ask already-confirmed values.
- Ask ONLY for missing params. Group multiple missing fields into one natural question.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📤 OUTPUT FORMAT — MANDATORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALWAYS respond with ONLY this JSON. No text outside it.

{{
  "reply": "<Singlish reply to send to user>",
  "extracted_params": {{
    "town": "<value or null>",
    "flat_type": "<value or null>",
    "flat_model": "<value or null>",
    "storey_range": "<value or null>",
    "floor_area_sqm": <number or null>,
    "remaining_lease_years": <number or null>,
    "street_name": "<value or null>",
    "block": "<value or null>"
  }},
  "ready_to_predict": <true or false>,
  "off_topic": <true or false>
}}

Rules:
- extracted_params: ONLY values extracted from THIS turn. null = not mentioned this turn.
- ready_to_predict: true ONLY when ALL 8 params are confirmed across all turns.
- off_topic: true when message is unrelated to HDB resale prices.
- reply: warm, natural Singlish. What the user sees.
- No markdown fences, no extra keys, no text outside the JSON.
"""
```

### 3.4 LLM Engine — Core Logic

```python
# bot/llm/engine.py
import json, re, logging
import google.generativeai as genai
from bot.llm.gemini_client import get_gemini_model
from bot.llm.system_prompt import build_system_prompt

logger = logging.getLogger(__name__)


async def process_message(
    user_message: str,
    conversation_history: list[dict],
    collected_params: dict,
) -> dict:
    """
    Send user message to Gemini with full context.
    Returns parsed dict: { reply, extracted_params, ready_to_predict, off_topic }
    """
    model = get_gemini_model()
    system_instruction = build_system_prompt(collected_params)

    # Build Gemini contents: alternate user/model turns
    contents = []
    for turn in conversation_history[-20:]:  # last 20 turns
        role = "user" if turn["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": turn["content"]}]})
    contents.append({"role": "user", "parts": [{"text": user_message}]})

    try:
        response = model.generate_content(
            contents=contents,
            generation_config=genai.GenerationConfig(
                temperature=0.4,
                max_output_tokens=1024,
                response_mime_type="application/json",
                system_instruction=system_instruction,
            ),
        )
        raw = response.text.strip()
        return _parse_response(raw)

    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return {
            "reply": "Aiyoh, something snagged on my end leh 😅 Can try again in a bit?",
            "extracted_params": {},
            "ready_to_predict": False,
            "off_topic": False,
        }


def _parse_response(raw: str) -> dict:
    """Parse JSON from Gemini response with fallback."""
    clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        data = json.loads(clean)
        data.setdefault("reply", "Hmm, say that again ah? I blur a bit 😅")
        data.setdefault("extracted_params", {})
        data.setdefault("ready_to_predict", False)
        data.setdefault("off_topic", False)
        return data
    except json.JSONDecodeError as exc:
        logger.warning(f"JSON parse failed: {exc} | raw={raw[:200]}")
        return {
            "reply": "Eh sorry, I got confused just now 😅 Can repeat what you said?",
            "extracted_params": {},
            "ready_to_predict": False,
            "off_topic": False,
        }


async def format_result_singlish(prediction: dict, params: dict) -> str:
    """Use Gemini to craft a warm Singlish result message."""
    model = get_gemini_model()
    prompt = f"""
You are Uncle HDB. A user just got their HDB resale price estimate.
Write a warm, natural Singlish reply (5–8 lines) that:
- Recaps the key flat details (town, flat type, storey, floor area)
- Highlights the predicted price in bold Markdown (*SGD X*)
- Mentions the low–high range casually
- Ends with a friendly closing line

Flat details: {json.dumps(params, indent=2)}
Prediction: {json.dumps(prediction, indent=2)}

Return ONLY the plain Singlish text (Markdown ok). No JSON. No preamble.
"""
    resp = model.generate_content(prompt)
    return resp.text.strip()
```

### 3.5 Conversation Examples

#### Multi-param extraction in one message

```
User:   "I want check price for 4-room flat at Tampines St 42, block 456B,
         around 8th floor, got about 61 years lease left"

Gemini extracts:
  flat_type=4 ROOM, street_name=TAMPINES ST 42, block=456B,
  storey_range=07 TO 09, remaining_lease_years=61.0
  (town inferred from street → TAMPINES)

Reply:  "Wah, you gave me a lot of info in one shot — steady lah! 👍
         Got Tampines, 4-room, Block 456B at Tampines St 42,
         around 7th–9th floor, 61 years lease. Noted!

         Two more things I need —
         what's the flat model (like Model A, Standard, DBSS etc.)?
         And the floor area in sqm?"
```

#### User correction mid-conversation

```
User:   "Actually wait, it's 5-room lah, I made mistake"

Gemini extracts: flat_type=5 ROOM (overwrites previous 4 ROOM)

Reply:  "Ohhh ok ok, 5-room noted, I update already! 😄
         So which flat model ah?"
```

#### Off-topic redirect

```
User:   "Eh what's good to eat in Tampines?"

Gemini: off_topic=true

Reply:  "Hahaha wah food question ah! Aiyoh I only expert in
         HDB prices lah, not food critic one 😄
         So back to your flat — which flat model is it?"
```

#### Ambiguous categorical

```
User:   "My flat model is the normal one"

Reply:  "Hmm 'normal one' got a few possibilities leh 😅
         Is it Standard, Model A, or Improved?
         These three are the most common 'normal' ones."
```

---

## 4. Singlish Persona & Tone Design

### 4.1 Singlish Vocabulary Reference


| Expression          | Meaning / Use Case                                               |
| ------------------- | ---------------------------------------------------------------- |
| `lah`               | Softener / affirmation — "Ok lah", "Can lah"                     |
| `lor`               | Resigned / explanatory — "Like that lor", "Just check lor"       |
| `leh`               | Mild surprise / emphasis — "Expensive leh!", "I dunno leh"       |
| `ah`                | Seeking confirmation / softener — "Which town ah?", "5-room ah?" |
| `sia`               | Exclamation of surprise — "700k sia!"                            |
| `wah`               | Exclamation — "Wah, nice area!"                                  |
| `aiyo` / `aiyah`    | Mild facepalm — "Aiyo, wrong floor lah"                          |
| `can`               | Affirmative — "Can, no problem"                                  |
| `liddat`            | "Like that" — "If liddat, different price"                       |
| `steady`            | Cool / solid — "Steady, good choice"                             |
| `confirm plus chop` | Absolutely certain — "Confirm plus chop worth it"                |
| `walao`             | Strong surprise — "Walao, that quite high leh!"                  |
| `shiok`             | Something great — "Shiok, good price!"                           |
| `blur`              | Confused — "Sorry, I blur a bit"                                 |
| `one`               | Emphasis (end of sentence) — "Confirm expensive one"             |
| `never mind lah`    | Reassurance — "Never mind lah, I help you"                       |
| `kakak` / `uncle`   | Friendly self-reference                                          |


### 4.2 Tone Rules

- **Short paragraphs**: 2–3 sentences, then newline.
- **Acknowledge first**: Comment on what user said before asking next question.
- **Warm, never sarcastic**: Even when redirecting off-topic messages.
- **No robotic phrasing**: Never "Parameter stored", "Value received", etc.
- **Emoji discipline**: 1 max on follow-up turns, 2 max on greetings/results.
- **Never repeat** the same particle (`lah`) in consecutive sentences.

### 4.3 Welcome Message Template

```
/start or /estimate:

"Eh hello! 👋 I'm Uncle HDB — your kakak for checking HDB resale prices in SG!

Just tell me about the flat lor. Can be one shot or slowly slowly, up to you.
Like: 'Looking at a 4-room in Tampines, around 8th floor' — liddat also can.

So, which flat you want to check ah? 🏠"
```

---

## 5. Topic Guardrail Design

### 5.1 Two-Layer Guardrail


| Layer       | Mechanism                                    | Purpose                                                      |
| ----------- | -------------------------------------------- | ------------------------------------------------------------ |
| **Layer 1** | Keyword pre-filter (`guards/topic_check.py`) | Fast O(1) check before LLM call — saves cost on obvious spam |
| **Layer 2** | Gemini system prompt instructions            | Nuanced handling of ambiguous / partially on-topic messages  |


### 5.2 Keyword Pre-Filter

```python
# bot/guards/topic_check.py

_OFF_TOPIC = {
    # food & lifestyle
    "recipe","chicken rice","bubble tea","hawker","makan","restaurant",
    # tech / coding
    "python","javascript","html","coding","code","chatgpt","openai",
    # finance (non-property)
    "crypto","bitcoin","stock market","shares","forex","etf",
    # politics
    "pap","election","parliament","minister","opposition",
    # other property types
    "condo","condominium","bto","ec property","private property","landed",
    "penthouse","villa","serviced apartment",
    # misc
    "weather","football","soccer","movie","song","recipe",
    "relationship","girlfriend","boyfriend",
}

_HDB_KEYWORDS = {
    "hdb","flat","resale","room","storey","floor","sqm","lease",
    "block","street","town","price","estimate","ang mo kio","bedok",
    "bishan","bukit","central","choa chu kang","clementi","geylang",
    "hougang","jurong","kallang","marine parade","pasir ris","punggol",
    "queenstown","sembawang","sengkang","serangoon","tampines",
    "toa payoh","woodlands","yishun","whampoa",
    "model a","dbss","maisonette","executive","standard","improved",
}

def quick_topic_check(message: str) -> str:
    """
    Returns: 'hdb' | 'off_topic' | 'ambiguous'
    'ambiguous' means pass to LLM for nuanced decision.
    """
    lower = message.lower()
    tokens = set(lower.split())

    has_hdb = bool(_HDB_KEYWORDS & tokens) or any(kw in lower for kw in _HDB_KEYWORDS)
    has_off = bool(_OFF_TOPIC & tokens)   or any(kw in lower for kw in _OFF_TOPIC)

    if has_hdb and not has_off:
        return "hdb"
    if has_off and not has_hdb:
        return "off_topic"
    return "ambiguous"
```

### 5.3 Allowed vs. Redirected Topics


| Topic                                | Handling                                                          |
| ------------------------------------ | ----------------------------------------------------------------- |
| ✅ HDB resale price queries           | Full assistance                                                   |
| ✅ Town / flat type / model questions | Answer + collect param                                            |
| ✅ How accurate is the estimate?      | Brief honest answer, then continue                                |
| ✅ What factors affect resale price?  | 2-sentence answer, redirect to collection                         |
| ⚠️ BTO / new launch                  | "BTO different story leh, I only do resale. Back to your flat..." |
| ⚠️ Condo / private property          | "Condo I cannot help leh, I'm HDB resale specialist lor 😄"       |
| ❌ Food, weather, politics, coding    | Warm Singlish redirect                                            |
| ❌ Pure chitchat                      | One-liner acknowledgement, steer back                             |


---

## 6. Telegram Bot — Integration Layer

### 6.1 Technology


| Component        | Choice                             |
| ---------------- | ---------------------------------- |
| Language         | Python 3.12                        |
| Telegram library | `python-telegram-bot` v21+         |
| LLM              | `google-generativeai` SDK          |
| Session cache    | `cachetools.TTLCache` (in-process) |
| HTTP client      | `httpx` (async)                    |
| Web server       | `uvicorn` (for webhook endpoint)   |


### 6.2 Main Bot Application

```python
# bot/main.py
import os, logging, asyncio
import httpx
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters
)
from bot.llm.engine import process_message, format_result_singlish
from bot.cache.session import SessionCache
from bot.guards.topic_check import quick_topic_check

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

session_cache = SessionCache()          # In-process TTLCache
BACKEND_URL   = os.environ["BACKEND_URL"]  # Cloud Run hdb-backend URL


async def handle_message(update: Update, context) -> None:
    chat_id  = update.effective_chat.id
    user_msg = update.message.text.strip()

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    # Layer 1 fast topic check — obvious off-topic caught cheaply
    topic = quick_topic_check(user_msg)

    state = session_cache.get(chat_id)

    # Call Gemini (handles both on-topic extraction and off-topic redirects)
    llm_result = await process_message(
        user_message=user_msg,
        conversation_history=state["history"],
        collected_params=state["collected_params"],
    )

    # Update session
    session_cache.append_history(chat_id, "user",      user_msg)
    session_cache.append_history(chat_id, "assistant", llm_result["reply"])
    if llm_result.get("extracted_params"):
        session_cache.merge_params(chat_id, llm_result["extracted_params"])

    # Send Gemini reply
    await update.message.reply_text(llm_result["reply"], parse_mode="Markdown")

    # Trigger ML prediction when all params collected
    if llm_result.get("ready_to_predict"):
        updated_state = session_cache.get(chat_id)
        if session_cache.is_complete(chat_id):
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.post(
                        f"{BACKEND_URL}/predict",
                        json=updated_state["collected_params"],
                    )
                    prediction = resp.json()

                singlish_reply = await format_result_singlish(
                    prediction, updated_state["collected_params"]
                )
                await update.message.reply_text(singlish_reply, parse_mode="Markdown")

            except Exception as exc:
                logger.error(f"Prediction call failed: {exc}")
                await update.message.reply_text(
                    "Aiyoh, something went wrong when I try to calculate leh 😅\n"
                    "Try again? Type /estimate to start fresh."
                )
            finally:
                session_cache.clear(chat_id)


async def cmd_start(update: Update, context) -> None:
    session_cache.clear(update.effective_chat.id)
    await update.message.reply_text(
        "Eh hello! 👋 I'm Uncle HDB — your kakak for checking HDB resale prices in SG!\n\n"
        "Just tell me about the flat lor — which area, what type, high or low floor, liddat. "
        "I'll figure out the rest.\n\n"
        "So, what flat you want to check ah? 🏠",
        parse_mode="Markdown",
    )


async def cmd_cancel(update: Update, context) -> None:
    session_cache.clear(update.effective_chat.id)
    await update.message.reply_text(
        "Ok lor, I clear everything already 👌\n"
        "Whenever ready, just /estimate and we start fresh can!"
    )


async def cmd_help(update: Update, context) -> None:
    await update.message.reply_text(
        "🏠 *Uncle HDB Help*\n\n"
        "Just chat with me about the HDB flat you want to check!\n"
        "Tell me the town, flat type, model, floor, area, lease, street and block.\n"
        "Can give all at once or one by one, up to you lor.\n\n"
        "Commands:\n"
        "/estimate — Start a new price check\n"
        "/cancel — Clear and start over\n"
        "/help — Show this message",
        parse_mode="Markdown",
    )


def main():
    app = Application.builder().token(os.environ["TELEGRAM_BOT_TOKEN"]).build()
    app.add_handler(CommandHandler("start",    cmd_start))
    app.add_handler(CommandHandler("estimate", cmd_start))
    app.add_handler(CommandHandler("cancel",   cmd_cancel))
    app.add_handler(CommandHandler("help",     cmd_help))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Cloud Run: use webhook, not polling
    port = int(os.environ.get("PORT", 8080))
    app.run_webhook(
        listen="0.0.0.0",
        port=port,
        webhook_url=os.environ["WEBHOOK_URL"],
        secret_token=os.environ.get("WEBHOOK_SECRET", ""),
    )


if __name__ == "__main__":
    main()
```

---

## 7. Python In-Process Session Cache

### 7.1 Design Rationale

Cloud Run is used with `**min-instances: 1**` for the bot service. This keeps one warm instance alive at all times, making in-process `TTLCache` a viable session store for MVP scale (hundreds of concurrent users). No Redis, no external dependency, no network hop.

> ⚠️ **Scaling note**: If you ever scale hdb-bot to `min-instances > 1` or `max-instances > 1`, sessions won't be shared across instances. For that scenario, either use Cloud Memorystore (managed Redis) or route Telegram traffic to a single instance via a session-affinity load balancer. For MVP (1 instance), TTLCache is perfectly sufficient.

### 7.2 SessionCache Implementation

```python
# bot/cache/session.py
import threading
from cachetools import TTLCache

DEFAULT_PARAMS = {
    "town": None, "flat_type": None, "flat_model": None,
    "storey_range": None, "floor_area_sqm": None,
    "remaining_lease_years": None, "street_name": None, "block": None,
}

class SessionCache:
    """
    Thread-safe in-process TTL cache for Telegram conversation sessions.
    maxsize=500: supports ~500 concurrent active sessions.
    ttl=3600:    sessions expire after 1 hour of inactivity.
    """

    def __init__(self, maxsize: int = 500, ttl: int = 3600):
        self._cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self._lock  = threading.Lock()

    def _default_state(self) -> dict:
        return {
            "history":          [],
            "collected_params": dict(DEFAULT_PARAMS),
        }

    def get(self, chat_id: int) -> dict:
        with self._lock:
            if chat_id not in self._cache:
                self._cache[chat_id] = self._default_state()
            return self._cache[chat_id]

    def clear(self, chat_id: int) -> None:
        with self._lock:
            self._cache.pop(chat_id, None)

    def append_history(self, chat_id: int, role: str, content: str) -> None:
        with self._lock:
            state = self._cache.setdefault(chat_id, self._default_state())
            state["history"].append({"role": role, "content": content})
            # Keep last 30 turns to cap memory usage
            state["history"] = state["history"][-30:]

    def merge_params(self, chat_id: int, new_params: dict) -> None:
        """Merge extracted params — null values do NOT overwrite existing values."""
        with self._lock:
            state = self._cache.setdefault(chat_id, self._default_state())
            for key, val in new_params.items():
                if val is not None and key in state["collected_params"]:
                    state["collected_params"][key] = val

    def is_complete(self, chat_id: int) -> bool:
        with self._lock:
            state = self._cache.get(chat_id, self._default_state())
            return all(v is not None for v in state["collected_params"].values())
```

### 7.3 Model Artifact Cache (Backend)

The backend uses `functools.lru_cache` to load model artifacts from GCS **once per process lifetime** — subsequent prediction calls reuse the in-memory objects with zero I/O.

```python
# backend/app/model_loader.py
import joblib, io, logging
from functools import lru_cache
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
```

### 7.4 Cache Sizing Guidelines


| Concurrent Sessions | Recommended `maxsize` | Avg Memory |
| ------------------- | --------------------- | ---------- |
| < 100               | 200                   | ~20 MB     |
| 100 – 500           | 500 (default)         | ~50 MB     |
| 500 – 2,000         | 2000                  | ~200 MB    |
| > 2,000             | Use Cloud Memorystore | N/A        |


---

## 8. Backend API Service

### 8.1 FastAPI Application

```python
# backend/app/main.py
import os, json, logging
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import numpy as np, pandas as pd
from google.cloud import storage
from backend.app.model_loader import load_model, load_preprocessor

logging.basicConfig(level=logging.INFO)
logger  = logging.getLogger(__name__)
app     = FastAPI(title="HDB Resale Price Estimator", version="3.0.0")
GCS_BUCKET = os.environ.get("GCS_BUCKET", "hdb-resale-artifacts")


class PredictRequest(BaseModel):
    town:                   str
    flat_type:              str
    flat_model:             str
    storey_range:           str           # "07 TO 09"
    floor_area_sqm:         float = Field(ge=20.0,  le=300.0)
    remaining_lease_years:  float = Field(ge=0.0,   le=99.0)
    street_name:            str
    block:                  str

    @field_validator("storey_range")
    @classmethod
    def validate_storey(cls, v: str) -> str:
        import re
        if not re.match(r"^\d{2} TO \d{2}$", v.strip()):
            raise ValueError("storey_range must match 'NN TO NN'")
        return v.strip().upper()

    @field_validator("town", "flat_type", "flat_model")
    @classmethod
    def uppercase_str(cls, v: str) -> str:
        return v.strip().upper()


class PredictResponse(BaseModel):
    predicted_price: float
    price_range:     dict
    confidence:      str
    model_version:   str
    input_echo:      dict


@app.on_event("startup")
async def startup():
    """Warm up model cache on container startup."""
    load_model()
    load_preprocessor()
    logger.info("Model and preprocessor loaded from GCS ✅")


@app.get("/health")
async def health():
    return {"status": "ok", "model_version": "3.0.0"}


@app.get("/meta")
async def meta():
    from backend.app.constants import VALID_TOWNS, VALID_FLAT_TYPES, VALID_FLAT_MODELS
    return {
        "towns":       VALID_TOWNS,
        "flat_types":  VALID_FLAT_TYPES,
        "flat_models": VALID_FLAT_MODELS,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    model        = load_model()
    preprocessor = load_preprocessor()

    parts  = req.storey_range.split(" TO ")
    storey_mid = (int(parts[0]) + int(parts[1])) / 2

    df = pd.DataFrame([{
        "town":                   req.town,
        "flat_type":              req.flat_type,
        "flat_model":             req.flat_model,
        "storey_midpoint":        storey_mid,
        "floor_area_sqm":         req.floor_area_sqm,
        "remaining_lease_years":  req.remaining_lease_years,
        "street_name":            req.street_name,
        "block":                  req.block,
        "transaction_year":       datetime.now(timezone.utc).year,
        "transaction_month":      datetime.now(timezone.utc).month,
    }])

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
        model_version="3.0.0",
        input_echo=req.model_dump(),
    )

    # Fire-and-forget: log to GCS (don't block response)
    import asyncio
    asyncio.create_task(_log_prediction_to_gcs(req, result))

    return result


async def _log_prediction_to_gcs(req: PredictRequest, result: PredictResponse):
    """Append prediction as JSONL to GCS — async, non-blocking."""
    try:
        now    = datetime.now(timezone.utc)
        record = {
            "timestamp":       now.isoformat(),
            "input":           req.model_dump(),
            "predicted_price": result.predicted_price,
            "price_range":     result.price_range,
        }
        path   = f"logs/predictions/{now.year}/{now.month:02d}/{now.day:02d}/pred_{now.timestamp():.0f}.json"
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        bucket.blob(path).upload_from_string(
            json.dumps(record) + "\n", content_type="application/json"
        )
    except Exception as exc:
        logger.warning(f"GCS log write failed (non-critical): {exc}")
```

---

## 9. Machine Learning Model

### 9.1 Input Features


| Feature             | Source Column         | Type             | Notes             |
| ------------------- | --------------------- | ---------------- | ----------------- |
| Town                | `town`                | Categorical      | 26 unique values  |
| Flat Type           | `flat_type`           | Categorical      | 7 values, ordinal |
| Flat Model          | `flat_model`          | Categorical      | 21 values         |
| Storey Midpoint     | `storey_range`        | Numerical        | `(low+high)/2`    |
| Floor Area          | `floor_area_sqm`      | Numerical        | 20–300 sqm        |
| Remaining Lease     | `remaining_lease`     | Numerical        | Decimal years     |
| Lease Commence Date | `lease_commence_date` | Numerical        | Year integer      |
| Transaction Year    | `month`               | Temporal         | `dt.year`         |
| Transaction Month   | `month`               | Temporal         | `dt.month`        |
| Street Name         | `street_name`         | High-cardinality | Target encoding   |
| Block               | `block`               | High-cardinality | Target encoding   |


**Target**: `resale_price` → `np.log1p` transformed during training; `np.expm1` at inference.

---

## 10. Data Preprocessing Pipeline

```python
# training/preprocessing.py
import pandas as pd, numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from category_encoders import TargetEncoder

ORDINAL_FLAT_TYPE = ['1 ROOM','2 ROOM','3 ROOM','4 ROOM',
                     '5 ROOM','EXECUTIVE','MULTI-GENERATION']

NUMERIC_FEATURES  = ['floor_area_sqm','storey_midpoint','remaining_lease_years',
                     'lease_commence_date','transaction_year','transaction_month']
ORDINAL_FEATURES  = ['flat_type']
NOMINAL_FEATURES  = ['town','flat_model']
HIGH_CARD         = ['street_name','block']


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ('num', StandardScaler(),
         NUMERIC_FEATURES),
        ('ord', OrdinalEncoder(categories=[ORDINAL_FLAT_TYPE]),
         ORDINAL_FEATURES),
        ('nom', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
         NOMINAL_FEATURES),
        ('hc',  TargetEncoder(smoothing=10),
         HIGH_CARD),
    ])


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['transaction_year']      = pd.to_datetime(df['month']).dt.year
    df['transaction_month']     = pd.to_datetime(df['month']).dt.month
    df['storey_midpoint']       = df['storey_range'].str.extract(
        r'(\d+) TO (\d+)').astype(float).mean(axis=1)
    y  = df['remaining_lease'].str.extract(r'(\d+)\s*year').fillna(0).astype(float)[0]
    m  = df['remaining_lease'].str.extract(r'(\d+)\s*month').fillna(0).astype(float)[0]
    df['remaining_lease_years'] = y + m / 12
    df['log_resale_price']      = np.log1p(df['resale_price'])
    return df
```

### Time-Based Split


| Split      | Period            | Share |
| ---------- | ----------------- | ----- |
| Train      | 2017-01 – 2024-12 | ~85%  |
| Validation | 2025-01 – 2025-09 | ~10%  |
| Test       | 2025-10 – 2026-03 | ~5%   |


> ⚠️ Always use **time-based splits** — random splits cause data leakage on price-trend features.

---

## 11. ML Algorithm Proposals

### 11.1 Recommended Models

#### 🥇 LightGBM *(Primary Recommendation)*

```python
import lightgbm as lgb

model = lgb.LGBMRegressor(
    n_estimators=1000, learning_rate=0.05, num_leaves=63,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
```

**Strengths**: Fastest, handles mixed types, built-in feature importance, minimal memory footprint on Cloud Run.

#### 🥈 XGBoost

```python
import xgboost as xgb

model = xgb.XGBRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='rmse', early_stopping_rounds=50, random_state=42
)
```

#### 🥉 CatBoost *(Best for raw categoricals)*

```python
from catboost import CatBoostRegressor

model = CatBoostRegressor(
    iterations=1000, learning_rate=0.05, depth=8,
    cat_features=['town','flat_type','flat_model','street_name','block'],
    loss_function='RMSE', early_stopping_rounds=50
)
```

#### 🔬 Stacking Ensemble *(Highest accuracy, highest complexity)*

```python
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

stack = StackingRegressor(
    estimators=[
        ('lgbm', lgb.LGBMRegressor(...)),
        ('xgb',  xgb.XGBRegressor(...)),
        ('cat',  CatBoostRegressor(...)),
    ],
    final_estimator=Ridge(alpha=1.0),
    cv=5
)
```

### 11.2 Model Comparison Matrix


| Model    | Accuracy | Train Speed | Inference Speed | Categorical Support | Interpretability |
| -------- | -------- | ----------- | --------------- | ------------------- | ---------------- |
| LightGBM | ⭐⭐⭐⭐⭐    | ⭐⭐⭐⭐⭐       | ⭐⭐⭐⭐⭐           | ⭐⭐⭐⭐                | ⭐⭐⭐⭐             |
| XGBoost  | ⭐⭐⭐⭐⭐    | ⭐⭐⭐⭐        | ⭐⭐⭐⭐⭐           | ⭐⭐⭐                 | ⭐⭐⭐⭐             |
| CatBoost | ⭐⭐⭐⭐⭐    | ⭐⭐⭐         | ⭐⭐⭐⭐            | ⭐⭐⭐⭐⭐               | ⭐⭐⭐              |
| Stacking | ⭐⭐⭐⭐⭐    | ⭐⭐          | ⭐⭐⭐             | ⭐⭐⭐⭐                | ⭐⭐               |


---

## 12. Model Training & Evaluation

### 12.1 Training Script

```python
# training/train.py
import pandas as pd, numpy as np, joblib, json, io
from pathlib import Path
from datetime import datetime
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from google.cloud import storage
from training.preprocessing import build_preprocessor, engineer_features

GCS_BUCKET   = "hdb-resale-artifacts"
DATA_BLOB    = "training/hdb_resale_2017_2026.csv"
LOCAL_ARTIFACTS = Path("artifacts/")
LOCAL_ARTIFACTS.mkdir(exist_ok=True)

# --- Load data from GCS ---
gcs = storage.Client()
csv_bytes = gcs.bucket(GCS_BUCKET).blob(DATA_BLOB).download_as_bytes()
df = pd.read_csv(io.BytesIO(csv_bytes))
df = engineer_features(df)

# --- Time-based split ---
train_df = df[df['transaction_year'] <= 2024]
val_df   = df[(df['transaction_year'] == 2025) & (df['transaction_month'] <= 9)]
test_df  = df[df['transaction_year'] >= 2026]

FEATURES = [
    'town','flat_type','flat_model','storey_midpoint','floor_area_sqm',
    'remaining_lease_years','lease_commence_date',
    'transaction_year','transaction_month','street_name','block'
]
TARGET = 'log_resale_price'

preprocessor = build_preprocessor()
X_train = preprocessor.fit_transform(train_df[FEATURES], train_df[TARGET])
X_val   = preprocessor.transform(val_df[FEATURES])
X_test  = preprocessor.transform(test_df[FEATURES])
y_train, y_val, y_test = train_df[TARGET], val_df[TARGET], test_df[TARGET]

# --- Train ---
model = lgb.LGBMRegressor(
    n_estimators=2000, learning_rate=0.03, num_leaves=127,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
)

# --- Evaluate ---
def evaluate(name: str, X, y_log):
    pred = np.expm1(model.predict(X))
    true = np.expm1(y_log)
    mae  = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    mape = np.mean(np.abs((true - pred) / true)) * 100
    r2   = r2_score(true, pred)
    print(f"{name}: MAE={mae:,.0f} | RMSE={rmse:,.0f} | MAPE={mape:.2f}% | R²={r2:.4f}")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

metrics = {
    "trained_at":  datetime.utcnow().isoformat(),
    "validation":  evaluate("Validation", X_val,  y_val),
    "test":        evaluate("Test",        X_test, y_test),
}

# --- Save locally then upload to GCS ---
joblib.dump(model,        LOCAL_ARTIFACTS / "model.pkl")
joblib.dump(preprocessor, LOCAL_ARTIFACTS / "preprocessor.pkl")
json.dump(metrics,        open(LOCAL_ARTIFACTS / "metrics.json", "w"), indent=2)

bucket = gcs.bucket(GCS_BUCKET)
for fname in ["model.pkl", "preprocessor.pkl", "metrics.json"]:
    bucket.blob(f"models/{fname}").upload_from_filename(str(LOCAL_ARTIFACTS / fname))
    print(f"Uploaded {fname} → gs://{GCS_BUCKET}/models/{fname}")
```

### 12.2 Target Metrics


| Metric | Target       |
| ------ | ------------ |
| MAE    | < SGD 25,000 |
| RMSE   | < SGD 40,000 |
| MAPE   | < 5%         |
| R²     | > 0.96       |


### 12.3 Hyperparameter Tuning (Optuna)

```python
import optuna

def objective(trial):
    params = {
        "num_leaves":        trial.suggest_int("num_leaves", 31, 255),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }
    m = lgb.LGBMRegressor(n_estimators=500, **params, random_state=42)
    m.fit(X_train, y_train, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(30)])
    return mean_absolute_error(np.expm1(y_val), np.expm1(m.predict(X_val)))

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
print("Best params:", study.best_params)
```

---

## 13. Feature Engineering


| Derived Feature         | Source                | Formula                                  |
| ----------------------- | --------------------- | ---------------------------------------- |
| `storey_midpoint`       | `storey_range`        | `(low + high) / 2`                       |
| `remaining_lease_years` | `remaining_lease`     | `years + months/12`                      |
| `flat_age`              | `lease_commence_date` | `transaction_year - lease_commence_date` |
| `transaction_year`      | `month`               | `pd.to_datetime().dt.year`               |
| `transaction_month`     | `month`               | `pd.to_datetime().dt.month`              |
| `transaction_quarter`   | `month`               | `pd.to_datetime().dt.quarter`            |


### Geospatial Enrichment *(Phase 2)*

```python
import requests

def geocode_hdb(block: str, street: str) -> dict:
    resp = requests.get(
        "https://www.onemap.gov.sg/api/common/elastic/search",
        params={"searchVal": f"{block} {street} SINGAPORE",
                "returnGeom": "Y", "getAddrDetails": "Y"}
    )
    results = resp.json().get("results", [])
    if results:
        return {"lat": float(results[0]["LATITUDE"]),
                "lng": float(results[0]["LONGITUDE"])}
    return {}
```

---

## 14. Google Cloud Storage — Artifact & Log Management

### 14.1 Bucket Structure

```
gs://hdb-resale-artifacts/
│
├── models/
│   ├── model.pkl              ← LightGBM trained model (latest)
│   ├── preprocessor.pkl       ← sklearn ColumnTransformer (latest)
│   └── metrics.json           ← Latest training metrics
│
├── training/
│   └── hdb_resale_2017_2026.csv  ← Source training data
│
└── logs/
    └── predictions/
        └── YYYY/MM/DD/
            └── pred_{timestamp}.json   ← One JSON file per prediction
```

### 14.2 GCS Access Pattern


| Service                      | Access                    | Method                         |
| ---------------------------- | ------------------------- | ------------------------------ |
| `hdb-backend` (Cloud Run)    | Read `models/`            | On cold start via `lru_cache`  |
| `hdb-backend` (Cloud Run)    | Write `logs/predictions/` | Fire-and-forget async          |
| `training/` (local or Colab) | Read `training/*.csv`     | Load data for training         |
| `training/` (local or Colab) | Write `models/`           | Upload artifacts post-training |


### 14.3 GCS Client Helper

```python
# shared/gcs_client.py
import io, joblib, json
from google.cloud import storage
from functools import lru_cache

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
```

### 14.4 IAM Permissions Required


| Service Account      | Role                                | Purpose                              |
| -------------------- | ----------------------------------- | ------------------------------------ |
| `hdb-backend@...`    | `roles/storage.objectAdmin`         | Read models from GCS, write logs     |
| `hdb-backend@...`    | `roles/secretmanager.secretAccessor`| (optional) future secret use         |
| `hdb-bot@...`        | `roles/secretmanager.secretAccessor`| Read Telegram token, Gemini key      |
| `github-actions@...` | `roles/run.admin`                   | Deploy Cloud Run services            |
| `github-actions@...` | `roles/artifactregistry.writer`     | Push Docker images to GCR            |
| `github-actions@...` | `roles/iam.serviceAccountUser`      | Act as hdb-bot and hdb-backend SAs   |
| `github-actions@...` | `roles/viewer`                      | Stream Cloud Build / Run logs        |


---

## 15. Google Cloud Run — Deployment

### 15.1 Service Configuration

Both services are deployed as separate Cloud Run services in the same GCP project.

#### hdb-bot (Telegram Bot + LLM Engine)

```yaml
# cloud-run/hdb-bot.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: hdb-bot
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "1"   # Keep warm — session cache must persist
        autoscaling.knative.dev/maxScale: "1"   # Single instance for MVP (see Section 7.1)
        run.googleapis.com/execution-environment: gen2
    spec:
      serviceAccountName: hdb-bot@PROJECT_ID.iam.gserviceaccount.com
      timeoutSeconds: 30
      containers:
        - image: gcr.io/PROJECT_ID/hdb-bot:latest
          resources:
            limits:
              cpu: "1"
              memory: "512Mi"
          env:
            - name: TELEGRAM_BOT_TOKEN
              valueFrom:
                secretKeyRef:
                  name: telegram-bot-token
                  key: latest
            - name: GEMINI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: gemini-api-key
                  key: latest
            - name: BACKEND_URL
              value: "https://hdb-backend-xxxx-as.a.run.app"
            - name: WEBHOOK_URL
              value: "https://hdb-bot-xxxx-as.a.run.app/webhook"
            - name: WEBHOOK_SECRET
              valueFrom:
                secretKeyRef:
                  name: webhook-secret
                  key: latest
```

#### hdb-backend (FastAPI ML Service)

```yaml
# cloud-run/hdb-backend.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: hdb-backend
  annotations:
    run.googleapis.com/ingress: internal   # Only accessible from hdb-bot
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "1"
        autoscaling.knative.dev/maxScale: "3"
        run.googleapis.com/execution-environment: gen2
    spec:
      serviceAccountName: hdb-backend@PROJECT_ID.iam.gserviceaccount.com
      timeoutSeconds: 30
      containers:
        - image: gcr.io/PROJECT_ID/hdb-backend:latest
          resources:
            limits:
              cpu: "2"
              memory: "1Gi"     # LightGBM model may be ~100–300 MB in memory
          env:
            - name: GCS_BUCKET
              value: "hdb-resale-artifacts"
```

### 15.2 Dockerfiles

```dockerfile
# bot/Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["python", "main.py"]
```

```dockerfile
# backend/Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["gunicorn", "app.main:app", \
     "-w", "2", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8080", \
     "--timeout", "60", \
     "--preload"]
```

### 15.3 Deployment Commands (gcloud CLI)

```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="asia-southeast1"

# ── Authenticate Docker to GCR ───────────────────────────────────
gcloud auth configure-docker --quiet

# ── Build & push images ──────────────────────────────────────────
docker build -t gcr.io/$PROJECT_ID/hdb-backend:latest ./backend
docker push gcr.io/$PROJECT_ID/hdb-backend:latest

docker build -t gcr.io/$PROJECT_ID/hdb-bot:latest ./bot
docker push gcr.io/$PROJECT_ID/hdb-bot:latest

# ── Deploy hdb-backend ───────────────────────────────────────────
gcloud run deploy hdb-backend --image gcr.io/$PROJECT_ID/hdb-backend:latest --region $REGION --service-account hdb-backend@$PROJECT_ID.iam.gserviceaccount.com --ingress internal --min-instances 1 --max-instances 3 --memory 1Gi --cpu 2 --timeout 30 --set-env-vars GCS_BUCKET=hdb-resale-artifacts,MODEL_VERSION=3.0.0,LOG_LEVEL=INFO --no-allow-unauthenticated --quiet

# Get backend URL
BACKEND_URL=$(gcloud run services describe hdb-backend --region $REGION --format='value(status.url)')

# ── Deploy hdb-bot ───────────────────────────────────────────────
gcloud run deploy hdb-bot --image gcr.io/$PROJECT_ID/hdb-bot:latest --region $REGION --service-account hdb-bot@$PROJECT_ID.iam.gserviceaccount.com --ingress all --min-instances 1 --max-instances 1 --memory 512Mi --cpu 1 --timeout 30 --set-secrets TELEGRAM_BOT_TOKEN=telegram-bot-token:latest,GEMINI_API_KEY=gemini-api-key:latest,WEBHOOK_SECRET=webhook-secret:latest --set-env-vars "BACKEND_URL=$BACKEND_URL,WEBHOOK_URL=https://hdb-bot-xxxx-as.a.run.app/webhook,LOG_LEVEL=INFO" --allow-unauthenticated --quiet

# Get bot URL and register Telegram webhook
BOT_URL=$(gcloud run services describe hdb-bot --region $REGION --format='value(status.url)')

curl "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/setWebhook" -d "url=${BOT_URL}/webhook" -d "secret_token=${WEBHOOK_SECRET}"
```

> ⚠️ Run gcloud CLI commands as single lines (no backslash continuation) to avoid shell parsing errors on zsh.

### 15.4 CI/CD Pipeline (GitHub Actions)

Both jobs run in parallel on every push to `main`. Docker images are built on the GitHub Actions runner directly — this avoids Cloud Build log-streaming permission issues.

```yaml
# .github/workflows/deploy.yml
name: Deploy to Cloud Run

on:
  push:
    branches: [main]

env:
  PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
  REGION: asia-southeast1

jobs:
  deploy-backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}
      - uses: google-github-actions/setup-gcloud@v2
      - name: Configure Docker for GCR
        run: gcloud auth configure-docker --quiet
      - name: Build and push backend image
        run: |
          docker build -t gcr.io/$PROJECT_ID/hdb-backend:${{ github.sha }} ./backend
          docker push gcr.io/$PROJECT_ID/hdb-backend:${{ github.sha }}
      - name: Deploy backend to Cloud Run
        run: |
          gcloud run deploy hdb-backend \
            --image gcr.io/$PROJECT_ID/hdb-backend:${{ github.sha }} \
            --region $REGION \
            --service-account hdb-backend@$PROJECT_ID.iam.gserviceaccount.com \
            --set-env-vars GCS_BUCKET=hdb-resale-artifacts,MODEL_VERSION=3.0.0,LOG_LEVEL=INFO \
            --quiet

  deploy-bot:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}
      - uses: google-github-actions/setup-gcloud@v2
      - name: Configure Docker for GCR
        run: gcloud auth configure-docker --quiet
      - name: Build and push bot image
        run: |
          docker build -t gcr.io/$PROJECT_ID/hdb-bot:${{ github.sha }} ./bot
          docker push gcr.io/$PROJECT_ID/hdb-bot:${{ github.sha }}
      - name: Deploy bot to Cloud Run
        run: |
          gcloud run deploy hdb-bot \
            --image gcr.io/$PROJECT_ID/hdb-bot:${{ github.sha }} \
            --region $REGION \
            --service-account hdb-bot@$PROJECT_ID.iam.gserviceaccount.com \
            --set-secrets TELEGRAM_BOT_TOKEN=telegram-bot-token:latest,GEMINI_API_KEY=gemini-api-key:latest,WEBHOOK_SECRET=webhook-secret:latest \
            --set-env-vars BACKEND_URL=${{ secrets.BACKEND_URL }},WEBHOOK_URL=${{ secrets.WEBHOOK_URL }},LOG_LEVEL=INFO \
            --quiet
```

**Required GitHub repository secrets** (Settings → Secrets and variables → Actions):

| Secret name | Value |
| ----------- | ----- |
| `GCP_PROJECT_ID` | GCP project ID string |
| `GCP_SA_KEY` | Full JSON content of the `github-actions` service account key |
| `BACKEND_URL` | Cloud Run URL of `hdb-backend` |
| `WEBHOOK_URL` | Cloud Run URL of `hdb-bot` + `/webhook` |

### 15.5 Full GCP Setup — One-Time Steps

Run all commands in order before the first deployment. All `gcloud` commands must be run as single lines on zsh to avoid shell parsing errors.

#### Step 1 — Set project ID

```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="asia-southeast1"
```

#### Step 2 — Create service accounts

```bash
gcloud iam service-accounts create hdb-bot --display-name="HDB Bot service account"

gcloud iam service-accounts create hdb-backend --display-name="HDB Backend service account"

gcloud iam service-accounts create github-actions --display-name="GitHub Actions deployer"
```

#### Step 3 — Grant IAM roles to hdb-backend

```bash
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:hdb-backend@$PROJECT_ID.iam.gserviceaccount.com" --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:hdb-backend@$PROJECT_ID.iam.gserviceaccount.com" --role="roles/secretmanager.secretAccessor"
```

#### Step 4 — Grant IAM roles to github-actions

```bash
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" --role="roles/run.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" --role="roles/viewer"
```

#### Step 5 — Allow github-actions to act as the service accounts

```bash
gcloud iam service-accounts add-iam-policy-binding hdb-bot@$PROJECT_ID.iam.gserviceaccount.com --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" --role="roles/iam.serviceAccountUser"

gcloud iam service-accounts add-iam-policy-binding hdb-backend@$PROJECT_ID.iam.gserviceaccount.com --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" --role="roles/iam.serviceAccountUser"
```

#### Step 6 — Create secrets in Secret Manager

Generate a random webhook secret:
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

Create the secrets (replace values with your actual tokens):
```bash
echo -n "your-telegram-bot-token" | gcloud secrets create telegram-bot-token --data-file=- --replication-policy=automatic

echo -n "your-gemini-api-key" | gcloud secrets create gemini-api-key --data-file=- --replication-policy=automatic

echo -n "your-webhook-secret" | gcloud secrets create webhook-secret --data-file=- --replication-policy=automatic
```

#### Step 7 — Grant hdb-bot access to secrets

```bash
gcloud secrets add-iam-policy-binding telegram-bot-token --member="serviceAccount:hdb-bot@$PROJECT_ID.iam.gserviceaccount.com" --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding gemini-api-key --member="serviceAccount:hdb-bot@$PROJECT_ID.iam.gserviceaccount.com" --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding webhook-secret --member="serviceAccount:hdb-bot@$PROJECT_ID.iam.gserviceaccount.com" --role="roles/secretmanager.secretAccessor"
```

#### Step 8 — Create github-actions key and add to GitHub

```bash
gcloud iam service-accounts keys create gha-key.json --iam-account=github-actions@$PROJECT_ID.iam.gserviceaccount.com
```

Go to your repo → **Settings → Secrets and variables → Actions → New repository secret** and add:

| Secret name | Value |
| ----------- | ----- |
| `GCP_PROJECT_ID` | Your GCP project ID string |
| `GCP_SA_KEY` | Full contents of `gha-key.json` |
| `BACKEND_URL` | Cloud Run URL of `hdb-backend` (get after first deploy) |
| `WEBHOOK_URL` | Cloud Run URL of `hdb-bot` + `/webhook` (get after first deploy) |

```bash
# Delete the local key file immediately after
rm gha-key.json
```

> ⚠️ Never commit `gha-key.json`. Verify it is listed in `.gitignore`.

---

### 15.6 Estimated Monthly Cost (GCP)


| Resource                                     | Usage Estimate                 | Monthly Cost (SGD) |
| -------------------------------------------- | ------------------------------ | ------------------ |
| Cloud Run — hdb-bot (1 instance, 0.5 vCPU)   | Always-on                      | ~$8                |
| Cloud Run — hdb-backend (1 instance, 1 vCPU) | Always-on                      | ~$15               |
| Gemini 2.0 Flash API                         | ~1,000 sessions/mo × 4K tokens | ~$1–2              |
| GCS Storage                                  | ~200 MB artifacts + logs       | ~$0.10             |
| Secret Manager                               | 3 secrets                      | ~$0.10             |
| Cloud Build                                  | ~50 builds/mo                  | ~$0 (free tier)    |
| **Total**                                    |                                | **~$25–30/mo**     |


---

## 16. API Contract

### 16.1 Bot ↔ LLM (Gemini)

```
Input:  system_prompt (with collected_params injected)
        + conversation_history (last 20 turns)
        + current user message

Output JSON:
{
  "reply":            string,    // Singlish message sent to user
  "extracted_params": {          // Newly extracted values; null = not mentioned
    "town":                  string | null,
    "flat_type":             string | null,
    "flat_model":            string | null,
    "storey_range":          string | null,
    "floor_area_sqm":        number | null,
    "remaining_lease_years": number | null,
    "street_name":           string | null,
    "block":                 string | null
  },
  "ready_to_predict": boolean,   // true only when ALL 8 params confirmed
  "off_topic":        boolean    // true when message is unrelated to HDB
}
```

### 16.2 Bot → Backend `POST /predict`


| Field                   | Type   | Validation                        |
| ----------------------- | ------ | --------------------------------- |
| `town`                  | string | One of 26 valid towns (uppercase) |
| `flat_type`             | string | One of 7 valid types              |
| `flat_model`            | string | One of 21 valid models            |
| `storey_range`          | string | Format `NN TO NN`                 |
| `floor_area_sqm`        | float  | 20.0 – 300.0                      |
| `remaining_lease_years` | float  | 0.0 – 99.0                        |
| `street_name`           | string | Non-empty                         |
| `block`                 | string | Alphanumeric                      |


### 16.3 Backend Response `200 OK`

```json
{
  "predicted_price": 650000,
  "price_range":     { "low": 617000, "high": 683000 },
  "confidence":      "medium",
  "model_version":   "3.0.0",
  "input_echo":      { "town": "TAMPINES", ... }
}
```

### 16.4 HTTP Status Codes


| Code                        | Meaning                       |
| --------------------------- | ----------------------------- |
| `200 OK`                    | Prediction successful         |
| `422 Unprocessable Entity`  | Pydantic validation failure   |
| `500 Internal Server Error` | Model inference error         |
| `503 Service Unavailable`   | Model not yet loaded from GCS |


---

## 17. Environment & Configuration

### 17.1 Environment Variables

```env
# hdb-bot (Cloud Run env + Secret Manager)
TELEGRAM_BOT_TOKEN=<from Secret Manager>
GEMINI_API_KEY=<from Secret Manager>
BACKEND_URL=https://hdb-backend-xxxx-as.a.run.app
WEBHOOK_URL=https://hdb-bot-xxxx-as.a.run.app/webhook
WEBHOOK_SECRET=<from Secret Manager>
SESSION_MAXSIZE=500
SESSION_TTL=3600
LOG_LEVEL=INFO
PORT=8080

# hdb-backend (Cloud Run env)
GCS_BUCKET=hdb-resale-artifacts
MODEL_VERSION=3.0.0
LOG_LEVEL=INFO
PORT=8080
```

### 17.2 Secrets Management (GCP Secret Manager)

All sensitive values are stored in **GCP Secret Manager** and injected at runtime via Cloud Run's secret binding — never baked into Docker images or `.env` files.

```bash
# Create secrets
gcloud secrets create telegram-bot-token --replication-policy=automatic
gcloud secrets create gemini-api-key     --replication-policy=automatic
gcloud secrets create webhook-secret     --replication-policy=automatic

# Grant Cloud Run service account access
gcloud secrets add-iam-policy-binding telegram-bot-token \
  --member="serviceAccount:hdb-bot@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

---

## 18. Tech Stack Summary

```
┌──────────────────────────────────────────────────────────────────┐
│                    FULL TECH STACK (v3.0)                        │
├──────────────────────┬───────────────────────────────────────────┤
│ Layer                │ Technology                                 │
├──────────────────────┼───────────────────────────────────────────┤
│ Bot Interface        │ Python 3.12, python-telegram-bot v21       │
│ LLM Engine           │ Google Gemini 2.0 Flash (google-genai SDK) │
│ Singlish Prompts     │ Dynamic system prompt injection             │
│ Session Cache        │ cachetools.TTLCache (in-process, maxsize=500, ttl=3600) │
│ Model Artifact Cache │ functools.lru_cache (in-process, load-once) │
│ Backend API          │ FastAPI 0.111+, Uvicorn, Gunicorn           │
│ API Validation       │ Pydantic v2                                 │
│ ML Training          │ scikit-learn, LightGBM, XGBoost, CatBoost  │
│ HPO                  │ Optuna                                      │
│ Feature Encoding     │ category_encoders (TargetEncoder)           │
│ Serialization        │ joblib                                      │
│ Data Wrangling       │ pandas, numpy                               │
│ Artifact Storage     │ Google Cloud Storage (GCS)                  │
│ Prediction Logs      │ GCS JSONL files (no DB needed)              │
│ Secrets              │ GCP Secret Manager                          │
│ Container Registry   │ Google Container Registry (GCR)             │
│ Deployment           │ Google Cloud Run (serverless)               │
│ CI/CD                │ GitHub Actions + gcloud CLI                 │
│ Monitoring           │ Cloud Run built-in metrics + Cloud Logging  │
└──────────────────────┴───────────────────────────────────────────┘
```

---

## 19. Project Structure

```
hdb-resale-bot/
│
├── bot/                              # Cloud Run: hdb-bot
│   ├── main.py                       # Entry point — webhook + message router
│   ├── llm/
│   │   ├── engine.py                 # Gemini API client, process_message()
│   │   ├── gemini_client.py          # GenerativeModel config + safety settings
│   │   └── system_prompt.py          # Dynamic Singlish system prompt builder
│   ├── cache/
│   │   └── session.py                # SessionCache (cachetools.TTLCache)
│   ├── services/
│   │   └── backend_client.py         # httpx async client → /predict
│   ├── guards/
│   │   └── topic_check.py            # Keyword pre-filter guardrail
│   ├── constants.py                  # VALID_TOWNS, FLAT_TYPES, FLAT_MODELS
│   ├── requirements.txt
│   └── Dockerfile
│
├── backend/                          # Cloud Run: hdb-backend
│   ├── app/
│   │   ├── main.py                   # FastAPI app, /predict /health /meta
│   │   ├── schemas.py                # Pydantic request/response models
│   │   ├── model_loader.py           # GCS download + lru_cache
│   │   ├── preprocessing.py          # Feature engineering at inference
│   │   └── constants.py              # Shared domain constants
│   ├── requirements.txt
│   └── Dockerfile
│
├── training/                         # Run locally or on Colab / Vertex AI
│   ├── train.py                      # Training pipeline (reads/writes GCS)
│   ├── preprocessing.py              # Feature engineering (shared with backend)
│   ├── evaluate.py                   # Metrics + SHAP analysis
│   ├── hpo.py                        # Optuna HPO
│   └── notebooks/
│       ├── 01_eda.ipynb
│       ├── 02_feature_engineering.ipynb
│       └── 03_model_selection.ipynb
│
├── shared/
│   └── gcs_client.py                 # Shared GCS upload/download helpers
│
├── cloud-run/
│   ├── hdb-bot.yaml                  # Cloud Run service spec
│   └── hdb-backend.yaml              # Cloud Run service spec
│
├── .github/
│   └── workflows/
│       └── deploy.yml                # GitHub Actions CI/CD
│
├── .env.example                      # Example env vars (no secrets)
├── .gitignore
└── README.md
```

---

## 20. Development Roadmap

### Phase 1 — GCP-Native MVP (Weeks 1–4)

- Set up GCP project, service accounts, GCS bucket structure
- Configure GCP Secret Manager with Telegram + Gemini keys
- Implement SessionCache with `cachetools.TTLCache`
- Build LLM engine with Gemini 2.0 Flash + Singlish system prompt
- Implement topic guardrail (keyword pre-filter + Gemini enforcement)
- Build FastAPI backend with GCS model loading + `lru_cache`
- Train baseline LightGBM, upload artifacts to GCS
- Build Docker images, deploy both services to Cloud Run
- Register Telegram webhook → Cloud Run URL
- End-to-end integration test: chat → Gemini → backend → Singlish reply

### Phase 2 — Quality & Accuracy (Weeks 5–8)

- Optuna HPO for LightGBM; benchmark CatBoost
- Add geospatial features via OneMap API geocoding
- Gemini prompt refinement — more Singlish variety, edge-case handling
- Prediction JSONL logs analysis in GCS (Google Colab / BigQuery)
- `/history` — summarise user's past queries (read from GCS logs)

### Phase 3 — Production Hardening (Weeks 9–12)

- Cloud Run request rate limiting (per Telegram user_id)
- Model versioning in GCS (`models/v1/`, `models/v2/`) + version env var
- Cloud Monitoring dashboard (latency, error rate, Gemini token cost)
- Cloud Logging structured JSON logs for all prediction events
- Automated retraining trigger (Cloud Scheduler → Cloud Build → GCS upload)
- Model drift detection (compare monthly MAPE on new transactions)
- Graceful fallback if Gemini API down → simple keyword FSM mode

### Phase 4 — Advanced (Future)

- SHAP explainability: "Wah why so expensive? Because high floor + near MRT lor"
- Comparable sales: "Here got 3 similar flats sold recently" (query GCS logs)
- Price trend sparkline chart per town (matplotlib → Telegram image)
- Mandarin / Malay code-switching (rojak style bilingual Singlish)
- Vertex AI Model Registry for governed model lifecycle management

---

*End of Technical Implementation Guidelines — v3.0*

---

> **Data Source**: HDB resale transaction data is publicly available at [data.gov.sg](https://data.gov.sg/). Comply with the Singapore Government Open Data Licence in production.
>
> **GCP Region**: Deploy to `asia-southeast1` (Singapore) for lowest latency to local users and data residency compliance.
>
> **Gemini Cost Estimate**: ~~4,000 tokens/session × 1,000 sessions/month = ~4M tokens/mo. At Gemini 2.0 Flash pricing (~~$0.075/1M input tokens), cost ≈ **< $1/month** for moderate usage.

