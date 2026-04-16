import json
import re
import logging
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
