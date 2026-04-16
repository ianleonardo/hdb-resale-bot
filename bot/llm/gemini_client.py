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
            response_mime_type="application/json",
        ),
        safety_settings=[
            {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ],
    )
