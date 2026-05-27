"""
translation_service.py
Malayalam Voice AI — Translation refinement service

Uses GPT-3.5-turbo to refine the raw English output from Whisper
according to the requested style. Falls back to raw text if
OPENAI_API_KEY is not set or if the API call fails.
"""

import os
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style system prompts
# Each prompt instructs GPT how to rewrite the raw Whisper English output.
# Keep prompts concise — GPT-3.5 follows short, direct instructions well.
# ---------------------------------------------------------------------------

STYLE_PROMPTS = {
    "standard": (
        "You are a precise English editor. "
        "Fix grammar, punctuation, and fluency in the text below. "
        "Preserve the original meaning exactly. "
        "Return only the corrected text, nothing else."
    ),
    "formal": (
        "You are a professional English writer. "
        "Rewrite the text below in a formal, polished register suitable "
        "for official documents, letters, or reports. "
        "Use complete sentences and elevated vocabulary. "
        "Return only the rewritten text."
    ),
    "casual": (
        "You are a friendly editor. "
        "Rewrite the text below in a natural, conversational tone — "
        "the kind of language used between friends or in everyday chat. "
        "Keep it relaxed and easy to read. "
        "Return only the rewritten text."
    ),
    "news": (
        "You are a news journalist. "
        "Rewrite the text below in a clear, objective journalistic style — "
        "third person, active voice, concise sentences, no opinion. "
        "Suitable for a news article or press release. "
        "Return only the rewritten text."
    ),
    "literary": (
        "You are a literary author. "
        "Rewrite the text below with expressive, vivid prose — "
        "rich in imagery, rhythm, and feeling. "
        "Elevate the language while keeping the core meaning. "
        "Return only the rewritten text."
    ),
    "business": (
        "You are a business communication expert. "
        "Rewrite the text below in a professional corporate tone — "
        "clear, structured, solution-focused, and suitable for "
        "emails, memos, or business reports. "
        "Return only the rewritten text."
    ),
    "academic": (
        "You are an academic writer. "
        "Rewrite the text below in a scholarly, precise style — "
        "formal diction, technical accuracy, and logical structure. "
        "Suitable for research papers or academic presentations. "
        "Return only the rewritten text."
    ),
    "simple": (
        "You are a plain-language specialist. "
        "Rewrite the text below using very simple, short sentences "
        "and everyday words. Aim for a reading level suitable for "
        "anyone, including non-native English speakers or children. "
        "Return only the rewritten text."
    ),
    "humorous": (
        "You are a witty English writer. "
        "Rewrite the text below with a light, playful, humorous tone. "
        "Add gentle wit or wordplay where natural, but keep the meaning intact. "
        "Do not force jokes. Return only the rewritten text."
    ),
    "emotional": (
        "You are an empathetic writer. "
        "Rewrite the text below with warmth, sincerity, and emotional depth — "
        "the kind of language used in heartfelt personal messages or speeches. "
        "Return only the rewritten text."
    ),
    "bullet": (
        "You are a concise summariser. "
        "Convert the text below into a clean bullet-point list in English. "
        "Each bullet should capture one key idea. "
        "Use '• ' to start each bullet. "
        "Return only the bullet list, nothing else."
    ),
}

# Fallback if an unknown style key is received
DEFAULT_STYLE = "standard"


# ---------------------------------------------------------------------------
# Main refinement function
# ---------------------------------------------------------------------------

def refine_translation(raw_text: str, style: str = DEFAULT_STYLE) -> str:
    """
    Refine raw English text from Whisper using GPT-3.5-turbo.

    Args:
        raw_text: The raw English output from the Whisper translate pass.
        style:    One of the 11 style keys. Defaults to 'standard'.

    Returns:
        Refined text string. Returns raw_text unchanged if:
        - OPENAI_API_KEY is not set
        - The style key is unrecognised (logs a warning)
        - The GPT API call fails for any reason
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.info("OPENAI_API_KEY not set — skipping refinement, returning raw text.")
        return raw_text

    if not raw_text or not raw_text.strip():
        return raw_text

    # Resolve style → system prompt
    resolved_style = style.lower().strip() if style else DEFAULT_STYLE
    if resolved_style not in STYLE_PROMPTS:
        logger.warning(
            "Unknown style '%s' received — falling back to '%s'.",
            resolved_style,
            DEFAULT_STYLE,
        )
        resolved_style = DEFAULT_STYLE

    system_prompt = STYLE_PROMPTS[resolved_style]

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_text},
            ],
            temperature=0.4,   # low temp = consistent, faithful rewrites
            max_tokens=1024,
        )
        refined = response.choices[0].message.content.strip()
        return refined if refined else raw_text

    except Exception as exc:
        logger.error("GPT refinement failed (%s) — returning raw text.", exc)
        return raw_text


# ---------------------------------------------------------------------------
# Helper: list available styles (useful for API documentation / frontend sync)
# ---------------------------------------------------------------------------

def available_styles() -> list[str]:
    """Return the list of valid style keys."""
    return list(STYLE_PROMPTS.keys())
