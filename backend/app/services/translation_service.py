"""
translation_service.py — GPT-4o-mini English refinement
Uses OpenAI if OPENAI_API_KEY is set, otherwise returns raw Whisper output.
"""
import os

STYLE_INSTRUCTIONS = {
    "standard":      "Translate naturally and accurately. Preserve meaning exactly.",
    "formal":        "Translate in formal, polished English. Preserve meaning exactly.",
    "casual":        "Translate in casual conversational English. Preserve meaning exactly.",
    "business":      "Translate in professional business English. Preserve meaning exactly.",
    "academic":      "Translate in precise academic English. Preserve meaning exactly.",
    "news":          "Translate in concise news-report style. Preserve meaning exactly.",
    "literary":      "Translate in literary English with expressive phrasing. Preserve meaning exactly.",
    "simple":        "Translate in simple plain English anyone can understand. Preserve meaning exactly.",
    "humorous":      "Translate keeping a light humorous tone. Preserve meaning exactly.",
    "emotional":     "Translate capturing the emotional tone. Preserve meaning exactly.",
    "bullet":        "Translate as clean bullet points. Preserve all content.",
}


def refine_english(text: str, style: str = "standard", source_lang: str = "ml") -> str:
    """
    Refine English translation using GPT-4o-mini.

    - Without OPENAI_API_KEY: returns raw Whisper translation as-is.
    - With OPENAI_API_KEY: uses GPT-4o-mini for high-quality styled translation.
    """
    if not text or not text.strip():
        return ""

    clean = text.strip()
    api_key = os.environ.get("OPENAI_API_KEY", "")

    if not api_key:
        print("refine_english: no OPENAI_API_KEY — returning raw Whisper output")
        return clean

    style_instruction = STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS["standard"])

    LANGUAGE_NAMES = {
        "ml": "Malayalam", "ta": "Tamil", "te": "Telugu",
        "kn": "Kannada", "hi": "Hindi",
    }
    lang_name = LANGUAGE_NAMES.get(source_lang, "Indian language")

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are an expert translator specializing in {lang_name} to English translation. "
                        f"{style_instruction} "
                        "Return only the translated/refined English text — no explanations, no quotes."
                    ),
                },
                {"role": "user", "content": clean},
            ],
            max_tokens=800,
            temperature=0.3,
        )
        refined = response.choices[0].message.content.strip()
        print(f"GPT-4o-mini refined ({style}): {refined[:80]}")
        return refined if refined else clean
    except Exception as e:
        print(f"refine_english error ({e}) — returning raw text")
        return clean
