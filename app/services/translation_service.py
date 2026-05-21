from deep_translator import GoogleTranslator
import anthropic

# ========================
# CLAUDE API KEY
# ========================
# CLAUDE API KEY
# ========================
import os

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)


# ========================
# CLAUDE POST-PROCESSING
# ========================
def improve_translation_with_claude(raw_english: str) -> str:
    try:
        message = claude_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": f"""You are an expert Malayalam-English translator specializing in Kerala dialects and culture.

Speech was recorded and auto-translated from Malayalam. Fix this translation considering:

KERALA DIALECT RULES:
- Malappuram dialect: informal speech patterns, Arabic-Malayalam mix words
- Thrissur dialect: unique pronunciation, "Thrissur pooram" cultural references
- Kozhikode dialect: coastal Malayalam, fish/trade related expressions
- Old people: classical Malayalam words, formal speech patterns, slower cadence
- Fast speakers: dropped syllables - reconstruct full meaning from context
- Village accents: vowel sound changes but meaning preserved
- Manglish (mixed Malayalam-English): treat English words as part of Malayalam flow

COMMON KERALA EXPRESSIONS TO TRANSLATE CORRECTLY:
- "cool mood" = calm/relaxed state
- "enthu" = what/enthusiasm
- "adipoli" = awesome/excellent
- "machi" = friend/buddy
- "alle" = isn't it/right
- "aano" = is it so
- "sherikkum" = really/truly
- "poda" = get lost (casual)
- "moksha" = relief/salvation
- "timepass" = passing time/entertainment

TRANSLATION QUALITY RULES:
1. Fix grammar to natural flowing English
2. Translate idioms culturally, not literally
3. Preserve emotional tone (formal/informal/urgent/casual)
4. Handle noisy audio artifacts - ignore filler sounds
5. Old people voices: expand contracted speech to full sentences
6. Fast speech: fill in dropped words from context
7. Mixed Malayalam-English: seamlessly integrate English words

Raw translation to improve: {raw_english}

Output ONLY the improved English translation.
No explanations. No quotes. No extra text."""
                }
            ]
        )
        improved = message.content[0].text.strip()
        print(f"Claude improved translation: {improved}")
        return improved
    except Exception as e:
        print(f"Claude improvement error: {e}")
        return raw_english


# ========================
# ENGLISH TO MALAYALAM
# ========================
def english_to_malayalam_google(text):
    try:
        if not text or text.strip() == "":
            return ""
        translator = GoogleTranslator(source='en', target='ml')
        result = translator.translate(text.strip())
        return result if result else text
    except Exception as e:
        print(f"Reverse translation error: {e}")
        return text


# ========================
# REFINE ENGLISH TEXT
# ========================
def refine_english(text):
    if not text:
        return ""
    refined = text.strip()
    if refined:
        refined = refined[0].upper() + refined[1:]
    if refined and not refined.endswith(('.', '!', '?')):
        refined += '.'
    return refined


# ========================
# TRANSLITERATION
# ========================
def malayalam_to_romanized(malayalam_text):
    malayalam_roman_map = {
        'അ': 'a', 'ആ': 'aa', 'ഇ': 'i', 'ഈ': 'ee',
        'ഉ': 'u', 'ഊ': 'oo', 'എ': 'e', 'ഏ': 'ae',
        'ഒ': 'o', 'ഓ': 'oa', 'ക': 'ka', 'ഗ': 'ga',
        'ച': 'cha', 'ജ': 'ja', 'ട': 'ta', 'ഡ': 'da',
        'ത': 'tha', 'ദ': 'da', 'ന': 'na', 'പ': 'pa',
        'ബ': 'ba', 'മ': 'ma', 'യ': 'ya', 'ര': 'ra',
        'ല': 'la', 'വ': 'va', 'ശ': 'sha', 'സ': 'sa',
        'ഹ': 'ha', 'ള': 'la', 'ഴ': 'zha', 'റ': 'ra'
    }
    if not malayalam_text:
        return ""
    romanized = malayalam_text
    for mal_char, rom_char in malayalam_roman_map.items():
        romanized = romanized.replace(mal_char, rom_char)
    return romanized


# ========================
# MAIN TRANSLATION PIPELINE
# ========================
def translate_text_dummy(text):
    try:
        text = text.strip()
        transliteration = malayalam_to_romanized(text)

        # Use Claude to improve the WhisperX translation
        improved = improve_translation_with_claude(text)
        refined = refine_english(improved)

        return {
            "status": "success",
            "original": text,
            "transliteration": transliteration,
            "translation": improved,
            "refined": refined,
            "confidence": "high"
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "original": text,
            "translation": text,
            "refined": text,
            "transliteration": text
        }


# ========================
# ENGLISH TO MALAYALAM
# ========================
def translate_eng_to_ml(text):
    try:
        text = text.strip()
        ml_text = english_to_malayalam_google(text)
        if not ml_text:
            ml_text = text
        return {"malayalam": ml_text, "status": "success"}
    except Exception as e:
        return {"malayalam": text, "status": "failed", "error": str(e)}