from deep_translator import GoogleTranslator


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
# (used when WhisperX translate task is not used)
# ========================
def translate_text_dummy(text):
    try:
        text = text.strip()
        transliteration = malayalam_to_romanized(text)
        refined = refine_english(text)
        return {
            "status": "success",
            "original": text,
            "transliteration": transliteration,
            "translation": text,
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
