from deep_translator import GoogleTranslator


def is_english(text):
    if not text:
        return False
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    total = sum(1 for c in text if c.isalpha())
    if total == 0:
        return True
    return latin / total > 0.85


def malayalam_to_english_google(text):
    if not text or text.strip() == "":
        return "No text to translate"

    text = text.strip()

    try:
        translator = GoogleTranslator(source='ml', target='en')
        result = translator.translate(text)
        if result and is_english(result):
            print(f"Translation success: {result}")
            return result
    except Exception as e:
        print(f"Translation error: {e}")

    try:
        translator = GoogleTranslator(source='auto', target='en')
        result = translator.translate(text)
        return result if result else text
    except Exception as e:
        print(f"Fallback translation error: {e}")
        return text


def english_to_malayalam_google(text):
    try:
        translator = GoogleTranslator(source='en', target='ml')
        result = translator.translate(text)
        return result
    except Exception as e:
        print(f"Reverse translation error: {e}")
        return None


def malayalam_to_romanized(malayalam_text):
    malayalam_roman_map = {
        'അ': 'a', 'ആ': 'aa', 'ഇ': 'i', 'ഈ': 'ee', 'ഉ': 'u', 'ഊ': 'oo',
        'എ': 'e', 'ഏ': 'ae', 'ഒ': 'o', 'ഓ': 'oa', 'ക': 'ka', 'ഗ': 'ga',
        'ച': 'cha', 'ജ': 'ja', 'ട': 'ta', 'ഡ': 'da', 'ത': 'tha', 'ദ': 'da',
        'ന': 'na', 'പ': 'pa', 'ബ': 'ba', 'മ': 'ma', 'യ': 'ya', 'ര': 'ra',
        'ല': 'la', 'വ': 'va', 'ശ': 'sha', 'സ': 'sa', 'ഹ': 'ha', 'ള': 'la',
        'ഴ': 'zha', 'റ': 'ra'
    }
    if not malayalam_text:
        return ""
    romanized = malayalam_text
    for mal_char, rom_char in malayalam_roman_map.items():
        romanized = romanized.replace(mal_char, rom_char)
    return romanized


def refine_english(english_text):
    if not english_text:
        return ""
    refined = english_text.strip()
    if refined:
        refined = refined[0].upper() + refined[1:]
    if refined and not refined.endswith(('.', '!', '?')):
        refined += '.'
    return refined


def translate_text_dummy(text):
    try:
        text = text.strip()
        transliteration = malayalam_to_romanized(text)
        translation = malayalam_to_english_google(text)
        if not translation:
            translation = "Could not translate"
        refined = refine_english(translation)
        return {
            "transliteration": transliteration,
            "translation": translation,
            "refined": refined,
            "status": "success"
        }
    except Exception as e:
        return {
            "transliteration": text,
            "translation": f"Error: {str(e)}",
            "refined": text,
            "status": "failed"
        }


def translate_eng_to_ml(text):
    try:
        text = text.strip()
        ml_text = english_to_malayalam_google(text)
        if not ml_text:
            ml_text = text
        return {"malayalam": ml_text, "status": "success"}
    except Exception as e:
        return {"malayalam": text, "status": "failed", "error": str(e)}
