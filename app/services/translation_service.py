# ========================
# FAST LOCAL TRANSLATION SERVICE
# ========================

def refine_english(text):
    """
    Fast local text cleanup
    """

    if not text:
        return ""

    refined = text.strip()

    # Remove extra spaces
    refined = " ".join(refined.split())

    # Capitalize first letter
    if refined:
        refined = refined[0].upper() + refined[1:]

    # Add punctuation if missing
    if refined and not refined.endswith(('.', '!', '?')):
        refined += '.'

    return refined


# ========================
# ENGLISH TO MALAYALAM
# TEMPORARY FAST VERSION
# ========================
def translate_eng_to_ml(text):
    """
    Temporary fast local translation
    Removes API delay
    """

    if not text:
        text = ""

    return {
        "malayalam": text,
        "status": "success"
    }


# ========================
# OPTIONAL PLACEHOLDER
# ========================
def translate_text_dummy(text):

    refined = refine_english(text)

    return {
        "status": "success",
        "original": text,
        "translation": refined,
        "refined": refined,
        "confidence": "medium"
    }