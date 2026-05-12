from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

# Global variables to load models once
device = "cuda" if torch.cuda.is_available() else "cpu"

# IndicTrans2 Malayalam→English model
MODEL_NAME = "ai4bharat/indictrans2-indic-en-1B"
tokenizer = None
model = None

def load_translation_model():
    """Load IndicTrans2 model once"""
    global tokenizer, model
    if tokenizer is None or model is None:
        try:
            print("Loading IndicTrans2 Malayalam→English model...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
            model = model.to(device)
            print("Translation model loaded successfully")
        except Exception as e:
            print(f"Error loading translation model: {e}")
            # Fallback to dummy responses
            return False
    return True

def malayalam_to_english(malayalam_text: str):
    """
    Translate Malayalam text to English using IndicTrans2
    """
    if not malayalam_text or malayalam_text.strip() == "":
        return "No text to translate"
    
    try:
        # Load model if needed
        if not load_translation_model():
            return f"Translation unavailable - using original: {malayalam_text}"
        
        # Prepare input with language tokens
        input_text = f"<2en> {malayalam_text}"
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate translation
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=512, num_beams=4)
        
        # Decode result
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation.strip()
        
    except Exception as e:
        print(f"Translation error: {e}")
        return f"Translation failed: {malayalam_text}"

def malayalam_to_romanized(malayalam_text: str):
    """
    Convert Malayalam to Roman/Latin script (simple transliteration)
    """
    # Basic Malayalam to Roman mapping (simplified)
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
    
    # Simple character replacement
    romanized = malayalam_text
    for mal_char, rom_char in malayalam_roman_map.items():
        romanized = romanized.replace(mal_char, rom_char)
    
    return romanized

def refine_english(english_text: str, mode: str = "general"):
    """
    Improve English grammar and style
    """
    if not english_text:
        return ""
    
    # Basic grammar fixes (you can enhance this later with GPT)
    refined = english_text.strip()
    
    # Capitalize first letter
    if refined:
        refined = refined[0].upper() + refined[1:]
    
    # Add period if missing
    if refined and not refined.endswith(('.', '!', '?')):
        refined += '.'
    
    # Mode-specific adjustments
    if mode == "formal":
        refined = refined.replace("I'm", "I am").replace("can't", "cannot")
    elif mode == "casual":
        refined = refined.lower().replace(".", "")
    
    return refined

def translate_text_dummy(text: str):
    """
    Main translation function - replaces your old dummy function
    """
    try:
        # Step 1: Transliteration
        transliteration = malayalam_to_romanized(text)
        
        # Step 2: Translation
        translation = malayalam_to_english(text)
        
        # Step 3: Refinement
        refined = refine_english(translation, mode="general")
        
        return {
            "transliteration": transliteration,
            "translation": translation,
            "refined": refined,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "transliteration": text,
            "translation": f"Translation error: {str(e)}",
            "refined": text,
            "status": "failed"
        }
