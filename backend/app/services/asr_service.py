import os
import re
import numpy as np
import tempfile
import soundfile as sf
from groq import Groq

groq_client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
WHISPER_MODEL = 'whisper-large-v3'
LLM_MODEL = 'llama-3.3-70b-versatile'
DEVICE = 'groq-api'

print(f'ASR ready: {WHISPER_MODEL} + {LLM_MODEL} via Groq API')

LANG_NAMES = {'ml':'Malayalam','ta':'Tamil','te':'Telugu','kn':'Kannada','hi':'Hindi'}

TRANSLATE_PROMPTS = {
    'ml': 'You are an expert Malayalam-English translator. Translate the Malayalam text to natural English. Output ONLY English.',
    'ta': 'You are an expert Tamil-English translator. Translate the Tamil text to natural English. Output ONLY English.',
    'te': 'You are an expert Telugu-English translator. Translate the Telugu text to natural English. Output ONLY English.',
    'kn': 'You are an expert Kannada-English translator. Translate the Kannada text to natural English. Output ONLY English.',
    'hi': 'You are an expert Hindi-English translator. Translate the Hindi text to natural English. Output ONLY English.',
}

UNICODE_PROMPTS = {
    'ml': 'You are a Malayalam expert. Write this text in natural Malayalam Unicode script (like: നമസ്കാരം). Output ONLY Malayalam characters, no English.',
    'ta': 'You are a Tamil expert. Write this text in natural Tamil Unicode script (like: வணக்கம்). Output ONLY Tamil characters, no English.',
    'te': 'You are a Telugu expert. Write this text in natural Telugu Unicode script (like: నమస్కారం). Output ONLY Telugu characters, no English.',
    'kn': 'You are a Kannada expert. Write this text in natural Kannada Unicode script (like: ನಮಸ್ಕಾರ). Output ONLY Kannada characters, no English.',
    'hi': 'You are a Hindi expert. Write this text in natural Hindi Devanagari script (like: नमस्ते). Output ONLY Hindi characters, no English.',
}

HALLUCINATIONS = [
    'thank you for watching','thanks for watching','subscribe','music','[music]',
    'subtitles','captions','hello and welcome','welcome to my channel',
    'translated by','english is a language','language of the language',
    "i'm here with a story",'hello everyone','story about a little',
]

def cleanup_text(text):
    if not text: return ''
    text = re.sub(r'\s+', ' ', text.strip())
    return text.strip()

def _is_hallucination(text):
    if not text or len(text) < 3: return False
    from collections import Counter
    counts = Counter(text.replace(' ', ''))
    if counts:
        ratio = counts.most_common(1)[0][1] / max(len(text.replace(' ', '')), 1)
        if ratio > 0.6: return True
    lower = text.lower()
    if any(lower.startswith(p) or (p in lower and len(text) < 80) for p in HALLUCINATIONS): return True
    words = text.split()
    if len(words) >= 4 and len(set(w.lower() for w in words)) / len(words) < 0.5: return True
    return False

def _to_wav_bytes(audio):
    if not isinstance(audio, np.ndarray):
        audio, _ = sf.read(audio, dtype='float32')
        if len(audio.shape) > 1: audio = audio.mean(axis=1)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, audio, 16000)
        p = tmp.name
    data = open(p,'rb').read()
    os.unlink(p)
    return data

def _load_audio(x):
    if isinstance(x, np.ndarray): return x
    a, _ = sf.read(x, dtype='float32')
    return a.mean(axis=1) if len(a.shape) > 1 else a

def _whisper_transcribe(wav_bytes, lang):
    try:
        r = groq_client.audio.transcriptions.create(
            file=('audio.wav', wav_bytes), model=WHISPER_MODEL,
            language=lang, response_format='text')
        t = r.text if hasattr(r,'text') else str(r)
        return cleanup_text(t)
    except Exception as e:
        print(f'[ASR] transcribe error: {e}')
        return ''

def _llm_call(system_prompt, user_text):
    if not user_text: return ''
    try:
        r = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{'role':'system','content':system_prompt},{'role':'user','content':user_text}],
            max_tokens=512, temperature=0.1)
        return cleanup_text(r.choices[0].message.content.strip())
    except Exception as e:
        print(f'[ASR] LLM error: {e}')
        return ''

def _to_english(native_text, lang):
    result = _llm_call(TRANSLATE_PROMPTS.get(lang, TRANSLATE_PROMPTS['ml']), native_text)
    print(f'[ASR] English: {result[:80]}')
    return result

def _to_unicode(native_text, lang):
    if not native_text: return ''
    ascii_ratio = sum(1 for c in native_text if ord(c)<128)/max(len(native_text),1)
    if ascii_ratio < 0.3: return native_text
    result = _llm_call(UNICODE_PROMPTS.get(lang, UNICODE_PROMPTS['ml']), native_text)
    print(f'[ASR] Unicode fix: {result[:80]}')
    return result

def transcribe_audio(audio_input, style='standard', source_lang='ml'):
    try:
        wav = _to_wav_bytes(_load_audio(audio_input))
        print(f'[ASR] lang={source_lang} bytes={len(wav)}')
        native = _whisper_transcribe(wav, source_lang)
        print(f'[ASR] Whisper: {native[:80]}')
        if _is_hallucination(native): native = ''
        english = _to_english(native, source_lang)
        native_unicode = _to_unicode(native, source_lang)
        print(f'[ASR] Native : {native_unicode[:80]}')
        return {'status':'success','text':english,'malayalam_text':native_unicode,
                'raw_text':english,'language':source_lang,'segments':[],'device':DEVICE,'model':WHISPER_MODEL}
    except Exception as e:
        print(f'[ASR] Error: {e}')
        return {'status':'failed','error':str(e),'text':'','malayalam_text':'','segments':[]}

def transcribe_audio_stream(audio_input, style='standard', source_lang='ml'):
    try:
        wav = _to_wav_bytes(_load_audio(audio_input))
        print(f'[ASR] stream lang={source_lang} bytes={len(wav)}')
        native = _whisper_transcribe(wav, source_lang)
        if _is_hallucination(native): native = ''
        print(f'[ASR] Whisper: {native[:80]}')
        english = _to_english(native, source_lang)
        if english: yield {'type':'english_segment','text':english,'accumulated':english}
        native_unicode = _to_unicode(native, source_lang)
        print(f'[ASR] Native : {native_unicode[:80]}')
        if native_unicode: yield {'type':'malayalam_segment','text':native_unicode,'accumulated':native_unicode}
        yield {'type':'complete','english_text':english,'malayalam_text':native_unicode,'language':source_lang,'source_lang':source_lang}
    except Exception as e:
        print(f'[ASR] Stream Error: {e}')
        yield {'type':'error','error':str(e)}
