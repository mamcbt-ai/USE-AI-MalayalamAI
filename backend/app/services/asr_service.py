import os, re, numpy as np, tempfile, requests
import soundfile as sf
from groq import Groq

groq_client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
SARVAM_KEY = os.environ.get('SARVAM_API_KEY', '')
LLM_MODEL = 'llama-3.3-70b-versatile'
DEVICE = 'sarvam+groq'
print(f'ASR ready: Sarvam AI (Indian STT) + {LLM_MODEL}')

LANG_CODES = {'ml':'ml-IN','ta':'ta-IN','te':'te-IN','kn':'kn-IN','hi':'hi-IN'}
LANG_NAMES = {'ml':'Malayalam','ta':'Tamil','te':'Telugu','kn':'Kannada','hi':'Hindi'}

UNICODE_PROMPTS = {
    'ml': 'You are a Malayalam expert. The following is transliterated or informal Malayalam text. Rewrite it in natural, fluent Malayalam Unicode script. Output ONLY Malayalam characters. No English, no explanation.',
    'ta': 'You are a Tamil expert. Rewrite the following in natural Tamil Unicode script. Output ONLY Tamil characters.',
    'te': 'You are a Telugu expert. Rewrite the following in natural Telugu Unicode script. Output ONLY Telugu characters.',
    'kn': 'You are a Kannada expert. Rewrite the following in natural Kannada Unicode script. Output ONLY Kannada characters.',
    'hi': 'You are a Hindi expert. Rewrite the following in natural Hindi Devanagari script. Output ONLY Hindi characters.',
}

TRANSLATE_PROMPTS = {
    'ml': 'Translate this Malayalam text to natural English. Output ONLY the English translation.',
    'ta': 'Translate this Tamil text to natural English. Output ONLY the English translation.',
    'te': 'Translate this Telugu text to natural English. Output ONLY the English translation.',
    'kn': 'Translate this Kannada text to natural English. Output ONLY the English translation.',
    'hi': 'Translate this Hindi text to natural English. Output ONLY the English translation.',
}

def cleanup_text(text):
    if not text: return ''
    text = re.sub(r'\s+', ' ', text.strip()).strip()
    # Remove repeated sentences
    sentences = [s.strip() for s in re.split(r'[.!?。।]', text) if s.strip()]
    seen = []
    for s in sentences:
        if s not in seen:
            seen.append(s)
    return '. '.join(seen).strip()

def _to_wav_bytes(audio):
    if not isinstance(audio, np.ndarray):
        audio, _ = sf.read(audio, dtype='float32')
        if len(audio.shape) > 1: audio = audio.mean(axis=1)
    # Trim to 25 seconds max (Sarvam API limit is 30s)
    audio = audio[:25*16000]
    peak = np.max(np.abs(audio))
    if peak > 0.95: audio = audio * (0.95 / peak)
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

def _sarvam_transcribe(wav_bytes, lang):
    lang_code = LANG_CODES.get(lang, 'ml-IN')
    try:
        resp = requests.post(
            'https://api.sarvam.ai/speech-to-text',
            headers={'api-subscription-key': SARVAM_KEY},
            files={'file': ('audio.wav', wav_bytes, 'audio/wav')},
            data={'language_code': lang_code, 'model': 'saarika:v2.5'},
            timeout=30)
        if resp.status_code == 200:
            result = resp.json().get('transcript', '')
            print(f'[ASR] Sarvam ({lang}): {result[:80]}')
            return cleanup_text(result)
        else:
            print(f'[ASR] Sarvam error {resp.status_code}: {resp.text[:100]}')
            return ''
    except Exception as e:
        print(f'[ASR] Sarvam exception: {e}')
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
    print(f'[ASR] Unicode: {result[:80]}')
    return result

def transcribe_audio(audio_input, style='standard', source_lang='ml'):
    try:
        wav = _to_wav_bytes(_load_audio(audio_input))
        print(f'[ASR] lang={source_lang} bytes={len(wav)}')
        native = _sarvam_transcribe(wav, source_lang)
        english = _to_english(native, source_lang) if native else ''
        native_unicode = _to_unicode(native, source_lang) if native else ''
        print(f'[ASR] Native : {native_unicode[:80]}')
        return {'status':'success','text':english,'malayalam_text':native_unicode,
                'raw_text':english,'language':source_lang,'segments':[],'device':DEVICE,'model':'saaras:v2'}
    except Exception as e:
        print(f'[ASR] Error: {e}')
        return {'status':'failed','error':str(e),'text':'','malayalam_text':'','segments':[]}

def transcribe_audio_stream(audio_input, style='standard', source_lang='ml'):
    try:
        wav = _to_wav_bytes(_load_audio(audio_input))
        print(f'[ASR] stream lang={source_lang} bytes={len(wav)}')
        native = _sarvam_transcribe(wav, source_lang)
        english = _to_english(native, source_lang) if native else ''
        if english: yield {'type':'english_segment','text':english,'accumulated':english}
        native_unicode = _to_unicode(native, source_lang) if native else ''
        if native_unicode: yield {'type':'malayalam_segment','text':native_unicode,'accumulated':native_unicode}
        yield {'type':'complete','english_text':english,'malayalam_text':native_unicode,'language':source_lang,'source_lang':source_lang}
    except Exception as e:
        print(f'[ASR] Stream Error: {e}')
        yield {'type':'error','error':str(e)}

