path = r'C:\Users\User\Desktop\USE AI_MalayalamAI_8 MAY 2026\backend\app\services\asr_service.py'
with open(path, encoding='utf-8') as f:
    c = f.read()

# Add audio normalization to prevent clipping
old = "def _to_wav_bytes(audio):\n    if not isinstance(audio, np.ndarray):\n        audio, _ = sf.read(audio, dtype='float32')\n        if len(audio.shape) > 1: audio = audio.mean(axis=1)\n    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:"

new = "def _to_wav_bytes(audio):\n    if not isinstance(audio, np.ndarray):\n        audio, _ = sf.read(audio, dtype='float32')\n        if len(audio.shape) > 1: audio = audio.mean(axis=1)\n    # Normalize to prevent clipping (max=1.0)\n    peak = np.max(np.abs(audio))\n    if peak > 0.95:\n        audio = audio * (0.95 / peak)\n    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:"

c = c.replace(old, new)

# Also add hallucination phrases that appeared
old2 = "    'story about a little',\n]"
new2 = "    'story about a little',\n    'romantic music',\n    'my all dear people',\n    'new episode of the video game',\n    'living thank you',\n]"
c = c.replace(old2, new2)

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)
print('Done')
