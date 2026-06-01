path = r'C:\Users\User\Desktop\USE AI_MalayalamAI_8 MAY 2026\backend\app\services\asr_service.py'
with open(path, encoding='utf-8') as f:
    c = f.read()

# Show current _to_wav_bytes function
idx = c.find('def _to_wav_bytes')
print("CURRENT:")
print(c[idx:idx+300])
print("---")

# Replace with trimmed version
old = "def _to_wav_bytes(audio):\n    if not isinstance(audio, np.ndarray):\n        audio, _ = sf.read(audio, dtype='float32')\n        if len(audio.shape) > 1: audio = audio.mean(axis=1)\n    peak = np.max(np.abs(audio))"

new = "def _to_wav_bytes(audio):\n    if not isinstance(audio, np.ndarray):\n        audio, _ = sf.read(audio, dtype='float32')\n        if len(audio.shape) > 1: audio = audio.mean(axis=1)\n    # Trim to 25 seconds max (Sarvam API limit is 30s)\n    audio = audio[:25*16000]\n    peak = np.max(np.abs(audio))"

if old in c:
    c = c.replace(old, new)
    print("REPLACED OK")
else:
    print("NOT FOUND - showing wav_bytes func:")
    print(c[idx:idx+400])

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)