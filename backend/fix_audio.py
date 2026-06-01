import re

with open('app/routers/audio.py', 'r') as f:
    c = f.read()

# Fix 1: Save as webm instead of wav
c = c.replace('suffix=".wav"', 'suffix=".webm"')

# Fix 2: Add ffmpeg conversion
old = 'print("Temp audio saved")\n        # =========================\n        # Fast ASR\n        # =========================\n        asr_result = transcribe_audio(\n            temp_file_path\n        )'
new = 'print("Temp audio saved")\n        import subprocess\n        wav_path = temp_file_path.replace(".webm", ".wav")\n        subprocess.run(["ffmpeg","-y","-i",temp_file_path,"-ar","16000","-ac","1","-f","wav",wav_path], capture_output=True)\n        print("Converted to wav")\n        asr_result = transcribe_audio(wav_path)'
c = c.replace(old, new)

with open('app/routers/audio.py', 'w') as f:
    f.write(c)

print("Done" if "wav_path" in c else "FAILED")
