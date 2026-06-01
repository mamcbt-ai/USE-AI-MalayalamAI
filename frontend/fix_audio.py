content = open("app/routers/audio.py", "r").read()
old = '''        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as tmp:
            temp_file_path = tmp.name
            content = await file.read()
            tmp.write(content)
        print("Temp audio saved")
        # =========================
        # Fast ASR
        # =========================
        asr_result = transcribe_audio(
            temp_file_path
        )'''
new = '''        # Save with correct format
        suffix = ".webm"
        if file.content_type and "wav" in file.content_type:
            suffix = ".wav"
        elif file.filename and file.filename.endswith(".wav"):
            suffix = ".wav"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as tmp:
            temp_file_path = tmp.name
            content = await file.read()
            tmp.write(content)
        print("Temp audio saved:", suffix)

        # Convert to wav if needed
        wav_path = temp_file_path
        if suffix != ".wav":
            import subprocess
            wav_path = temp_file_path.replace(suffix, ".wav")
            subprocess.run([
                "ffmpeg", "-y", "-i", temp_file_path,
                "-ar", "16000", "-ac", "1", "-f", "wav", wav_path
            ], capture_output=True)
            print("Converted to wav")

        # =========================
        # Fast ASR
        # =========================
        asr_result = transcribe_audio(
            wav_path
        )'''
result = content.replace(old, new)
open("app/routers/audio.py", "w").write(result)
print("Done" if old in content else "NOT REPLACED - pattern mismatch")
