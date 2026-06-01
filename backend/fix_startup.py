content = open("main.py", "r").read()
old = '@app.on_event("startup")\ndef on_startup():\n    init_db()'
new = '@app.on_event("startup")\ndef on_startup():\n    try:\n        init_db()\n        print("DB init OK")\n    except Exception as e:\n        print(f"DB init FAILED: {e}")\n        import traceback\n        traceback.print_exc()'
content = content.replace(old, new)
open("main.py", "w").write(content)
print("Done")
