content = open("main.py", "r").read()
old = '    try:\n        init_db()\n        print("DB init OK")\n    except Exception as e:\n        print(f"DB init FAILED: {e}")\n        import traceback\n        traceback.print_exc()'
new = '    print("Startup: skipping init_db for debug")'
content = content.replace(old, new)
open("main.py", "w").write(content)
print("Done:", "init_db skipped" if "skipping" in content else "NOT replaced")
