content = open("main.py", "r").read()
old = '    print("Startup: skipping init_db for debug")'
new = '    init_db()\n    print("DB init OK")'
content = content.replace(old, new)
open("main.py", "w").write(content)
print("Done")
