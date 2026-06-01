content = open("app/core/db.py", "r").read()
old = 'engine = create_engine(DATABASE_URL, echo=False)'
new = 'engine = create_engine(DATABASE_URL, echo=True, connect_args={"connect_timeout": 10})'
content = content.replace(old, new)
open("app/core/db.py", "w").write(content)
print("Done")
