import os
for root, dirs, files in os.walk("../frontend"):
    for f in files:
        if f.endswith((".js", ".jsx", ".ts", ".tsx")):
            path = os.path.join(root, f)
            try:
                content = open(path, encoding="utf-8", errors="ignore").read()
                if "fester-yonder-stoplight.ngrok-free.dev" in content:
                    new = content.replace("https://fester-yonder-stoplight.ngrok-free.dev", "https://use-ai-malayalamai-production-ee70.up.railway.app")
                    open(path, "w", encoding="utf-8").write(new)
                    print("Updated:", path)
            except Exception as e:
                print("Skip:", path, e)
