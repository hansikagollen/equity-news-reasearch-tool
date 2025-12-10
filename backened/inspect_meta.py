# inspect_meta.py
import json, itertools

META = "faiss_meta.json"
n = 20  # change how many entries to show

with open(META, "r", encoding="utf-8") as f:
    meta = json.load(f)

print(f"Total metadata entries: {len(meta)}\n")
for i, item in enumerate(itertools.islice(meta, n)):
    print(f"--- entry {i} ---")
    print("chunk_id:", item.get("chunk_id"))
    print("title:", item.get("title"))
    print("url:", item.get("url"))
    txt = item.get("text","")
    print("text (first 300 chars):")
    print(txt[:300].replace("\n"," ") + ("..." if len(txt) > 300 else ""))
    print()
