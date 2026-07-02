import re, json, glob

rows = {}
pattern = re.compile(
    r"^\|\s*\d+\s*\|\s*(.+?)\s*\|\s*([A-Z](?:,[A-Z])*)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*<(https://[^\s>]+)>\s*\|",
    re.M,
)

for path in glob.glob("data/traces/*.md"):
    text = open(path).read()
    for m in pattern.finditer(text):
        name, ttype, keys, duration, languages, url = [g.strip() for g in m.groups()]
        rows[url] = {
            "name": name,
            "url": url,
            "test_type": ttype,
            "test_type_label": keys,
            "duration": duration,
            "languages": languages,
            "source": "verified_from_trace",
        }

items = sorted(rows.values(), key=lambda x: x["name"])
with open("data/catalog_from_traces.json", "w") as f:
    json.dump(items, f, indent=2)
print(f"Extracted {len(items)} distinct verified assessments from traces")
for it in items:
    print(" -", it["name"], "|", it["test_type"])
