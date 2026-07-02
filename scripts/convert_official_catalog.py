"""
Converts data/catalog_raw.json (SHL Labs' official pre-scraped catalog dump,
downloaded from the link in the assignment PDF) into the schema app/catalog.py
expects.

Usage:
    python scripts/convert_official_catalog.py
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_PATH = DATA_DIR / "catalog_raw.json"
OUT_PATH = DATA_DIR / "catalog.json"

KEY_TO_LETTER = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Biodata & Situational Judgement": "B",  # handle either spelling
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}


def convert(raw_item: dict) -> dict:
    keys = raw_item.get("keys", [])
    letters = [KEY_TO_LETTER.get(k, "") for k in keys]
    letters = [l for l in letters if l]

    job_levels = raw_item.get("job_levels", [])
    languages = raw_item.get("languages", [])

    return {
        "name": raw_item.get("name", ""),
        "url": raw_item.get("link", ""),
        "test_type": ",".join(letters),
        "test_type_label": ", ".join(keys),
        "description": raw_item.get("description", "") or "",
        "job_levels": ", ".join(job_levels) if isinstance(job_levels, list) else str(job_levels),
        "languages": ", ".join(languages) if isinstance(languages, list) else str(languages),
        "duration": raw_item.get("duration", "") or "",
        "remote_testing": raw_item.get("remote", "") == "yes",
        "adaptive_irt": raw_item.get("adaptive", "") == "yes",
    }


def main():
    with open(RAW_PATH, encoding="utf-8") as f:
        raw_items = json.load(f, strict=False)

    converted = [convert(item) for item in raw_items]
    # Drop anything without a usable name/url (defensive, shouldn't happen).
    converted = [c for c in converted if c["name"] and c["url"]]

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=2)

    print(f"Converted {len(converted)} items -> {OUT_PATH}")
    missing_desc = sum(1 for c in converted if not c["description"])
    print(f"  {missing_desc} items have no description text (out of {len(converted)})")


if __name__ == "__main__":
    main()