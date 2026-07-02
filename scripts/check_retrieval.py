"""
Offline check: does BM25 retrieval (now with the full catalog + default-
candidate injection) actually surface each trace's gold assessments in the
candidate pool handed to the LLM? This costs zero Groq tokens -- it only
exercises app/catalog.py -- so it's safe to run anytime, including while
your daily Groq quota is maxed out.

Usage:
    python scripts/check_retrieval.py
"""
import glob
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.catalog import load_catalog
from app.agent import _retrieval_queries
from app.schemas import Message

USER_RE = re.compile(r"\*\*User\*\*\s*\n\s*>\s*(.+?)\n\n\*\*Agent\*\*", re.S)
GOLD_URL_RE = re.compile(r"<(https://[^\s>]+)>")


def main():
    catalog = load_catalog()
    print(f"Catalog loaded: {len(catalog.items)} items\n")

    total_gold = 0
    total_hit = 0
    for path in sorted(glob.glob("data/traces/*.md")):
        text = open(path, encoding="utf-8").read()
        user_turns = USER_RE.findall(text)
        tables = text.split("| # | Name |")
        gold = GOLD_URL_RE.findall(tables[-1]) if len(tables) > 1 else []
        if not gold:
            continue

        msgs = [Message(role="user", content=t.strip()) for t in user_turns]
        cand_urls = set()
        for q in _retrieval_queries(msgs):
            for item in catalog.search(q, top_k=40):
                cand_urls.add(item["url"])
        hits = [g for g in gold if g in cand_urls]

        total_gold += len(gold)
        total_hit += len(hits)
        name = Path(path).name
        print(f"{name:10s} {len(hits)}/{len(gold)} gold items reachable by retrieval")
        for g in gold:
            if g not in cand_urls:
                item = catalog.by_url.get(g)
                label = item["name"] if item else "(URL NOT IN CATALOG AT ALL)"
                print(f"    still missing: {label}")

    print(f"\nOverall: {total_hit}/{total_gold} gold items reachable ({total_hit/total_gold:.1%})")
    print("(This is an upper bound on Recall@10 -- the LLM still has to actually pick")
    print(" the right ones from what's shown to it, so real recall will be <= this.)")


if __name__ == "__main__":
    main()