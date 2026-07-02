"""
Scrapes SHL's public Individual Test Solutions catalog into data/catalog.json.

Two-phase scrape:
  1. List pages (32 pages x 12 rows) -> name, url, test_type letters.
  2. Detail pages (one per assessment) -> description, job levels, languages,
     remote_testing, adaptive_irt. This phase is what makes retrieval actually
     work well (matching "Java developer, mid-level, stakeholder comms" needs
     more than a name string) so don't skip it.

Run this from an environment with real internet access (your laptop, or as
part of the Render build step) -- it will NOT work from a sandboxed dev
container with restricted egress.

Usage:
    python scrape_catalog.py                # full scrape (list + details)
    python scrape_catalog.py --list-only     # just the fast list pass
    python scrape_catalog.py --workers 8     # parallel detail fetches
"""
import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "https://www.shl.com/products/product-catalog/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SHLCatalogBot/1.0; research/assignment use)"}
TEST_TYPE_LEGEND = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.json"
PAGE_SIZE = 12
NUM_PAGES = 32  # observed: 32 pages of Individual Test Solutions (type=1) as of mid-2026


def fetch(url: str, session: requests.Session, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            resp = session.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("unreachable")


def scrape_list_pages(session: requests.Session) -> list[dict]:
    items = []
    seen_urls = set()
    for page in range(NUM_PAGES):
        start = page * PAGE_SIZE
        url = f"{BASE}?start={start}&type=1"
        html = fetch(url, session)
        soup = BeautifulSoup(html, "html.parser")

        # The Individual Test Solutions table is identified by its header cell text.
        table = None
        for t in soup.find_all("table"):
            header = t.find("th")
            if header and "Individual Test Solutions" in header.get_text():
                table = t
                break
        if table is None:
            print(f"  [warn] no catalog table found on page start={start}", file=sys.stderr)
            continue

        rows = table.find_all("tr")[1:]  # skip header row
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 4:
                continue
            link = cells[0].find("a")
            if not link:
                continue
            name = link.get_text(strip=True)
            href = link.get("href", "")
            full_url = href if href.startswith("http") else f"https://www.shl.com{href}"
            remote_testing = bool(cells[1].find(["img", "span"]))
            adaptive_irt = bool(cells[2].find(["img", "span"]))
            test_type = cells[3].get_text(strip=True)

            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)
            items.append({
                "name": name,
                "url": full_url,
                "test_type": test_type,
                "test_type_label": TEST_TYPE_LEGEND.get(test_type, test_type),
                "remote_testing": remote_testing,
                "adaptive_irt": adaptive_irt,
            })
        print(f"  page start={start}: {len(rows)} rows (total so far: {len(items)})")
        time.sleep(0.3)  # be a polite scraper
    return items


def scrape_detail(item: dict, session: requests.Session) -> dict:
    """Enrich one catalog item with description / job level / duration if present."""
    try:
        html = fetch(item["url"], session)
    except requests.RequestException as e:
        item["description"] = ""
        item["scrape_error"] = str(e)
        return item

    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n", strip=True)

    def extract_after(label: str) -> str:
        m = re.search(re.escape(label) + r"\s*:?\s*\n?(.{0,400}?)(?:\n[A-Z][a-zA-Z ]{2,30}:|\Z)", text, re.S)
        return m.group(1).strip() if m else ""

    description = extract_after("Description")
    job_levels = extract_after("Job Levels") or extract_after("Job Level")
    languages = extract_after("Languages") or extract_after("Language")
    duration = extract_after("Assessment Length") or extract_after("Approximate Completion Time")

    item.update({
        "description": description[:1000],
        "job_levels": job_levels[:300],
        "languages": languages[:300],
        "duration": duration[:100],
    })
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-only", action="store_true", help="skip detail-page enrichment")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    session = requests.Session()

    print("Phase 1: scraping list pages...")
    items = scrape_list_pages(session)
    print(f"Collected {len(items)} unique individual test solutions.")

    if not args.list_only:
        print(f"Phase 2: scraping detail pages with {args.workers} workers...")
        enriched = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(scrape_detail, item, requests.Session()): item for item in items}
            for i, fut in enumerate(as_completed(futures), 1):
                enriched.append(fut.result())
                if i % 25 == 0:
                    print(f"  detail progress: {i}/{len(items)}")
        items = enriched

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(items, f, indent=2)
    print(f"Wrote {len(items)} items to {OUT_PATH}")


if __name__ == "__main__":
    main()
