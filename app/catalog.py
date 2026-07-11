"""
Loads data/catalog.json and builds a BM25 index over it so the agent can
ground its recommendations in retrieval rather than the model's prior.

NOTE: an earlier version of this file added a hybrid BM25 + local
sentence-embedding layer (all-MiniLM-L6-v2, combined via Reciprocal Rank
Fusion) to close a specific retrieval gap where the correct assessment is
implied rather than named (e.g. "containerization skills" -> Docker). That
was reverted for deployment: sentence-transformers + its torch dependency
pushed memory usage over Render's free-tier 512MB limit and caused a boot
crash-loop. BM25-only is what's actually deployed; see the approach doc's
"Known Limitations" section for detail on the tradeoff.

Falls back to data/catalog_seed.json (a small manually-verified sample) if
the full catalog file is missing.
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FULL_CATALOG = DATA_DIR / "catalog.json"
SEED_CATALOG = DATA_DIR / "catalog_seed.json"

_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


_NEW_SUFFIX_RE = re.compile(r"\s*\(new\)\s*$", re.IGNORECASE)


def _strip_new_suffix(name: str) -> str:
    """Catalog items are heavily suffixed with '(New)' which is noise for
    substring/exact-mention matching (e.g. matching 'SQL' against the query
    shouldn't depend on whether the query also happens to contain the word
    'new')."""
    return _NEW_SUFFIX_RE.sub("", name).strip()


# Name substrings for instruments that are conventionally offered as a
# default add-on regardless of the specific role being discussed (broad
# personality/behavioral and general-skills measures). Matched case-
# insensitively.
_DEFAULT_CANDIDATE_NAME_SUBSTRINGS = [
    "occupational personality questionnaire",
    "opq32r",
    "global skills assessment",
]


class Catalog:
    def __init__(self, items: List[Dict[str, Any]]):
        self.items = items
        self.by_url = {item["url"]: item for item in items}
        corpus = [self._doc_text(item) for item in items]
        self._tokenized_corpus = [_tokenize(doc) for doc in corpus]
        self.bm25 = BM25Okapi(self._tokenized_corpus) if items else None
        self._default_candidates = [
            item for item in items
            if any(sub in item.get("name", "").lower() for sub in _DEFAULT_CANDIDATE_NAME_SUBSTRINGS)
        ]

    @staticmethod
    def _doc_text(item: Dict[str, Any]) -> str:
        parts = [
            item.get("name", ""),
            item.get("test_type_label", item.get("test_type", "")),
            item.get("description", ""),
            item.get("job_levels", ""),
        ]
        return " ".join(p for p in parts if p)

    def search(self, query: str, top_k: int = 18) -> List[Dict[str, Any]]:
        # Default lowered 40->18 to match agent.py's call site -- see the
        # comment there for why (Groq free-tier TPM budget). Callers that
        # need the old wider pool (e.g. offline retrieval-reachability
        # checks that don't cost any LLM tokens) can still pass top_k=40
        # explicitly.
        if not self.items:
            return []
        if not query.strip():
            results = self.items[:top_k]
        else:
            tokens = _tokenize(query)
            if not tokens:
                results = self.items[:top_k]
            else:
                scores = self.bm25.get_scores(tokens)
                ranked = sorted(zip(self.items, scores), key=lambda x: x[1], reverse=True)
                results = [item for item, score in ranked[:top_k] if score > 0] or self.items[:top_k]

        # Some instruments (broad personality/behavior measures like OPQ32r,
        # generalist skills assessments like GSA) are conventionally offered
        # as a default add-on across nearly any hiring scenario -- that's a
        # business-rule fact, not a lexical-similarity fact, so pure BM25
        # will systematically miss them. Force them into the pool so the
        # LLM can choose to offer or skip them per its instructions.
        seen_urls = {item["url"] for item in results}
        for default_item in self._default_candidates:
            if default_item["url"] not in seen_urls:
                results.append(default_item)
                seen_urls.add(default_item["url"])

        # If the conversation explicitly names an assessment (e.g. "needs
        # SQL and Docker skills"), that's a much stronger signal than BM25's
        # aggregate score across a possibly-long message. Force those exact-
        # name mentions into the pool.
        query_padded = f" {_strip_new_suffix(query.lower())} "
        for item in self.items:
            if item["url"] in seen_urls:
                continue
            name_key = _strip_new_suffix(item.get("name", "").lower())
            if len(name_key) >= 2 and f" {name_key} " in query_padded:
                results.append(item)
                seen_urls.add(item["url"])

        return results

    def is_valid_url(self, url: str) -> bool:
        return url in self.by_url


def load_catalog() -> Catalog:
    path = FULL_CATALOG if FULL_CATALOG.exists() else SEED_CATALOG
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    return Catalog(items)


catalog = load_catalog()