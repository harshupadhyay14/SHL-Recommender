"""
Loads data/catalog.json and builds a hybrid retrieval index over it so the
agent can ground its recommendations in retrieval rather than the model's
prior.

Retrieval is BM25 (lexical) + local sentence-embedding cosine similarity
(semantic), combined via Reciprocal Rank Fusion. Why hybrid: pure BM25 is
great when the conversation names a skill/tool directly ("needs SQL and
Docker") but structurally can't handle inferred needs where the query and
the catalog item share no vocabulary at all ("containerization skills" ->
Docker, "documentation skills" -> MS Word). The embedding model closes that
gap; BM25 stays because exact terminology matches (product names, acronyms)
are still often more reliable from lexical overlap than from embeddings.

The embedding model runs locally (sentence-transformers, all-MiniLM-L6-v2,
~80MB) -- no API calls, no cost, doesn't touch the Groq token budget at all.

Falls back to data/catalog_seed.json (a small manually-verified sample) if
the full scrape hasn't been run yet -- this keeps local dev/tests working,
but production MUST use the full catalog (run scripts/scrape_catalog.py or
scripts/convert_official_catalog.py).
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FULL_CATALOG = DATA_DIR / "catalog.json"
SEED_CATALOG = DATA_DIR / "catalog_seed.json"
EMBEDDING_CACHE = DATA_DIR / "catalog_embeddings.npy"

_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_RRF_K = 60  # standard Reciprocal Rank Fusion constant

_embed_model = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embed_model


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
# insensitively so this keeps working once the full ~380-item catalog is
# scraped, not just against today's dev catalog.
_DEFAULT_CANDIDATE_NAME_SUBSTRINGS = [
    "occupational personality questionnaire",
    "opq32r",
    "global skills assessment",
]


class Catalog:
    def __init__(self, items: List[Dict[str, Any]], compute_embeddings: bool = True):
        self.items = items
        self.by_url = {item["url"]: item for item in items}
        self._doc_texts = [self._doc_text(item) for item in items]
        self._tokenized_corpus = [_tokenize(doc) for doc in self._doc_texts]
        self.bm25 = BM25Okapi(self._tokenized_corpus) if items else None
        self._default_candidates = [
            item for item in items
            if any(sub in item.get("name", "").lower() for sub in _DEFAULT_CANDIDATE_NAME_SUBSTRINGS)
        ]

        self.embeddings = None
        if compute_embeddings and items:
            self.embeddings = self._load_or_build_embeddings()

    def _load_or_build_embeddings(self) -> np.ndarray:
        # Cache embeddings to disk keyed loosely by item count -- avoids
        # recomputing ~377 embeddings (a few seconds) on every process
        # start/reload. If the catalog changes size the cache is rebuilt.
        if EMBEDDING_CACHE.exists():
            try:
                cached = np.load(EMBEDDING_CACHE)
                if cached.shape[0] == len(self.items):
                    return cached
            except Exception:
                pass  # corrupt/stale cache -- fall through and rebuild

        model = _get_embed_model()
        embeddings = model.encode(
            self._doc_texts, normalize_embeddings=True, show_progress_bar=False
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        try:
            np.save(EMBEDDING_CACHE, embeddings)
        except Exception:
            pass  # non-fatal if we can't write the cache (e.g. read-only fs)
        return embeddings

    @staticmethod
    def _doc_text(item: Dict[str, Any]) -> str:
        parts = [
            item.get("name", ""),
            item.get("test_type_label", item.get("test_type", "")),
            item.get("description", ""),
            item.get("job_levels", ""),
        ]
        return " ".join(p for p in parts if p)

    def _bm25_ranking(self, query: str) -> List[int]:
        """Returns item indices ranked best-to-worst by BM25 score."""
        tokens = _tokenize(query)
        if not tokens or self.bm25 is None:
            return list(range(len(self.items)))
        scores = self.bm25.get_scores(tokens)
        return list(np.argsort(-scores))

    def _embedding_ranking(self, query: str) -> List[int]:
        """Returns item indices ranked best-to-worst by cosine similarity."""
        if self.embeddings is None:
            return list(range(len(self.items)))
        model = _get_embed_model()
        query_emb = model.encode([query], normalize_embeddings=True)[0]
        sims = self.embeddings @ query_emb
        return list(np.argsort(-sims))

    def search(self, query: str, top_k: int = 40) -> List[Dict[str, Any]]:
        if not self.items:
            return []

        if not query.strip():
            results = self.items[:top_k]
        else:
            bm25_rank = self._bm25_ranking(query)
            embed_rank = self._embedding_ranking(query)

            # Reciprocal Rank Fusion: combine two ranked lists without
            # needing to normalize/tune raw BM25 vs. cosine-similarity
            # score scales against each other.
            rrf_scores: Dict[int, float] = {}
            for rank, idx in enumerate(bm25_rank):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (_RRF_K + rank)
            for rank, idx in enumerate(embed_rank):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (_RRF_K + rank)

            ranked_idx = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)
            results = [self.items[i] for i in ranked_idx[:top_k]]

        # Some instruments (broad personality/behavior measures like OPQ32r,
        # generalist skills assessments like GSA) are conventionally offered
        # as a default add-on across nearly any hiring scenario -- that's a
        # business-rule fact, not a similarity fact, so retrieval alone can
        # still miss them. Force them into the pool so the LLM can choose to
        # offer or skip them per its instructions.
        seen_urls = {item["url"] for item in results}
        for default_item in self._default_candidates:
            if default_item["url"] not in seen_urls:
                results.append(default_item)
                seen_urls.add(default_item["url"])

        # If the conversation explicitly names an assessment (e.g. "needs
        # SQL and Docker skills"), that's a much stronger, cheaper-to-trust
        # signal than either ranking's aggregate score across a possibly
        # long message. Force those exact-name mentions into the pool too.
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