# SHL Assessment Recommender

## What's here
- `app/` — FastAPI service (`main.py`), hybrid BM25 + embedding retrieval (`catalog.py`), Groq agent (`agent.py`), schemas
- `scripts/convert_official_catalog.py` — converts SHL's official pre-scraped catalog JSON (`data/catalog_raw.json`) into the schema this app uses
- `data/catalog.json` — **full catalog, 377 items**, already committed and ready to use as-is. No scrape step needed at setup or deploy time.
- `data/catalog_raw.json` — the original official catalog JSON, kept for reference
- `data/traces/` — the 10 labeled conversation traces
- `tests/replay_harness.py` — replays each trace's real user turns against the live agent, computes Recall@10
- `scripts/check_retrieval.py` — offline check (no LLM calls) of what fraction of gold-standard assessments are reachable by retrieval alone

## 1. Setup
```powershell
cd shl-recommender
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# create a .env file with:
#   GROQ_API_KEY=your_key_here
```

## 2. Run the server locally
```powershell
uvicorn app.main:app --reload --port 8000
```
```powershell
python -c "import requests; r = requests.get('http://localhost:8000/health'); print(r.status_code, r.text)"
python -c "import requests; r = requests.post('http://localhost:8000/chat', json={'messages':[{'role':'user','content':'Hiring a Java developer'}]}); print(r.status_code, r.text)"
```
(Prefer the Python `requests` snippets above over `curl` on Windows/PowerShell — PowerShell's quoting rules mangle escaped JSON in `curl.exe -d "..."` calls.)

## 3. Run the replay harness
```powershell
python tests/replay_harness.py
```
Prints per-trace turn counts, gold shortlist size, predicted shortlist size, and Recall@10, plus a mean across all 10 traces. Paced with a delay between calls to respect Groq's free-tier rate limits (12K TPM / 100K TPD).

## 4. Deploy to Render
- Push this repo to GitHub, connect it as a new Web Service on Render.
- `render.yaml` defines the build/start commands — build is just `pip install -r requirements.txt` (the catalog is already committed, no scrape needed at build time).
- Add `GROQ_API_KEY` as a secret env var in the Render dashboard (never commit it).
- First request after a cold start may take longer than usual — Render's free tier sleeps idle services, and the embedding model + index also build in-memory on first load.

## Design notes (see approach doc for full detail)
- **Retrieval**: hybrid BM25 (lexical) + local sentence-embedding cosine similarity (`all-MiniLM-L6-v2`, runs offline, no API cost), combined via Reciprocal Rank Fusion. BM25 alone was replaced with this hybrid approach because pure lexical matching missed assessments that are semantically implied rather than named (e.g. "containerization skills" → Docker).
- **Agent**: single Groq call per turn (JSON mode), not a multi-hop tool-calling loop — this is a single-hop retrieval task, and one grounded call comfortably fits the 30s timeout and is easier to keep from hallucinating than an open-ended agentic loop.
- **Hallucination guard**: every recommendation URL is checked against the actual catalog after the LLM responds; anything not in the catalog is silently dropped. Enforced in code, not just prompted for.
- **Error handling**: the Groq call is wrapped with a timeout and exception handling, so a rate limit or transient failure degrades to a valid response instead of a raw 500.
- **Stateless design**: full history is re-sent and re-parsed each call; there's no server-side session state.
- **Known limitation**: retrieval is grounded in the user's literal words, so recommendations that depend on a domain inference not yet stated in the conversation (e.g. inferring "dependability/safety" relevance from "handles patient records") can be missed. See approach doc for detail.
