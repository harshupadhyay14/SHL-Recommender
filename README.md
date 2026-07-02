# SHL Assessment Recommender

## What's here
- `app/` — FastAPI service (`main.py`), BM25 retrieval (`catalog.py`), Groq agent (`agent.py`), schemas
- `scripts/scrape_catalog.py` — full 2-phase catalog scraper (run this to get all ~380 Individual Test Solutions)
- `scripts/extract_from_traces.py` — pulls verified assessment data straight out of the trace tables
- `data/catalog.json` — **dev catalog** (54 items: partial scrape + trace-verified). Replace with the real scrape before submitting.
- `data/traces/` — your 10 conversation traces
- `tests/replay_harness.py` — replays each trace's real user turns against the live agent, computes Recall@10

## 1. Setup
```bash
cd shl-recommender
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit .env and add your real GROQ_API_KEY
export $(cat .env | xargs)   # or just export GROQ_API_KEY=... directly
```

## 2. Get the full catalog (do this before final submission)
```bash
python scripts/scrape_catalog.py --workers 8
```
This overwrites `data/catalog.json` with the full ~380-item Individual Test Solutions catalog,
including descriptions/job levels/duration scraped from each detail page. Takes a few minutes.
The dev catalog checked in now (54 items) is enough to test the pipeline end-to-end but will hurt
your Recall@10 score if you deploy with it as-is — SHL's holdout traces will ask about assessments
outside this small set.

## 3. Run the replay harness (this is the important one — send me the output)
```bash
python tests/replay_harness.py
```
This prints per-trace turn counts, gold shortlist size, predicted shortlist size, and Recall@10,
plus a mean across all 10 traces. **Paste that output back to me** (not your API key) so I can see
where retrieval or the agent prompt is under/over-shooting and iterate.

## 4. Run the server locally
```bash
uvicorn app.main:app --reload --port 8000
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hiring a Java developer who works with stakeholders"}]}'
```

## 5. Deploy to Render
- Push this repo to GitHub.
- New Web Service on Render, connect the repo. `render.yaml` already defines the build/start commands
  (build step runs the full scraper automatically) — Render should pick it up via "Infrastructure as Code" or
  you can set Build/Start commands manually to match render.yaml.
- Add `GROQ_API_KEY` as a secret env var in the Render dashboard (never commit it).
- First `/health` call after a cold start can take up to 2 minutes (per the assignment spec) — Render free tier sleeps idle services.

## Design notes (for the approach doc)
- **Retrieval**: BM25 over name + test-type label + description + job levels. Chosen over embeddings because
  the catalog is a few hundred short structured records with domain-specific vocabulary (product names, acronyms
  like "SQL", "AWS") where lexical match outperforms semantic embeddings for a corpus this size and structure,
  and it's zero-cost/zero-latency vs. calling an embeddings API.
- **Agent**: single Groq call per turn (JSON mode) rather than a multi-hop tool-calling loop — this is a
  single-hop retrieval task, and one grounded call comfortably fits the 30s timeout and is easier to keep
  from hallucinating than an open-ended agentic loop.
- **Hallucination guard**: every recommendation URL is checked against the actual catalog after the LLM
  responds; anything not in the catalog is silently dropped rather than trusted. This is enforced in code,
  not just prompted for.
- **Stateless design**: full history is re-sent and re-parsed each call; there's no server-side session state,
  so a restart or load-balanced instance can't lose context.
