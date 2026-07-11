"""
Local dev harness -- replays each trace's real user turns against our own
agent (not the gold agent replies), feeding our actual responses back into
history each turn. This is a stand-in for SHL's LLM-simulated-user replay
harness: it won't perfectly reproduce their eval (their simulated user
reacts to what OUR agent asks, using free text, not a fixed script), but it
catches obvious regressions cheaply and computes Recall@10 against the gold
final shortlist per trace.

Usage:
    export GROQ_API_KEY=...
    python tests/replay_harness.py
"""
import glob
import re
import sys
import time
from pathlib import Path

# Raised from 3->20. Each call is far larger than it looks: system prompt
# (~700 tok) + up to ~2 x top_k candidate lines with descriptions + the
# ENTIRE conversation history so far (re-sent every turn, since agent.py is
# stateless) + max_tokens=1024 reserved for the reply. On a long trace
# (e.g. C9.md's 7 turns) later calls are meaningfully bigger than earlier
# ones. Against Groq's free-tier 12K TPM budget, 3s between calls isn't
# enough headroom to avoid stacking into the same per-minute window --
# that's almost certainly what caused every trace to hit the except-block
# fallback and return recommendations=[] on both prior full-harness runs.
# If you still see failures at 20s, check the new stderr logging in
# agent.py's except block for the real exception (rate limit vs. something
# else) before raising this further.
SECONDS_BETWEEN_CALLS = 20

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.schemas import Message
from app.agent import run_chat

TURN_RE = re.compile(
    r"\*\*User\*\*\s*\n\s*>\s*(?P<user>.+?)\n\n\*\*Agent\*\*", re.S
)
GOLD_URL_RE = re.compile(r"<(https://[^\s>]+)>")


def parse_trace(path: str):
    """Extract (user_turns, gold_final_urls) from a trace markdown file."""
    text = open(path).read()
    user_turns = [m.group("user").strip() for m in TURN_RE.finditer(text)]
    # Gold final shortlist = URLs in the LAST table found in the file.
    tables = text.split("| # | Name |")
    gold_urls = []
    if len(tables) > 1:
        last_table = tables[-1]
        gold_urls = GOLD_URL_RE.findall(last_table)
    return user_turns, gold_urls


def recall_at_10(predicted_urls, gold_urls):
    if not gold_urls:
        return None
    hit = len(set(predicted_urls[:10]) & set(gold_urls))
    return hit / len(gold_urls)


def run_trace(path: str):
    user_turns, gold_urls = parse_trace(path)
    history = []
    last_recs = []
    for i, turn_text in enumerate(user_turns, 1):
        print(f"  [{Path(path).name}] turn {i}/{len(user_turns)}...", flush=True)
        history.append(Message(role="user", content=turn_text))
        time.sleep(SECONDS_BETWEEN_CALLS)
        resp = run_chat(history)
        history.append(Message(role="assistant", content=resp.reply))
        if resp.recommendations:
            last_recs = [r.url for r in resp.recommendations]
        if resp.end_of_conversation:
            break
    recall = recall_at_10(last_recs, gold_urls)
    return {
        "trace": Path(path).name,
        "turns_used": len(history) // 2,
        "gold_count": len(gold_urls),
        "predicted_count": len(last_recs),
        "recall_at_10": recall,
    }


def main():
    results = []
    for path in sorted(glob.glob("data/traces/*.md")):
        try:
            results.append(run_trace(path))
        except Exception as e:
            results.append({"trace": Path(path).name, "error": str(e)})

    print(f"{'trace':10s} {'turns':6s} {'gold':5s} {'pred':5s} {'recall@10':10s}")
    recalls = []
    for r in results:
        if "error" in r:
            print(f"{r['trace']:10s} ERROR: {r['error']}")
            continue
        rec_str = f"{r['recall_at_10']:.2f}" if r["recall_at_10"] is not None else "n/a"
        if r["recall_at_10"] is not None:
            recalls.append(r["recall_at_10"])
        print(f"{r['trace']:10s} {r['turns_used']:<6d} {r['gold_count']:<5d} {r['predicted_count']:<5d} {rec_str:10s}")

    if recalls:
        print(f"\nMean Recall@10 across {len(recalls)} traces: {sum(recalls)/len(recalls):.3f}")


if __name__ == "__main__":
    main()