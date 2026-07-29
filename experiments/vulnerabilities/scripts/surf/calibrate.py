#!/usr/bin/env python3
"""Measure SURF's real per-candidate API cost from a completed calibration run.

SURF does not record token usage, so this reconstructs it. Judge and query-gen
prompts are re-rendered exactly as the EM loop built them and measured with
Anthropic's count_tokens endpoint (which is free and exact, unlike a chars/4
guess). Output tokens are measured the same way by counting the text the judge
actually produced, including its extended-thinking trace, which SURF stores in
score_metadata.thinking.

The resulting per-candidate figure is what sizes the real sweep against the
remaining Anthropic budget.

Run through Invoke-WithPetriSecrets so ANTHROPIC_API_KEY reaches this process
only.

Usage: calibrate.py <run-dir> [rubric-path]
"""
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "SURF"))

from surf.core.utils import render_jinja  # noqa: E402
from surf.em_loop.judge import get_principle_from_rubric, load_rubric  # noqa: E402
from surf.em_loop.prompts import OPUS_DIRECT_SCORE_PROMPT, QUERY_GEN_PROMPT  # noqa: E402

import anthropic  # noqa: E402

# claude-haiku-4-5 list price, USD per million tokens (docs/08 decision).
PRICE_IN = 1.00
PRICE_OUT = 5.00
MODEL = "claude-haiku-4-5"

run_dir = Path(sys.argv[1])
rubric_path = Path(sys.argv[2]) if len(sys.argv) > 2 else (
    Path(__file__).resolve().parents[2] / "seeds" / "surf-rubrics" / "harmful-omission.yaml"
)

records = [json.loads(l) for l in open(run_dir / "results.jsonl", encoding="utf-8") if l.strip()]
if not records:
    raise SystemExit(f"no scored candidates in {run_dir}/results.jsonl")

principle = get_principle_from_rubric(load_rubric(str(rubric_path)))
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def ntok(text: str) -> int:
    r = client.messages.count_tokens(
        model=MODEL, messages=[{"role": "user", "content": text or "-"}]
    )
    return r.input_tokens


judge_in, judge_out, query_in, query_out = [], [], [], []
resp_tokens, scores = [], []

for rec in records:
    query = rec.get("query", "")
    response = rec.get("response", "")
    meta = rec.get("score_metadata") or {}

    judge_prompt = render_jinja(
        OPUS_DIRECT_SCORE_PROMPT,
        principle_specific_details=principle,
        query=query,
        response=response,
    )
    judge_in.append(ntok(judge_prompt))

    # What the judge emitted: thinking trace + the scored answer.
    thinking = meta.get("thinking") or ""
    reasoning = meta.get("reasoning") or ""
    judge_out.append(ntok(thinking + reasoning) + 20)  # +20 for XML scaffolding

    attrs = rec.get("attributes", [])
    qprompt = render_jinja(
        QUERY_GEN_PROMPT, attributes_text="\n".join(f"- {a}" for a in attrs)
    )
    query_in.append(ntok(qprompt))
    query_out.append(ntok(query))  # stop token cuts generation after query_1

    resp_tokens.append(ntok(response))
    scores.append(rec.get("reward_score", 0))


def mean(xs):
    return statistics.mean(xs) if xs else 0.0


n = len(records)
m_ji, m_jo = mean(judge_in), mean(judge_out)
m_qi, m_qo = mean(query_in), mean(query_out)

cost_judge = (m_ji * PRICE_IN + m_jo * PRICE_OUT) / 1e6
cost_query = (m_qi * PRICE_IN + m_qo * PRICE_OUT) / 1e6
per_candidate = cost_judge + cost_query

print(f"calibration source : {run_dir}")
print(f"candidates scored  : {n}")
print()
print("MEAN TOKENS PER CANDIDATE")
print(f"  judge  in / out  : {m_ji:8.0f} / {m_jo:8.0f}")
print(f"  query  in / out  : {m_qi:8.0f} / {m_qo:8.0f}")
print(f"  target response  : {mean(resp_tokens):8.0f}  (GPU, already paid for)")
print()
print("MEAN API COST PER CANDIDATE (claude-haiku-4-5)")
print(f"  judge            : ${cost_judge:.5f}")
print(f"  query generation : ${cost_query:.5f}")
print(f"  TOTAL            : ${per_candidate:.5f}")
print()
print("SCORES")
print(f"  mean / max       : {mean(scores):.1f} / {max(scores):.1f}")
print(f"  flagged (>50)    : {sum(1 for s in scores if s > 50)}/{n}")
print()
print("PROJECTED SWEEP COST (API only)")
for label, runs, iters, cands in [
    ("SURF default 5x20x120", 5, 20, 120),
    ("3 runs x 20 x 60", 3, 20, 60),
    ("3 runs x 15 x 50", 3, 15, 50),
    ("2 runs x 15 x 50", 2, 15, 50),
]:
    total = runs * iters * cands
    print(f"  {label:24s} = {total:6,d} candidates -> ${total * per_candidate:7.2f}")

out = {
    "run_dir": str(run_dir),
    "n_candidates": n,
    "judge_model": MODEL,
    "mean_tokens": {
        "judge_in": m_ji, "judge_out": m_jo,
        "query_in": m_qi, "query_out": m_qo,
        "target_response": mean(resp_tokens),
    },
    "cost_usd_per_candidate": {
        "judge": cost_judge, "query": cost_query, "total": per_candidate,
    },
    "scores": {
        "mean": mean(scores), "max": max(scores),
        "n_flagged_gt50": sum(1 for s in scores if s > 50),
    },
}
dest = run_dir / "calibration.json"
json.dump(out, open(dest, "w", encoding="utf-8"), indent=2)
print(f"\nwritten: {dest}")
