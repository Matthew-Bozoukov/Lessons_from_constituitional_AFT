# ABOUTME: Join the LESS influence ranking to its scoring pool and describe what the
# ABOUTME: top/bottom slices differ on BEFORE any SAE work — trait, domain, length.

"""What is already visible in the LESS ranking from metadata alone?

    uv run --project scratch/sae_properties python scratch/sae_properties/less_split.py

The point is confound triage. Dataset diffing will happily rediscover any attribute that
correlates with the split (trait, domain, length), so the cheap move is to measure those
first and decide what the SAE is actually being asked to explain.

Writes `split.jsonl` (pool rows + rank + scores, ready to embed) alongside the report.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

LESS_REPO = "LASR-Callum/2026-08-14-less-selection-difficult-advice"
POOL_REPO, POOL_FILE = "matboz/synthdoc-v2-difficult-advice", "stage_7_sft.jsonl"


def load() -> list[dict]:
    """Pool rows joined to their LESS rank/score by less_id (scenario_id#pool_index)."""
    scores = [json.loads(l) for l in
              open(hf_hub_download(LESS_REPO, "scores/scores.jsonl", repo_type="dataset"))]
    pool = [json.loads(l) for l in
            open(hf_hub_download(POOL_REPO, POOL_FILE, repo_type="dataset"))]
    for i, r in enumerate(pool):
        r["metadata"]["less_id"] = f"{r['metadata'].get('scenario_id', 'row')}#{i}"
    by_id = {r["metadata"]["less_id"]: r for r in pool}
    joined = []
    for s in scores:
        r = by_id.get(s["less_id"])
        if r is None:
            continue
        joined.append({**r, "less": s})
    if len(joined) != len(scores):
        raise SystemExit(f"join lost rows: {len(joined)} of {len(scores)} — id scheme changed?")
    return joined


def text_of(row: dict, channel: str) -> str:
    msgs = row["messages"]
    if channel == "query":
        return next((m.get("content") or "" for m in msgs if m.get("role") == "user"), "")
    asst = [m for m in msgs if m.get("role") == "assistant"]
    if channel == "reasoning":
        return "\n\n".join(m["reasoning_content"] for m in asst if m.get("reasoning_content"))
    return (asst[-1].get("content") or "") if asst else ""


def dist(rows: list[dict], key: str, top: int = 8) -> list[tuple[str, float]]:
    c = Counter(str(r["metadata"].get(key)) for r in rows)
    n = max(1, len(rows))
    return [(k, v / n) for k, v in c.most_common(top)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.10, help="slice size as a fraction of the pool")
    ap.add_argument("--out", default="output/sae_properties/less_split")
    args = ap.parse_args()

    rows = load()
    rows.sort(key=lambda r: r["less"]["rank"])  # rank 0 = most influential
    k = int(len(rows) * args.frac)
    top, bottom = rows[:k], rows[-k:]
    print(f"pool {len(rows)} rows · slice {k} ({args.frac:.0%}) each end\n")

    for key in ("trait_id", "domain"):
        pool_d = dict(dist(rows, key, 40))
        print(f"--- {key}: share in TOP / BOTTOM / pool (top 8 by top-share) ---")
        for name, share in dist(top, key, 8):
            b = dict(dist(bottom, key, 40)).get(name, 0.0)
            p = pool_d.get(name, 0.0)
            lift = share / p if p else float("inf")
            print(f"  {name[:44]:46s} top {share:6.1%}  bot {b:6.1%}  pool {p:6.1%}  lift {lift:4.1f}x")
        print()

    for ch in ("query", "reasoning", "response"):
        tl = [len(text_of(r, ch)) for r in top]
        bl = [len(text_of(r, ch)) for r in bottom]
        med = lambda xs: sorted(xs)[len(xs) // 2] if xs else 0
        print(f"{ch:10s} median chars  top {med(tl):6d}  bottom {med(bl):6d}  "
              f"ratio {med(tl) / max(1, med(bl)):.2f}")

    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "split.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {out / 'split.jsonl'} ({len(rows)} rows, rank-ordered)")


if __name__ == "__main__":
    main()
