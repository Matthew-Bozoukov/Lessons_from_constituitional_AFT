# ABOUTME: Pairs Claude's and GPT's answers to the SAME difficult-advice scenario (da-gptresp responder swap) and writes a
# ABOUTME: blinded, trait-stratified sample of N pairs as markdown (writers labelled X/Y, key kept apart) for close reading.
# Run: uv run python scratch/gpt_vs_claude_da/build_paired_sample.py --n 50 --seed 0
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--claude", default="data/da_gptresp_pool_source/stage_6_revise_responses.jsonl")
ap.add_argument("--gpt", nargs="+", default=[
    "output/synth_da_gptresp/20261124_012046/stage_4_revise_responses.jsonl",
    "output/synth_da_gptresp/20261124_020636/stage_4_revise_responses.jsonl"])
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="output/gpt_vs_claude_da/2026-11-24_paired_sample")
args = ap.parse_args()

read = lambda p: [json.loads(line) for line in open(p, encoding="utf-8")]
claude = {(r["system"], r["user"]): r for r in read(args.claude)}
gpt = {}
for p in args.gpt:
    for r in read(p):
        gpt[(r["system"], r["user"])] = r
keys = [k for k in gpt if k in claude]
assert len(keys) == len(gpt), f"{len(gpt) - len(keys)} GPT rows have no Claude counterpart"
pairs = [(claude[k], gpt[k]) for k in keys]

# Trait-stratified: round-robin over traits in shuffled order, so 50 pairs cover every trait about evenly.
rng = random.Random(args.seed)
by_trait = defaultdict(list)
for c, g in pairs:
    by_trait[c["trait_id"]].append((c, g))
for v in by_trait.values():
    rng.shuffle(v)
traits = sorted(by_trait); rng.shuffle(traits)
sample = []
while len(sample) < args.n:
    for t in traits:
        if by_trait[t] and len(sample) < args.n:
            sample.append(by_trait[t].pop())

# Blinding: one global assignment (consistent across pairs so systematic differences stay findable).
x_is_claude = rng.random() < 0.5
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
for i, (c, g) in enumerate(sample, 1):
    x, y = (c, g) if x_is_claude else (g, c)
    md = [f"# Pair {i:02d} (scenario {c['scenario_id']})", "",
          f"**Trait:** {c['trait_name']}", "", f"**Shortcut being tested (the tempting wrong move):** {c['shortcut']}", "",
          "## System prompt", "", c["system"], "", "## User message", "", c["user"], ""]
    for name, r in (("X", x), ("Y", y)):
        md += [f"## Writer {name}: reasoning (chain of thought)", "", r["reasoning"], "",
               f"## Writer {name}: reply to the user", "", r["response"], ""]
    (out / f"pair_{i:02d}.md").write_text("\n".join(md), encoding="utf-8")
json.dump({"x_is_claude": x_is_claude, "seed": args.seed, "n_pairs_pool": len(pairs),
           "pairs": [{"pair": i, "scenario_id": c["scenario_id"], "trait_id": c["trait_id"]} for i, (c, _) in enumerate(sample, 1)]},
          open(out.parent / "2026-11-24_paired_sample_KEY.json", "w"), indent=1)
print(f"{len(pairs)} matched pairs; wrote {len(sample)} to {out}; traits covered: {len({c['trait_id'] for c, _ in sample})}")
