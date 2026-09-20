# ABOUTME: One-off: seed a fresh synth run dir with the scenarios of finished da-multiagent-sprinkled runs,
# ABOUTME: so `synth run --resume <dir>` re-runs every stage after write_scenarios under the current prompts.
"""Seed a run directory from earlier runs' scenarios.

Run: uv run python scratch/da_multiagent_sprinkled/combine_scenarios.py <out dir> <run dir> [<run dir> ...] \
        [--multi N] [--single M]

Why: the first full run (2026-09-20) steered only `write_scenarios`, and the two prompt
stages after it rewrote most other-agent scenarios into one human asking one assistant. The
scenarios themselves were fine, so they are reused and only the stages after them re-run.

Copies stage 1 from the first run dir and writes stage 2 as, per principle, every
multi-agent scenario (the keyword rule, capped at --multi) plus up to --single of the
single-agent ones, which exist only to fill a principle the content filter leaves short.
Stage 3 (dedupe_scenarios) is left for the resume to compute over the combined set, which is
what applies the near-duplicate rule ACROSS the source runs. Writes combined_from.json.
"""

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from multiagent_rule import MULTI  # same directory

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("runs", nargs="+")
ap.add_argument("--multi", type=int, default=10**6)
ap.add_argument("--single", type=int, default=20)
args = ap.parse_args()

out = Path(args.out)
assert not out.exists(), f"{out} exists; a seeded dir is written once"
out.mkdir(parents=True)
shutil.copy2(Path(args.runs[0]) / "stage_1_chunk_constitution.jsonl", out)

multi, single = defaultdict(list), defaultdict(list)
seen = set()
for run in args.runs:
    for line in (
        (Path(run) / "stage_2_write_scenarios.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ):
        if not line.strip():
            continue
        r = json.loads(line)
        assert r["scenario_id"] not in seen, (
            f"scenario id collision: {r['scenario_id']}"
        )
        seen.add(r["scenario_id"])
        hit = MULTI.search(f"{r.get('situation') or ''} {r.get('shortcut') or ''}")
        (multi if hit else single)[r["trait_id"]].append(r)

rng = random.Random(0)
kept, counts = [], {}
for t in sorted(set(multi) | set(single), key=lambda x: int(x.lstrip("t"))):
    rng.shuffle(multi[t])
    rng.shuffle(single[t])
    take_m, take_s = multi[t][: args.multi], single[t][: args.single]
    kept += take_m + take_s
    counts[t] = {"multiagent": len(take_m), "single_agent": len(take_s)}

with open(out / "stage_2_write_scenarios.jsonl", "w", encoding="utf-8") as f:
    for r in kept:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
record = {
    "source_runs": args.runs,
    "multi_cap": args.multi,
    "single_cap": args.single,
    "scenarios": len(kept),
    "per_trait": counts,
    "by_trait_total": dict(Counter(r["trait_id"] for r in kept)),
}
(out / "combined_from.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record, indent=2))
