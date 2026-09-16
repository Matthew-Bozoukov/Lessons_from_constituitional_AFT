# ABOUTME: List the (condition, block, seed) cells the plan asks for that have no finished
# ABOUTME: episode yet, and print the queue commands that would run them.

"""Which seeds are missing, and the commands to top them up.

    uv run python scratch/colosseum_hospital/topup_plan.py [--analysis output/colosseum_hospital/analysis/episodes.json]

Reads the analysis' episode rows (run analyse.py first) and compares them with the plan:
baseline 1-30, self_promotional 1-60, self_sacrificial 1-30, covert 1-30, each in both
blocks. Prints one queue job per (condition, block) with the missing seeds as an explicit
comma list, which `run_hospital_queue.sh` accepts in place of a lo-hi range.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CONTROL = "LASR-Callum/2026-08-04-qwen36-lora-table2-only-9284-rank-64"
TREATMENT = (
    "LASR-Callum/2026-08-21-qwen36-lora-table2-9284-difficult-advice-"
    "chunk-only-702-rank-64-dynbatch"
)
PLAN = {
    "baseline": range(1, 31),
    "self_promotional": range(1, 61),
    "self_sacrificial": range(1, 31),
    "covert": range(1, 31),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--analysis", default="output/colosseum_hospital/analysis/episodes.json"
    )
    ap.add_argument(
        "--conditions", default="baseline,self_promotional,self_sacrificial"
    )
    args = ap.parse_args()
    rows = json.loads(Path(args.analysis).read_text())
    have = {(r["condition"], r["block"], int(r["seed"])) for r in rows}
    for condition in args.conditions.split(","):
        for block, target in (("control", CONTROL), ("treatment", TREATMENT)):
            missing = [s for s in PLAN[condition] if (condition, block, s) not in have]
            done = sum(1 for s in PLAN[condition] if (condition, block, s) in have)
            print(
                f"{condition:18s} {block:9s} done {done:2d}/{len(PLAN[condition])}  missing {len(missing):2d}: {missing}"
            )
            if missing:
                seeds = ",".join(str(s) for s in missing)
                print(f"    {condition}:{seeds}:{target}")


if __name__ == "__main__":
    main()
