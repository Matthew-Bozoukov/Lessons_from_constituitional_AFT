# ABOUTME: Find the run directories that belong to one experiment — and say why each other
# ABOUTME: one was excluded. Run: uv run python scratch/colosseum_find_runs.py <experiment>

"""Which directories make up an experiment, and which are debris.

By the end of a night of resubmissions `output/colosseum_jira/` holds a directory per
attempt, not per result: shards that were cancelled when the seed plan changed, jobs that
died on a port collision, arms that crashed naming their successor. Most carry some
episodes; only some carry a finished `results/per_seed.json`. Picking the right ones by
hand across fourteen directories is how a contrast quietly ends up over the wrong data.

So this classifies every directory and prints the verdict for each, rather than silently
returning a filtered list. A directory belongs to the experiment if it has per-seed
measures at all and its cells are exactly that experiment's cells — which is also what
distinguishes `collusion` from `single`, since both have a `baseline` cell and differ
only in the treated one.

Prints the qualifying directories last, space-separated, ready to paste into
`colosseum_pool_split_arms.py` or `colosseum_gate.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.eval.misalignment.colosseum.config import EXPERIMENTS

ROOT = Path("output/colosseum_jira")


def main(argv: list[str]) -> None:
    experiment = argv[1]
    expected = {name for name, _, _ in EXPERIMENTS[experiment]["cells"]}
    keep: list[Path] = []

    for d in sorted(ROOT.glob("*/")):
        if d.name in {"server", "pooled", "merged"} or d.name.startswith("server_"):
            continue
        per_seed = d / "results" / "per_seed.json"
        meta = d / "metadata" / "run_meta.json"
        episodes = len(list(d.rglob("metrics.json")))
        if not meta.is_file():
            # `_publish` MOVES run_meta.json from the root into metadata/ when an arm
            # finishes, so a root-level one means the arm is still running — which is a
            # "come back later", not a "this is debris". Worth separating at 1am.
            if (d / "run_meta.json").is_file():
                print(
                    f"  wait  {d.name[-42:]:44s} IN PROGRESS ({episodes} episodes so far)"
                )
            else:
                print(
                    f"  skip  {d.name[-42:]:44s} no run_meta.json at all "
                    f"({episodes} episodes) — died before writing one"
                )
            continue
        if not per_seed.is_file():
            print(
                f"  skip  {d.name[-42:]:44s} no per_seed.json "
                f"({episodes} episodes — the arm never finished harvesting)"
            )
            continue
        measures = json.loads(per_seed.read_text())
        cells: set[str] = set()
        for cells_of_measure in measures.values():
            cells |= set(cells_of_measure)
        if cells != expected:
            print(
                f"  skip  {d.name[-42:]:44s} cells {sorted(cells)} "
                f"!= {sorted(expected)} (different experiment)"
            )
            continue
        target = json.loads(meta.read_text())["target"].split("/")[-1]
        seeds = {c: len(s) for c, s in next(iter(measures.values())).items()}
        arm = "treatment" if "difficult" in target else "control  "
        print(f"  KEEP  {d.name[-42:]:44s} {arm} {seeds}")
        keep.append(d)

    print(f"\n{len(keep)} directories for {experiment}:\n")
    print(" ".join(str(p) for p in keep))


if __name__ == "__main__":
    main(sys.argv)
