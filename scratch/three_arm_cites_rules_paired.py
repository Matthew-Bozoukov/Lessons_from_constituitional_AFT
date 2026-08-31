# ABOUTME: Cell-paired check that the full 9-trait arm's rule-citation rate really sits between
# ABOUTME: the two trait-subset arms, plus the restricted judged sample sizes.

"""Verify the cites_rules ordering cell by cell, and report restricted n."""

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import fire

from scratch.compare_trait_arm_reasoning import FAMILIES, load_arm
from scratch.three_arm_reasoning_compare import RESULTS
from scratch.three_arm_scenario_overlap import ARMS3


def main(family: str = "cites_rules") -> None:
    """Print cell-paired means and pairwise win counts for one stance family."""
    loaded = {a: load_arm(Path(p)) for a, p in ARMS3.items()}
    cells = {a: {(r["condition"], r["scenario"]) for r in recs} for a, recs in loaded.items()}
    keep = set.intersection(*cells.values())

    rx = re.compile(FAMILIES[family], re.I)
    per = {a: defaultdict(list) for a in ARMS3}
    for a, recs in loaded.items():
        for r in recs:
            k = (r["condition"], r["scenario"])
            if k in keep and r["reasons"]:
                per[a][k].append(bool(rx.search(" ".join(r["reasons"]))))

    means = {a: {k: 100 * sum(v) / len(v) for k, v in per[a].items()} for a in ARMS3}
    print(f"=== {family}: cell-averaged over {len(keep)} shared cells ===")
    for a in ARMS3:
        print(f"  {a:<10} {statistics.mean(means[a].values()):5.1f}%")

    order = list(ARMS3)
    print("\npairwise cells higher (X > Y / Y > X / tied):")
    for i, x in enumerate(order):
        for y in order[i + 1:]:
            gx = sum(means[x][k] > means[y][k] for k in keep)
            gy = sum(means[y][k] > means[x][k] for k in keep)
            print(f"  {x} vs {y}: {gx} / {gy} / {len(keep) - gx - gy}")

    print("\nrestricted judged sample sizes:")
    for a, p in RESULTS.items():
        r = json.load(open(p))
        n = sum(1 for c in ("mandated", "incentivized")
                for k in r["per_scenario_medians"].get(c, {})
                if (c, k.split("/")[0]) in keep)
        print(f"  {a:<10} {n} rollout-level medians")


if __name__ == "__main__":
    fire.Fire(main)
