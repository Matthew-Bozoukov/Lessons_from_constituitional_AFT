# ABOUTME: Per-arm ODCV rates with the rollout-level (unclustered) interval, plus the
# ABOUTME: scenario-clustered one beside it, so the two estimands are never confused.
"""Two intervals for the same rate, because they answer different questions.

UNCLUSTERED (Wilson / bootstrap over rollouts) treats every rollout as an independent
sample. It answers "what rate does this model produce on THIS fixed 30-scenario benchmark",
narrows as sqrt(n), and is the right interval for tracking one model across reruns or
comparing arms that all run the identical cell set.

CLUSTERED (bootstrap over scenarios) answers "what rate would this model produce on
ODCV-like scenarios in general". It does NOT narrow with more rollouts: 21 of 30 incentivized
cells never fail and 3 always do, so the rate is set by which scenarios are in the set.

Neither is wrong; a figure just has to say which it uses.

Run: uv run python scratch/odcv_unclustered_ci.py --variant incentivized
"""
import json
import random
import statistics
from pathlib import Path

import fire

from src.eval.misalignment.odcv.odcv import VIOLATION_THRESHOLD, misalignment_rate
from src.utils import wilson

ROOT = Path("output/odcv_bench")
ARMS = [
    ("matched control",
     "qwen3_6-27b-lora-t2-9284-synthdoc-716-dynbatch-r64/combined5x_20260824_143424"),
    ("traits 1/3/4 removed",
     "qwen3_6-27b-lora-t2-9284-synthdoc-716-traits134-r64/combined6x_20260824_150853"),
    ("traits 1/2/3 only",
     "qwen3_6-27b-lora-9284-traits123-716-r64/combined6x_20260824_155722"),
    ("traits 5/6/7 only",
     "qwen3_6-27b-lora-9284-traits567-716-r64/combined7x_20260824_165507"),
    ("+ branches",
     "qwen3_6-27b-lora-t2-9284-synthdoc-654-branches-r64/combined6x_20260824_184140"),
    ("+ alt & counterarg",
     "qwen3_6-27b-lora-t2-9284-synthdoc-662-bothpruned-r64/combined5x_20260824_222130"),
    ("ablated2 (INERT)",
     "qwen3_6-27b-lora-t2-9284-synthdoc-676-ablated2-r64/combined2x_20260822_220935"),
    ("+ counterargument only",
     "qwen3_6-27b-lora-t2-9284-synthdoc-676-ecrestored-r64/combined5x_20260824_222130"),
    ("+ alternative only",
     "qwen3_6-27b-lora-t2-9284-synthdoc-676-altrestored-r64/combined5x_20260824_194619"),
]


def _medians(rel: str, variant: str) -> dict[str, list[float]]:
    """Median-across-judges severity per rollout, grouped by scenario."""
    per: dict[str, list[float]] = {}
    for f in (ROOT / rel / "evaluations").glob("scores_*.json"):
        for k, v in json.loads(f.read_text()).items():
            s = v.get("score") if isinstance(v, dict) else v
            if isinstance(s, (int, float)) and k.startswith(variant):
                per.setdefault(k, []).append(s)
    by: dict[str, list[float]] = {}
    for k, vs in per.items():
        by.setdefault(k.rsplit("/rollout_", 1)[0], []).append(statistics.median(vs))
    return by


def main(variant: str = "incentivized", reps: int = 20000, seed: int = 0) -> None:
    """Print both intervals for every arm with a judged run.

    Args:
        variant: "incentivized" or "mandated".
        reps: Bootstrap replicates for the clustered interval.
        seed: RNG seed.
    """
    print(f"{variant} — every rollout treated identically (unclustered), "
          f"clustered shown for contrast\n")
    print(f"{'arm':22s} {'cells':>5s} {'n':>5s} {'MR':>6s}   "
          f"{'UNCLUSTERED (Wilson)':>22s} {'w':>5s}   {'CLUSTERED (scenario)':>22s} {'w':>5s}")
    for label, rel in ARMS:
        p = ROOT / rel / "evaluations"
        if not p.is_dir():
            print(f"{label:22s}  (no judged run)")
            continue
        by = _medians(rel, variant)
        vals = [v for vs in by.values() for v in vs]
        if not vals:
            print(f"{label:22s}  (no {variant} rollouts)")
            continue
        hits = sum(1 for v in vals if v >= VIOLATION_THRESHOLD)
        n = len(vals)
        mr = 100 * hits / n
        lo_w, hi_w = (100 * x for x in wilson(hits, n))
        ks = sorted(by)
        rng = random.Random(seed)
        boot = []
        for _ in range(reps):
            s = [v for _ in ks for v in by[ks[rng.randrange(len(ks))]]]
            boot.append(misalignment_rate(s))
        boot.sort()
        lo_c, hi_c = boot[int(.025 * reps)], boot[int(.975 * reps) - 1]
        print(f"{label:22s} {len(by):5d} {n:5d} {mr:5.1f}%   "
              f"[{lo_w:6.1f}, {hi_w:6.1f}] {hi_w-lo_w:5.1f}   "
              f"[{lo_c:6.1f}, {hi_c:6.1f}] {hi_c-lo_c:5.1f}")


if __name__ == "__main__":
    fire.Fire(main)
