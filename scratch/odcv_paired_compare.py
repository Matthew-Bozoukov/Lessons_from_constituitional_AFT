# ABOUTME: Paired per-scenario comparison of two ODCV arms, cancelling scenario variance.
# ABOUTME: Every arm runs the same cells, so the difference is far tighter than two CIs.
"""Does arm A differ from arm B on ODCV?

Two independent intervals are the wrong tool here. 21 of 30 incentivized scenarios never
fail and 3 always do, so an arm's rate is dominated by WHICH scenarios are in the set -- and
that variance is IDENTICAL for both arms, because every arm runs the same cells. Comparing
two marginal CIs throws that shared structure away and reports intervals ~25pp wide that
overlap no matter what.

This pairs on scenario instead. For each cell it takes each arm's violation fraction, forms
the per-cell difference, and bootstraps over CELLS. Scenario difficulty cancels in the
difference, so what remains is the effect. A paired interval excluding zero is evidence of a
real difference; a marginal-CI overlap is not evidence of no difference.

Also reports McNemar-style counts: cells where one arm fails and the other does not.

Run: uv run python scratch/odcv_paired_compare.py --a MATCHED_CONTROL --b TRAITS134
"""
import json
import random
import statistics
from pathlib import Path

import fire

from src.eval.misalignment.odcv.odcv import VIOLATION_THRESHOLD

ROOT = Path("output/odcv_bench")
ARMS = {
    "MATCHED_CONTROL":
        "qwen3_6-27b-lora-t2-9284-synthdoc-716-dynbatch-r64/combined5x_20260824_143424",
    "MATCHED_CONTROL_2X":
        "qwen3_6-27b-lora-t2-9284-synthdoc-716-dynbatch-r64/combined2x_20260820_223257",
    "TRAITS134":
        "qwen3_6-27b-lora-t2-9284-synthdoc-716-traits134-r64/combined6x_20260824_150853",
    "TRAITS134_3X":
        "qwen3_6-27b-lora-t2-9284-synthdoc-716-traits134-r64/combined3x_20260819_092850",
    "TRAITS123":
        "qwen3_6-27b-lora-9284-traits123-716-r64/combined6x_20260824_155722",
    "TRAITS123_3X":
        "qwen3_6-27b-lora-9284-traits123-716-r64/combined3x_20260819_221522",
    "TRAITS567":
        "qwen3_6-27b-lora-9284-traits567-716-r64/combined7x_20260824_165507",
    "TRAITS567_3X":
        "qwen3_6-27b-lora-9284-traits567-716-r64/combined3x_20260819_184957",
    "BOTH_PRUNED":
        "qwen3_6-27b-lora-t2-9284-synthdoc-662-bothpruned-r64/combined2x_20260823_221456",
    "BRANCHES":
        "qwen3_6-27b-lora-t2-9284-synthdoc-654-branches-r64/combined2x_20260824_101532",
    "ABLATED2":
        "qwen3_6-27b-lora-t2-9284-synthdoc-676-ablated2-r64/combined2x_20260822_220935",
}


def _cell_fractions(rel: str, variant: str) -> dict[str, float]:
    """Violation fraction per scenario: rollouts with median severity >= 3, over rollouts."""
    per: dict[str, list[float]] = {}
    for f in (ROOT / rel / "evaluations").glob("scores_*.json"):
        for k, v in json.loads(f.read_text()).items():
            s = v.get("score") if isinstance(v, dict) else v
            if isinstance(s, (int, float)) and k.startswith(variant):
                per.setdefault(k, []).append(s)
    by: dict[str, list[float]] = {}
    for k, vs in per.items():
        by.setdefault(k.rsplit("/rollout_", 1)[0], []).append(statistics.median(vs))
    return {k: sum(1 for x in v if x >= VIOLATION_THRESHOLD) / len(v) for k, v in by.items()}


def main(a: str = "MATCHED_CONTROL", b: str = "TRAITS134", variant: str = "incentivized",
         reps: int = 20000, seed: int = 0) -> None:
    """Report the paired per-scenario difference between two arms.

    Args:
        a: Baseline arm key.
        b: Comparison arm key.
        variant: "incentivized" or "mandated".
        reps: Bootstrap replicates.
        seed: RNG seed.
    """
    for k in (a, b):
        assert k in ARMS, f"unknown arm {k}; have {sorted(ARMS)}"
    fa, fb = _cell_fractions(ARMS[a], variant), _cell_fractions(ARMS[b], variant)
    # Only cells BOTH arms ran; a cell missing from either cannot be paired.
    cells = sorted(set(fa) & set(fb))
    only_a, only_b = sorted(set(fa) - set(fb)), sorted(set(fb) - set(fa))
    assert cells, "no shared cells"
    print(f"{variant}: {len(cells)} shared cells "
          f"({len(only_a)} only in {a}, {len(only_b)} only in {b})")

    ra = 100 * sum(fa[c] for c in cells) / len(cells)
    rb = 100 * sum(fb[c] for c in cells) / len(cells)
    diffs = [fb[c] - fa[c] for c in cells]
    obs = 100 * statistics.fmean(diffs)

    rng = random.Random(seed)
    boot = []
    for _ in range(reps):
        s = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        boot.append(100 * statistics.fmean(s))
    boot.sort()
    lo, hi = boot[int(.025 * reps)], boot[int(.975 * reps) - 1]
    p_two = 2 * min(sum(1 for x in boot if x <= 0), sum(1 for x in boot if x >= 0)) / reps

    print(f"\n  {a:18s} {ra:5.1f}%")
    print(f"  {b:18s} {rb:5.1f}%")
    print(f"\n  PAIRED difference ({b} - {a}): {obs:+.1f} pp")
    print(f"  95% CI [{lo:+.1f}, {hi:+.1f}]   width {hi-lo:.1f} pp"
          f"   {'EXCLUDES zero' if lo > 0 or hi < 0 else 'includes zero'}")
    print(f"  bootstrap two-sided p ~ {min(p_two, 1.0):.3f}")

    worse = [c for c in cells if fb[c] > fa[c]]
    better = [c for c in cells if fb[c] < fa[c]]
    print(f"\n  cells where {b} is worse: {len(worse)}   better: {len(better)}   "
          f"tied: {len(cells)-len(worse)-len(better)}")
    if worse:
        print(f"    worse : {', '.join(c.split('/')[-1] for c in worse[:6])}"
              f"{' …' if len(worse) > 6 else ''}")
    if better:
        print(f"    better: {', '.join(c.split('/')[-1] for c in better[:6])}"
              f"{' …' if len(better) > 6 else ''}")


if __name__ == "__main__":
    fire.Fire(main)
