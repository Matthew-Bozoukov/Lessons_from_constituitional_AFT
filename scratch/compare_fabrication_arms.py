# ABOUTME: Compare two fabrication-scenario runs family by family with Fisher exact tests
# ABOUTME: and Wilson intervals, so small-n differences are not over-read.

"""Paired comparison of two arms on the same scenario set.

Run: uv run python scratch/compare_fabrication_arms.py <run-dir-A> <run-dir-B>

Both runs must use the same scenarios, judge and sampling settings. Rates here come from
24-36 samples per family, which is wide enough that a several-point gap is indistinguishable
from noise — hence the intervals and the per-scenario view, which is where a real effect
shows up as a scenario flipping wholesale rather than as a family drifting.
"""

import glob
import json
import sys
from collections import defaultdict

from scipy.stats import beta, fisher_exact

FAMS = ["hard_constraint", "false_precision", "sycophantic", "fabricated_cites",
        "pseudoscience_formalism"]


def load(d: str) -> list[dict]:
    """Read the graded records of a run directory."""
    return [r for r in (json.loads(l) for l in open(f"{d}/judged.jsonl")) if "fabricated" in r]


def rate(rs: list[dict]) -> tuple[int, int]:
    """Return (fabricated, n)."""
    return sum(r["fabricated"] for r in rs), len(rs)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial CI; avoids a statsmodels dependency for one function."""
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def main() -> None:
    """Print the family-level and per-scenario comparison."""
    da, db = sys.argv[1], sys.argv[2]
    A, B = load(da), load(db)
    na = json.load(open(f"{da}/run_meta.json"))["target"]
    nb = json.load(open(f"{db}/run_meta.json"))["target"]

    print(f"A = {na}  (n={len(A)})")
    print(f"B = {nb}  (n={len(B)})\n")
    print(f"{'family':26s} {na:>18s} {nb:>18s}   Fisher p")
    print("-" * 78)
    for fam in FAMS + ["ALL"]:
        ra = [r for r in A if fam == "ALL" or r["family"] == fam]
        rb = [r for r in B if fam == "ALL" or r["family"] == fam]
        if not ra or not rb:
            continue
        fa, va = rate(ra)
        fb, vb = rate(rb)
        ca = clopper_pearson(fa, va)
        cb = clopper_pearson(fb, vb)
        p = fisher_exact([[fa, va - fa], [fb, vb - fb]])[1]
        star = " *" if p < 0.05 else ""
        print(f"{fam:26s} {fa:2d}/{va:2d} {fa/va*100:5.1f}% "
              f"[{ca[0]*100:4.0f}-{ca[1]*100:3.0f}] "
              f"{fb:2d}/{vb:2d} {fb/vb*100:5.1f}% [{cb[0]*100:4.0f}-{cb[1]*100:3.0f}]"
              f"   p={p:.3f}{star}")

    print(f"\nknew_better: {na} {sum(r['knew_better'] for r in A)}  |  "
          f"{nb} {sum(r['knew_better'] for r in B)}")

    print("\nPER-SCENARIO (only where the arms differ by >=2 of 3 samples)")
    ba, bb = defaultdict(list), defaultdict(list)
    for r in A:
        ba[r["id"]].append(r["fabricated"])
    for r in B:
        bb[r["id"]].append(r["fabricated"])
    for sid in sorted(set(ba) & set(bb)):
        x, y = sum(ba[sid]), sum(bb[sid])
        if abs(x - y) >= 2:
            print(f"  {sid}: {na} {x}/{len(ba[sid])}   {nb} {y}/{len(bb[sid])}")


if __name__ == "__main__":
    main()
