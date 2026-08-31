# ABOUTME: Pass-to-pass ODCV MR for one arm, computed on the scenario core common to the
# ABOUTME: full passes, plus the scenario-level flip rate that MR stability can hide.
"""How much does ODCV MR move between rollout passes of the SAME adapter?

    uv run python scratch/stats/odcv_pass_stability.py --arm <odcv_bench subdir>

Two traps this avoids:

1. A "combined4x" dir is not always four re-rolls. Some passes are partial TOP-UPS that
   cover only the scenarios an earlier pass missed. Comparing a 12-scenario top-up against
   a 33-scenario pass measures the scenario list, not the model, so partial passes are
   reported separately rather than folded into the spread.
2. MR is an aggregate: it can look stable across passes while individual scenarios flip
   violation status in both directions and cancel. Both are reported.

Full passes are those covering >= `full_frac` of the largest scenario count seen for that
variant; MR is then computed on the core those passes share, so every pass is scored on an
identical scenario list.
"""

from __future__ import annotations

import glob
import json
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import fire
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.eval.misalignment.odcv.odcv import (  # noqa: E402
    VARIANTS, VIOLATION_THRESHOLD, median_score, misalignment_rate,
)

KEY = re.compile(r"^(?P<variant>[^/]+)/(?P<scenario>[^/]+)/rollout_(?P<idx>\d+)$")


def _load(eval_dir: Path) -> dict:
    """{rollout_idx: {variant: {scenario: [judge scores]}}}."""
    files = sorted(glob.glob(str(eval_dir / "scores_*.json")))
    assert files, f"no judge score files under {eval_dir}"
    per: dict = defaultdict(lambda: {v: defaultdict(list) for v in VARIANTS})
    for f in files:
        for key, val in json.load(open(f)).items():
            m = KEY.match(key)
            if not m or m["variant"] not in VARIANTS:
                continue
            score = val.get("score") if isinstance(val, dict) else val
            if score is not None:
                per[m["idx"]][m["variant"]][m["scenario"]].append(float(score))
    return per


def _report(label: str, cells: dict, full_frac: float) -> None:
    """Print per-pass MR on the shared core, plus flip rates, for one variant."""
    sizes = {i: len(c) for i, c in cells.items() if c}
    if len(sizes) < 2:
        print(f"--- {label}: only {len(sizes)} pass(es) present, skipped\n")
        return
    biggest = max(sizes.values())
    full = [i for i, n in sizes.items() if n >= full_frac * biggest]
    partial = sorted(set(sizes) - set(full))
    core = sorted(set.intersection(*(set(cells[i]) for i in full)))
    assert core, f"{label}: full passes {full} share no scenarios"

    print(f"--- {label}")
    print(f"    full passes {full} (n={[sizes[i] for i in full]}), "
          f"shared core = {len(core)} scenarios")
    if partial:
        print(f"    partial top-up passes EXCLUDED from spread: "
              + ", ".join(f"p{i} (n={sizes[i]})" for i in partial))

    mrs, flags = {}, {}
    for i in full:
        med = {s: median_score(cells[i][s]) for s in core}
        mrs[i] = misalignment_rate(list(med.values()))
        flags[i] = {s: v >= VIOLATION_THRESHOLD for s, v in med.items()}
    print("    MR on shared core: " + "  ".join(f"p{i}={mrs[i]:.1f}%" for i in full))
    deltas = [abs(mrs[a] - mrs[b]) for a, b in combinations(full, 2)]
    print(f"    pairwise |dMR|: " + " ".join(f"{d:.1f}" for d in deltas)
          + f"   (mean {np.mean(deltas):.1f}pp, max {max(deltas):.1f}pp)")
    print(f"    range {min(mrs.values()):.1f}-{max(mrs.values()):.1f}%   "
          f"SD={np.std(list(mrs.values()), ddof=1):.2f}pp")

    stable = sum(len({flags[i][s] for i in full}) == 1 for s in core)
    flips = len(core) - stable
    always = sum(all(flags[i][s] for i in full) for s in core)
    never = sum(not any(flags[i][s] for i in full) for s in core)
    disag = [100.0 * sum(flags[a][s] != flags[b][s] for s in core) / len(core)
             for a, b in combinations(full, 2)]
    print(f"    scenario verdicts: {always} always violating, {never} never, "
          f"{flips} UNSTABLE ({100*flips/len(core):.0f}% of core)")
    print(f"    mean pairwise scenario disagreement: {np.mean(disag):.1f}%\n")


def main(arm: str = "qwen3_6-27b-lora-9284-numina-control-716-r64",
         full_frac: float = 0.8) -> None:
    """Print per-pass MR on a fixed scenario core, and scenario flip rates."""
    root = REPO / "output/odcv_bench" / arm
    cands = sorted(root.glob("*/evaluations"))
    assert cands, f"no evaluations/ under {root}"
    eval_dir = max(cands, key=lambda p: len(_load(p)))
    per = _load(eval_dir)
    print(f"arm      : {arm}")
    print(f"eval dir : {eval_dir.relative_to(REPO)}\n")

    for v in VARIANTS:
        _report(v, {i: dict(per[i][v]) for i in sorted(per)}, full_frac)

    both = {i: ({f"{v}/{s}": js for v in VARIANTS for s, js in per[i][v].items()}
                if all(per[i][v] for v in VARIANTS) else {}) for i in sorted(per)}
    _report("overall", both, full_frac)


if __name__ == "__main__":
    fire.Fire(main)
