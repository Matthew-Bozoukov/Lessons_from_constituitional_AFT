# ABOUTME: ODCV MR standard errors following Miller 2024 (arXiv:2411.00640): average the K
# ABOUTME: rollout passes per scenario first, then take the CLT SE across scenarios.
"""Miller-style standard errors for ODCV MR.

    uv run python scratch/stats/odcv_miller_se.py --arm <odcv_bench subdir>

Miller's recipe for resampled evals: with K answers per question, the question score s_i is
the MEAN of the K scores, and the SE is taken across questions -- never pooled over all K*n
answers, which would violate independence. Here a "question" is an ODCV scenario and the K
answers are the rollout passes, so s_i in {0, 1/K, ..., 1}.

This matters because s_i is then FRACTIONAL, and the paper's named mistake -- the Llama 3
report "incorrectly estimates all standard errors using SE_Bernoulli, even when s_i takes
fractional values" -- is exactly what sqrt(p(1-p)/n) does to this data. Both are printed.
"""

from __future__ import annotations

import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import fire
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.eval.misalignment.odcv.odcv import (  # noqa: E402
    VARIANTS, VIOLATION_THRESHOLD, median_score,
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


def main(arm: str = "qwen3_6-27b-lora-9284-numina-control-716-r64",
         full_frac: float = 0.8) -> None:
    """Print Miller-style CLT SEs beside the naive Bernoulli SE for one arm."""
    root = REPO / "output/odcv_bench" / arm
    cands = sorted(root.glob("*/evaluations"))
    assert cands, f"no evaluations/ under {root}"
    eval_dir = max(cands, key=lambda p: len(_load(p)))
    per = _load(eval_dir)
    print(f"arm      : {arm}")
    print(f"eval dir : {eval_dir.relative_to(REPO)}\n")

    # s_i per scenario = share of full passes whose median severity is a violation.
    si: dict[str, list[float]] = {}
    for v in VARIANTS:
        sizes = {i: len(per[i][v]) for i in per if per[i][v]}
        if len(sizes) < 2:
            continue
        full = [i for i, n in sizes.items() if n >= full_frac * max(sizes.values())]
        core = sorted(set.intersection(*(set(per[i][v]) for i in full)))
        for s in core:
            si[f"{v}/{s}"] = [
                float(median_score(per[i][v][s]) >= VIOLATION_THRESHOLD) for i in full]

    def block(label: str, keys: list[str]) -> None:
        if not keys:
            return
        arr = np.array([np.mean(si[k]) for k in keys], float)
        n, K = len(arr), len(si[keys[0]])
        mr = 100.0 * arr.mean()
        se_clt = 100.0 * arr.std(ddof=1) / np.sqrt(n)
        p = arr.mean()
        se_bern = 100.0 * np.sqrt(p * (1 - p) / n)
        frac = int(np.sum((arr > 0) & (arr < 1)))
        print(f"--- {label}   n={n} scenarios, K={K} passes each")
        print(f"    MR                       = {mr:.2f}%")
        print(f"    SE  (Miller CLT, s_i mean over K) = {se_clt:.2f}pp")
        print(f"    SE  (naive Bernoulli)             = {se_bern:.2f}pp"
              f"   [{'OVERSTATES' if se_bern > se_clt else 'understates'}"
              f" by {abs(se_bern - se_clt) / se_clt * 100:.0f}%]")
        print(f"    report as                = {mr:.1f}% ({se_clt:.1f}%)   "
              f"[Miller's parenthetical-SE format]")
        print(f"    95% CI = +/-1.96*SE      = [{mr - 1.96*se_clt:.1f}, {mr + 1.96*se_clt:.1f}]"
              f"   (width {2*1.96*se_clt:.1f}pp)")
        print(f"    +/-1 SEM bar             = [{mr - se_clt:.1f}, {mr + se_clt:.1f}]"
              f"   (~68%, NOT a 95% interval)")
        print(f"    scenarios with fractional s_i (unstable across passes): {frac}/{n}\n")

    block("overall", sorted(si))
    for v in VARIANTS:
        block(v, sorted(k for k in si if k.startswith(f"{v}/")))

    # Miller's clustering point: scenarios are grouped by variant, and a between-cluster
    # difference in rate inflates the SE relative to treating all scenarios as independent.
    groups = [np.array([np.mean(si[k]) for k in sorted(si) if k.startswith(f"{v}/")])
              for v in VARIANTS]
    groups = [g for g in groups if len(g)]
    if len(groups) > 1:
        allv = np.concatenate(groups)
        n = len(allv)
        gm = [g.mean() for g in groups]
        between = sum(len(g) * (m - allv.mean()) ** 2 for g, m in zip(groups, gm))
        print(f"cluster check (variant as cluster): rates {[f'{100*m:.1f}%' for m in gm]}, "
              f"between-cluster SS = {between:.3f}")
        print("  Miller Sec 2.2: cluster when questions share structure; a large between-"
              "cluster\n  spread means the independent-scenario SE above is too small.")


if __name__ == "__main__":
    fire.Fire(main)
