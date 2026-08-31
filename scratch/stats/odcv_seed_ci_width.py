# ABOUTME: Projects ODCV MR confidence-interval width for k training seeds, anchored on
# ABOUTME: the measured per-rollout MR spread of one arm (default: numina control).
"""How wide would an ODCV MR interval be with k training seeds?

    uv run python scratch/stats/odcv_seed_ci_width.py --arm <odcv_bench subdir>

With k seeds the arm-level interval is a t-interval on k per-seed MR estimates:

    half-width = t(0.975, k-1) * s_seed / sqrt(k)

s_seed (the between-seed SD of MR) cannot be measured from a single trained adapter, so
this script measures what it CAN -- the eval-pass SD of the same adapter re-rolled, which
is a hard FLOOR on s_seed -- and then reports the width as a function of s_seed.
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
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.eval.misalignment.odcv.odcv import (  # noqa: E402
    VARIANTS, VIOLATION_THRESHOLD, median_score, misalignment_rate,
)

KEY = re.compile(r"^(?P<variant>[^/]+)/(?P<scenario>[^/]+)/rollout_(?P<idx>\d+)$")


def _load(eval_dir: Path) -> dict:
    """{rollout_idx: {variant: {scenario: [judge scores]}}} for one run dir."""
    files = sorted(glob.glob(str(eval_dir / "scores_*.json")))
    assert files, f"no judge score files under {eval_dir}"
    per: dict = defaultdict(lambda: {v: defaultdict(list) for v in VARIANTS})
    for f in files:
        for key, val in json.load(open(f)).items():
            m = KEY.match(key)
            if not m or m["variant"] not in VARIANTS:
                continue
            score = val.get("score") if isinstance(val, dict) else val
            if score is None:
                continue
            per[m["idx"]][m["variant"]][m["scenario"]].append(float(score))
    return per


def _mr(cell: dict) -> tuple[float, int]:
    """MR% and scenario count for {scenario: [judge scores]}."""
    med = [median_score(v) for v in cell.values() if v]
    return (misalignment_rate(med), len(med)) if med else (float("nan"), 0)


def main(arm: str = "qwen3_6-27b-lora-9284-numina-control-716-r64",
         k: int = 3) -> None:
    """Report measured per-rollout MR spread and projected k-seed CI widths."""
    root = REPO / "output/odcv_bench" / arm
    cands = sorted(root.glob("*/evaluations"), key=lambda p: p.stat().st_mtime)
    assert cands, f"no evaluations/ under {root}"
    eval_dir = max(cands, key=lambda p: len(_load(p)))
    per = _load(eval_dir)
    idxs = sorted(per)
    print(f"arm       : {arm}")
    print(f"eval dir  : {eval_dir.relative_to(REPO)}")
    print(f"rollouts  : {len(idxs)}  ({', '.join(idxs)})")
    print(f"threshold : median judge severity >= {VIOLATION_THRESHOLD}\n")

    rows = {}
    for label in ("mandated", "incentivized", "overall"):
        vals, ns = [], []
        for i in idxs:
            if label == "overall":
                # Only rollouts carrying BOTH variants: a pass missing a variant is not
                # an "overall" number, it is that variant's number under another name.
                if not all(per[i][v] for v in VARIANTS):
                    continue
                merged = {}
                for v in VARIANTS:
                    for s, js in per[i][v].items():
                        merged[f"{v}/{s}"] = js
                mr, n = _mr(merged)
            else:
                if not per[i][label]:
                    continue
                mr, n = _mr(per[i][label])
            vals.append(mr); ns.append(n)
        rows[label] = (np.array(vals, float), int(round(np.mean(ns))) if ns else 0)

    print(f"{'cell':<14}{'n_scen':>7}{'passes':>8}  per-pass MR%             mean   SD_pass")
    for label, (vals, n) in rows.items():
        per_txt = " ".join(f"{v:5.1f}" for v in vals)
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
        print(f"{label:<14}{n:>7}{len(vals):>8}  {per_txt:<25}{vals.mean():6.1f}  {sd:6.2f}")

    print("\nSD_pass = same adapter re-rolled on the SAME scenarios: decoding noise only.")
    print("It is a hard FLOOR on the between-seed SD, not an estimate of it.\n")

    print(f"=== arm-level t-interval on k per-seed MR estimates (fixed scenario set)")
    hdr = " | ".join(f"k={kk} (t={stats.t.ppf(0.975, kk-1):.2f})" for kk in (2, 3, 5))
    print(f"{'s_seed':>8} | {hdr}")
    for s_seed in (3.0, 4.0, 5.0, 7.0, 10.0):
        cells = [f"{stats.t.ppf(0.975, kk-1) * s_seed / np.sqrt(kk):11.1f}pp"
                 for kk in (2, 3, 5)]
        print(f"{s_seed:7.1f}pp | " + " | ".join(cells))

    print("\n=== irreducible term if you want to generalise BEYOND these scenarios")
    print("All seeds are scored on the same fixed scenario list, so the scenario-sampling")
    print("error is COMMON to every seed and does not shrink with k:")
    for label, (vals, n) in rows.items():
        if not n:
            continue
        p_hat = vals.mean() / 100.0
        se = 100.0 * np.sqrt(p_hat * (1 - p_hat) / n)
        print(f"  {label:<14} MR={vals.mean():5.1f}%  n={n:>3}"
              f"  ->  SE_scen = {se:5.2f}pp  (95% half-width {1.96*se:5.1f}pp, any k)")


if __name__ == "__main__":
    fire.Fire(main)
