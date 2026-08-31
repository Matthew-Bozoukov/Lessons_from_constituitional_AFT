# ABOUTME: Grok arm at R=3: splits the observed seed spread into real between-seed variance
# ABOUTME: and rollout measurement noise, using the two extra passes run per seed.
"""Does the grok arm have a real training-seed effect, or was it rollout noise?

    uv run python scratch/stats/grok_r3_decomp.py

At R=1 each seed's MR carried measurement error sigma_eps^2/(J*R) that is indistinguishable
from a seed difference. With three passes per seed the two separate:

    between-pass variance WITHIN a seed  ->  sigma_eps^2  (pure rollout noise)
    variance of the per-seed MEANS       ->  s_seed_true^2 + sigma_eps^2/R

so s_seed_true^2 = var(seed means) - sigma_eps^2/R, floored at zero.
"""

from __future__ import annotations

import glob
import json
import os
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
    VARIANTS, VIOLATION_THRESHOLD, median_score,
)

SP = ("/tmp/claude-1000/-home-matthewb-git-repos-teaching-claude-why-replication/"
      "68da2770-3921-48bd-8ea1-50b1272ad459/scratchpad/grokR3")
ORIG = {  # the pass each seed already had before this round
    0: "output/odcv_bench/qwen3_6-27b-lora-t2-9284-grokresp703-paired-r64",
    42: "output/odcv_bench/qwen3_6-27b-lora-t2-9284-grokresp703-paired-r64-seed42",
    69: "output/odcv_bench/qwen3_6-27b-lora-t2-9284-grokresp703-paired-r64-seed69",
}
KEY = re.compile(r"^(?P<v>[^/]+)/(?P<s>[^/]+?)(?:/rollout_(?P<i>\d+))?$")


def _cells(eval_dir: str, want: str | None = None) -> dict[str, float]:
    """{'variant/scenario': violation 0/1} from one pass's judge scores."""
    per: dict[str, list[float]] = defaultdict(list)
    for f in sorted(glob.glob(eval_dir + "/scores_*.json")):
        for k, v in json.load(open(f)).items():
            m = KEY.match(k)
            if not m or m["v"] not in VARIANTS:
                continue
            if want is not None and m["i"] not in (None, want):
                continue
            sc = v.get("score") if isinstance(v, dict) else v
            try:
                per[f'{m["v"]}/{m["s"]}'].append(float(sc))
            except (TypeError, ValueError):
                pass
    return {k: float(median_score(v) >= VIOLATION_THRESHOLD) for k, v in per.items() if v}


def main(variant: str = "incentivized") -> None:
    """Decompose the grok arm's seed spread at R=3."""
    passes: dict[int, list[dict]] = defaultdict(list)
    for seed, tag in ((0, "seed0"), (42, "seed42"), (69, "seed69")):
        for p in ("p1", "p2"):
            d = glob.glob(f"{SP}/out_{tag}_{p}/*/2026*/evaluations")
            if d:
                passes[seed].append(_cells(d[0]))
        cands = [q for q in glob.glob(str(REPO / ORIG[seed] / "*/evaluations"))
                 if glob.glob(q + "/scores_*.json")]
        if cands:
            passes[seed].append(_cells(max(cands, key=os.path.getmtime),
                                       want="000" if seed == 0 else None))

    sel = lambda d: {k: v for k, v in d.items() if k.startswith(variant + "/")}
    passes = {s: [sel(p) for p in ps] for s, ps in passes.items()}
    shared = sorted(set.intersection(*(set(p) for ps in passes.values() for p in ps)))
    print(f"grok arm, {variant}: {len(shared)} scenarios shared by all "
          f"{sum(len(v) for v in passes.values())} passes\n")

    mr = {s: [100.0 * float(np.mean([p[c] for c in shared])) for p in ps]
          for s, ps in passes.items()}
    R = min(len(v) for v in mr.values())
    within = []
    for s, v in mr.items():
        v = np.array(v[:R], float)
        print(f"  seed {s:<2}: passes {['%.1f%%' % x for x in v]}  mean {v.mean():6.2f}%"
              f"  within-seed SD {v.std(ddof=1):5.2f}pp")
        within.append(v.var(ddof=1))
    means = np.array([np.mean(v[:R]) for v in mr.values()], float)
    sig_eps = float(np.mean(within))           # pooled rollout variance of a pass MR
    obs = float(means.var(ddof=1))
    true = max(obs - sig_eps / R, 0.0)
    print(f"\n  R = {R} passes per seed")
    print(f"  pooled WITHIN-seed variance (rollout noise on one pass MR) = {sig_eps:.3f} pp^2"
          f"   -> SD {np.sqrt(sig_eps):.2f}pp")
    print(f"  variance of the three seed MEANS                          = {obs:.3f} pp^2"
          f"   -> SD {np.sqrt(obs):.2f}pp")
    print(f"  expected from noise alone at R={R}: {sig_eps/R:.3f} pp^2"
          f"   -> SD {np.sqrt(sig_eps/R):.2f}pp")
    print(f"\n  => s_seed_true = {np.sqrt(true):.2f}pp"
          + ("   (indistinguishable from zero)" if true <= 0 else ""))
    sem = np.sqrt(obs / 3)
    print(f"\n  mean MR over seeds = {means.mean():.2f}%")
    print(f"  +/-1.96 SEM (seed) = +/-{1.96*sem:.2f}pp  "
          f"[{means.mean()-1.96*sem:.2f}, {means.mean()+1.96*sem:.2f}]")
    t = stats.t.ppf(0.975, 2)
    print(f"  +/-t   95%         = +/-{t*sem:.2f}pp  "
          f"[{means.mean()-t*sem:.2f}, {means.mean()+t*sem:.2f}]")


if __name__ == "__main__":
    fire.Fire(main)
