# ABOUTME: ODCV MR intervals whose sampling unit is the SCENARIO (with its rollouts as
# ABOUTME: repeats), not the training seed: CLT SE across scenarios plus a cluster bootstrap.
"""MR +/-1.96 SE varying scenario and rollout, holding the model fixed.

    uv run python scratch/stats/odcv_scenario_se.py --run <combined dir>

The seed-level interval asks "how much would MR move if I retrained?" — it needs k models
and at k=3 carries a t multiplier of 4.303. This asks a different question: "how much would
MR move if I drew different SCENARIOS and re-rolled them?" — the model is fixed and the
sampling units are the scenarios, with each scenario's rollouts averaged first.

Following Miller (arXiv:2411.00640) sec 2.1: with K answers per question the question score
is the MEAN of the K, and the SE is taken ACROSS questions. Pooling all scenario-rollout
pairs as if independent would treat repeats of one scenario as new evidence and understate
the SE. Because s_i is then fractional, sqrt(p(1-p)/n) does NOT apply -- that is the paper's
named Llama-3 mistake -- so the Bernoulli value is printed only to show the gap.

The cluster bootstrap resamples SCENARIOS with replacement (carrying each one's rollouts
along), which is the same unit of variation without the normal-theory assumption.
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

KEY = re.compile(r"^(?P<variant>[^/]+)/(?P<scenario>[^/]+?)(?:/rollout_(?P<idx>\d+))?$")


def _load(eval_dir: Path) -> dict[str, dict[str, list[float]]]:
    """{'variant/scenario': {rollout: [judge scores]}}."""
    out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(str(eval_dir / "scores_*.json"))):
        for key, val in json.load(open(f)).items():
            m = KEY.match(key)
            if not m or m["variant"] not in VARIANTS:
                continue
            sc = val.get("score") if isinstance(val, dict) else val
            try:
                out[f'{m["variant"]}/{m["scenario"]}'][m["idx"] or "single"].append(float(sc))
            except (TypeError, ValueError):
                pass
    return {k: dict(v) for k, v in out.items()}


def main(run: str, boots: int = 20000, seed: int = 0) -> None:
    """Print scenario-level MR intervals for one run.

    Args:
        run: A combined ODCV run directory (holding evaluations/scores_*.json).
        boots: Cluster-bootstrap resamples.
        seed: RNG seed, so the bootstrap interval is reproducible.
    """
    ev = Path(run) / "evaluations"
    assert ev.is_dir(), f"no evaluations/ under {run}"
    data = _load(ev)
    rng = np.random.default_rng(seed)

    groups = {"overall": sorted(data)}
    for v in VARIANTS:
        groups[v] = sorted(k for k in data if k.startswith(f"{v}/"))

    for label, keys in groups.items():
        if not keys:
            continue
        # s_i = share of THIS scenario's rollouts that violate; rollouts are repeats, not
        # independent observations, so they are averaged before anything else happens.
        si = np.array([
            np.mean([median_score(v) >= VIOLATION_THRESHOLD for v in data[k].values()])
            for k in keys], float)
        n = len(si)
        n_roll = sum(len(data[k]) for k in keys)
        mr = 100 * si.mean()
        se = 100 * si.std(ddof=1) / np.sqrt(n)
        p = si.mean()
        se_bern = 100 * np.sqrt(p * (1 - p) / n)
        idx = rng.integers(0, n, size=(boots, n))
        draws = 100 * si[idx].mean(axis=1)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        frac = int(((si > 0) & (si < 1)).sum())

        print(f"=== {label}   {n} scenarios, {n_roll} rollouts "
              f"({n_roll / n:.2f} per scenario)")
        print(f"    MR                = {mr:.4f}%")
        print(f"    SE across scenarios = {se:.4f}pp   (rollouts averaged within scenario)")
        print(f"    +/-1.96 SE        = +/-{1.96 * se:.4f}pp  ->  "
              f"[{mr - 1.96 * se:.4f}, {mr + 1.96 * se:.4f}]")
        print(f"    cluster bootstrap 95% ({boots} resamples of scenarios) -> "
              f"[{lo:.4f}, {hi:.4f}]")
        print(f"    [naive Bernoulli SE would be {se_bern:.4f}pp -- WRONG here, s_i is "
              f"fractional for {frac} of {n} scenarios]\n")


if __name__ == "__main__":
    fire.Fire(main)
