# ABOUTME: Audits the seven-arm figure's seed means against the two rules the canonical
# ABOUTME: scratch/stats/odcv_seed_sem.py enforces: ONE PASS PER SEED and CELL INTERSECTION.
# Run: uv run python scratch/gpt_seeds/seed_pass_audit.py
#
# scratch/stats/odcv_seed_sem.py (the team's seed-variance reference) states two rules that
# plot_seed_mean.py does NOT currently apply, because it feeds each seed's COMBINED
# multi-pass results.json straight into summarise():
#
#   1. ONE PASS PER SEED. A seed evaluated over several passes gets a less noisy point
#      estimate than a one-pass sibling, which breaks the equal-variance assumption the
#      +-1.96*SEM / t interval over seeds rests on.
#   2. INTERSECTION OF CELLS. A cell missing from any seed is dropped from EVERY seed --
#      dropout is not random w.r.t. MR (the more elaborate seed overruns context and the
#      proxy timeout, losing precisely its longest, most agentic rollouts).
#
# That script reads LOCAL output/odcv_bench/<arm>/*/evaluations/scores_*.json, which this
# machine does not have. The published per-scenario medians are enough, though: a
# multi-pass run keys cells "<Scenario>/rollout_NNN", so a single pass IS recoverable.
# This script reports, per arm, the per-pass MR spread and what each rule does to the
# headline -- so the pass choice is a visible decision rather than a silent default.

import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import fire
import numpy as np
from omegaconf import OmegaConf
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval.misalignment.odcv.odcv import (  # noqa: E402
    median_score,
    misalignment_rate,
    scenario_violation_rate,
)

_spec = importlib.util.spec_from_file_location(
    "p", Path(__file__).with_name("plot_seed_mean.py")
)
p = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p)

EXCL = set(
    OmegaConf.to_container(
        OmegaConf.load(ROOT / p.CFG).get("exclude_scenarios", []) or []
    )
)


def passes(psm: dict) -> dict[str, dict[str, list[float]]]:
    """{pass_id: {'variant/scenario': [scores]}} from a published per_scenario_medians.

    Handles all three shapes in play: "<Scenario>/rollout_NNN" -> float (multi-pass),
    "<Scenario>" -> [s, s] (nested rollouts), "<Scenario>" -> float (single pass).
    """
    out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for variant, cells in psm.items():
        for key, value in cells.items():
            scenario, _, idx = key.partition("/")
            if f"{variant}/{scenario}" in EXCL:
                continue
            cell = f"{variant}/{scenario}"
            if isinstance(value, (list, tuple)):
                for i, s in enumerate(value):
                    out[f"rollout_{i:03d}"][cell].append(float(s))
            else:
                out[idx or "rollout_000"][cell].append(float(value))
    return {k: dict(v) for k, v in sorted(out.items())}


def mr_on(cells: dict[str, list[float]], keys, estimator: str = "rate") -> float:
    """MR over `keys`. Two estimators live in the repo and they are NOT the same number.

    "rate" is what src/eval/misalignment/odcv/odcv.py `summarise` uses today and what the figure is built on:
    a cell contributes the FRACTION of its rollouts that violated, so 1 of 2 is 0.5. Its
    docstring is explicit that collapsing a cell to a verdict throws away the only thing
    repeated rollouts add.

    "median" is the older binary form -- median rollout >= 3 makes the whole cell a
    violation -- and is what scratch/stats/odcv_seed_sem.py still uses. Reported here so
    the rules below are not silently confounded with an estimator change.
    """
    if estimator == "median":
        return misalignment_rate([median_score(cells[k]) for k in keys])
    return 100.0 * float(np.mean([scenario_violation_rate(cells[k]) for k in keys]))


def interval(mrs: np.ndarray) -> dict:
    k = len(mrs)
    mean, sd = float(mrs.mean()), float(mrs.std(ddof=1))
    sem = sd / np.sqrt(k)
    t = float(stats.t.ppf(0.975, k - 1))
    return dict(mean=mean, sd=sd, sem=sem, half=1.96 * sem, half_t=t * sem, t=t, k=k)


def main(pass_choice: str = "first") -> None:
    """Audit every multi-seed arm.

    Args:
        pass_choice: which single pass to keep per seed -- "first" (rollout_000) or
            "last". The canonical script defaults seed 0 to rollout_001 for the numina arm
            specifically, because its rollout_000 was an incomplete pass; there is no
            arm-general right answer, so both ends are reported per pass below.
    """
    for key, arm in p.ARMS.items():
        if len(arm["seeds"]) < 2:
            continue
        per_seed_passes, combined = {}, {}
        for sd, src in sorted(arm["seeds"].items()):
            psm = p._load(src)["per_scenario_medians"]
            per_seed_passes[sd] = passes(psm)
            merged: dict[str, list[float]] = defaultdict(list)
            for pcells in per_seed_passes[sd].values():
                for c, s in pcells.items():
                    merged[c].extend(s)
            combined[sd] = dict(merged)

        print(f"\n=== {key}  ({arm['long']})")
        for sd, ps in per_seed_passes.items():
            detail = "  ".join(
                f"{pid.replace('rollout_', 'p')}:{len(c)}c/{mr_on(c, c):.1f}%"
                for pid, c in ps.items()
            )
            print(f"  seed {sd:>2}: {len(ps)} pass(es)  {detail}")

        # --- as plotted: every pass pooled, each seed on its own cells ---------------
        as_plotted = np.array([mr_on(c, c) for c in combined.values()], float)
        # --- rule 1 only: one pass per seed, still each on its own cells -------------
        pick = {
            sd: ps[(min if pass_choice == "first" else max)(ps)]
            for sd, ps in per_seed_passes.items()
        }
        rule1 = np.array([mr_on(c, c) for c in pick.values()], float)
        # --- rules 1+2: one pass per seed, on the cells every seed kept -------------
        shared = sorted(set.intersection(*(set(c) for c in pick.values())))
        rule12 = np.array([mr_on(c, shared) for c in pick.values()], float)
        # --- rule 2 only: all passes pooled, on the shared cells --------------------
        shared_all = sorted(set.intersection(*(set(c) for c in combined.values())))
        rule2 = np.array([mr_on(c, shared_all) for c in combined.values()], float)

        for label, mrs, n in (
            ("as plotted (all passes, own cells)", as_plotted, None),
            ("rule 2 only  (all passes, shared) ", rule2, len(shared_all)),
            ("rule 1 only  (1 pass,   own cells)", rule1, None),
            ("rules 1+2    (1 pass,   shared)   ", rule12, len(shared)),
        ):
            i = interval(mrs)
            cells = f" on {n} cells" if n else ""
            print(
                f"    {label}: seeds "
                + ", ".join(f"{m:.1f}" for m in mrs)
                + f"  ->  {i['mean']:.1f} ±{i['half']:.1f} "
                f"(SD {i['sd']:.2f}; t-95% ±{i['half_t']:.1f}){cells}"
            )
        old = np.array([mr_on(c, c, "median") for c in combined.values()], float)
        print(
            f"    [older median-verdict estimator, same data: "
            + ", ".join(f"{m:.1f}" for m in old)
            + f"  ->  {interval(old)['mean']:.1f}]"
        )


if __name__ == "__main__":
    fire.Fire(main)
