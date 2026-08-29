# ABOUTME: Re-aggregate agentic_misalignment sweep results from the saved Inspect .eval logs
# ABOUTME: (source of truth) across the three numina seeds, with the across-seed +-1.96 SEM.

"""Recompute the 27-condition sweep per seed from the .eval logs and the k=3 seed interval.

The inline summary written during the run used a buggy harmful-parser (read the scorer's dict
value as a scalar -> spurious 0%). The .eval logs hold the correct per-sample scores, so this
recomputes each seed's aggregate from them and reports the across-seed mean +- 1.96 SEM (the
same k=3 statistic used for ODCV), with the caveat that at k=3 +-1.96 SEM is ~81% coverage,
not 95% (t(2)=4.30).
"""

from __future__ import annotations

import glob
import math

import fire
from inspect_ai.log import read_eval_log


def _harmful(sample) -> bool:
    for sc in (sample.scores or {}).values():
        v = sc.value
        if isinstance(v, dict):
            h = v.get("harmful", v.get("classifier_verdict"))
            if h is not None:
                return float(h) > 0
        if isinstance(v, (int, float)):
            return v > 0
    return False


def seed_rate(log_dir: str) -> dict:
    """Aggregate one seed's 27 conditions from its .eval logs."""
    conds, harmful, byscen = 0, 0, {}
    conflict = [0, 0]
    for fn in sorted(glob.glob(f"{log_dir}/*.eval")):
        log = read_eval_log(fn)
        if not log.samples:
            continue
        ta = log.eval.task_args
        sc = ta["scenario"]
        h = any(_harmful(s) for s in log.samples)
        conds += 1
        harmful += int(h)
        byscen.setdefault(sc, [0, 0])
        byscen[sc][0] += int(h)
        byscen[sc][1] += 1
        is_conf = ta["goal_type"] != "none"
        conflict[0] += int(h and is_conf)
        conflict[1] += int(is_conf)
    return {"n_conditions": conds, "n_harmful": harmful,
            "rate_pct": round(100.0 * harmful / conds, 2) if conds else None,
            "by_scenario": {k: round(100.0 * v[0] / v[1], 1) for k, v in byscen.items()}}


def main(base: str = "logs/agentic_numina") -> None:
    """Print per-seed rates and the across-seed +-1.96 SEM."""
    seeds = {"s0": f"{base}_s0", "s42": f"{base}_s42", "s69": f"{base}_s69"}
    rows = {k: seed_rate(v) for k, v in seeds.items()}
    print(f"{'seed':<6} {'harmful/27':>11} {'rate':>7}   by-scenario")
    print("-" * 60)
    rates = []
    for k, r in rows.items():
        if r["rate_pct"] is None:
            print(f"{k:<6} (no logs yet)")
            continue
        rates.append(r["rate_pct"])
        print(f"{k:<6} {r['n_harmful']:>2}/{r['n_conditions']:<8} {r['rate_pct']:>6}%   "
              f"{r['by_scenario']}")
    if len(rates) >= 2:
        mean = sum(rates) / len(rates)
        sd = math.sqrt(sum((x - mean) ** 2 for x in rates) / (len(rates) - 1))
        sem = sd / math.sqrt(len(rates))
        print()
        print(f"across {len(rates)} seeds: mean {mean:.2f}%  s_seed {sd:.2f}pp  SEM {sem:.2f}pp")
        print(f"  +-1.96 SEM = {mean:.2f} +- {1.96*sem:.2f}pp  "
              f"[{mean-1.96*sem:.2f}, {mean+1.96*sem:.2f}]   (k={len(rates)}: ~81% coverage)")
        if len(rates) == 3:
            t = 4.3027
            print(f"  +-t   SEM  = {mean:.2f} +- {t*sem:.2f}pp  "
                  f"[{mean-t*sem:.2f}, {mean+t*sem:.2f}]   (t(2), true 95%)")


if __name__ == "__main__":
    fire.Fire(main)
