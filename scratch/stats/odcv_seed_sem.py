# ABOUTME: Three-seed ODCV MR for the numina control arm: MR per seed on the cell set all
# ABOUTME: three share, then s_seed and the +/-1.96 SEM and t-based 95% intervals.
"""s_seed and interval for numina control over training seeds 0, 42 and 69.

    uv run python scratch/stats/odcv_seed_sem.py

Two rules this enforces, both of which change the answer:

1. ONE PASS PER SEED. Seeds 42 and 69 have a single rollout pass each; seed 0 has several.
   Averaging seed 0's passes would give it a less noisy point estimate than the others,
   breaking the equal-variance assumption the t-interval rests on, so seed 0 is taken at
   rollout_000. Its multi-pass mean is printed alongside for reference only.

2. INTERSECTION OF CELLS. A cell that produced no transcript in some seed is dropped from
   EVERY seed. The dropout is not random with respect to MR -- the seed that behaves more
   elaborately overruns context and the proxy timeout more often, and loses precisely its
   longest, most agentic rollouts -- so comparing seeds on their own cell sets is biased
   against the more active seed.
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

from src.utils import git_sha, write_run_meta

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.eval.misalignment.odcv.odcv import (  # noqa: E402
    VARIANTS,
    VIOLATION_THRESHOLD,
    median_score,
    misalignment_rate,
)

# Single-pass runs key cells "variant/scenario"; a combined multi-pass run appends
# "/rollout_NNN". Accept both, or the new arms silently parse to zero cells.
KEY = re.compile(r"^(?P<variant>[^/]+)/(?P<scenario>[^/]+?)(?:/rollout_(?P<idx>\d+))?$")
ARM_SETS = {
    "numina": {
        0: "qwen3_6-27b-lora-9284-numina-control-716-r64",
        42: "qwen3_6-27b-lora-9284-numina-control-716-r64-seed42",
        69: "qwen3_6-27b-lora-9284-numina-control-716-r64-seed69",
    },
    "synthdoc": {
        0: "qwen3_6-27b-lora-t2-9284-synthdoc-716-dynbatch-r64",
        42: "qwen3_6-27b-lora-t2-9284-synthdoc-716-dynbatch-r64-seed42",
        69: "qwen3_6-27b-lora-t2-9284-synthdoc-716-dynbatch-r64-seed69",
    },
    "grokresp": {
        0: "qwen3_6-27b-lora-t2-9284-grokresp703-paired-r64",
        42: "qwen3_6-27b-lora-t2-9284-grokresp703-paired-r64-seed42",
        69: "qwen3_6-27b-lora-t2-9284-grokresp703-paired-r64-seed69",
    },
    # GPT-responder paired arm (2026-08-28 replicates). Every seed here is a 2-pass combined
    # run, so `seed0_pass` picks the pass for ALL of them (one pass per seed, as above).
    "gptresp": {
        0: "qwen3_6-27b-lora-t2-9284-gptresp685-paired-r64",
        42: "qwen3_6-27b-lora-t2-9284-gptresp685-paired-r64-seed42",
        69: "qwen3_6-27b-lora-t2-9284-gptresp685-paired-r64-seed69",
    },
}
ARMS = ARM_SETS["numina"]


def _scores(eval_dir: Path, want_idx: str | None) -> dict[str, list[float]]:
    """{'variant/scenario': [judge scores]} for one rollout index (or the only one)."""
    out: dict[str, list[float]] = defaultdict(list)
    for f in sorted(glob.glob(str(eval_dir / "scores_*.json"))):
        for key, val in json.load(open(f)).items():
            m = KEY.match(key)
            if not m or m["variant"] not in VARIANTS:
                continue
            if want_idx is not None and m["idx"] is not None and m["idx"] != want_idx:
                continue
            if want_idx is None and m["idx"] is not None:
                continue  # multi-pass file but no pass chosen -- ambiguous, skip
            s = val.get("score") if isinstance(val, dict) else val
            if s is not None:
                out[f"{m['variant']}/{m['scenario']}"].append(float(s))
    return dict(out)


def _scores_all(eval_dir: Path) -> dict[str, list[float]]:
    """Every judge score per cell, pooled over all passes -- reference only."""
    out: dict[str, list[float]] = defaultdict(list)
    for f in sorted(glob.glob(str(eval_dir / "scores_*.json"))):
        for key, val in json.load(open(f)).items():
            m = KEY.match(key)
            if not m or m["variant"] not in VARIANTS:
                continue
            sc = val.get("score") if isinstance(val, dict) else val
            if sc is not None:
                out[f"{m['variant']}/{m['scenario']}"].append(float(sc))
    return dict(out)


def _latest_eval(arm: str) -> Path:
    """Newest evaluations/ dir holding judge scores for one arm."""
    cands = [
        p
        for p in (REPO / "output/odcv_bench" / arm).glob("*/evaluations")
        if glob.glob(str(p / "scores_*.json"))
    ]
    assert cands, f"no judged evaluations/ under {arm}"
    return max(cands, key=lambda p: p.stat().st_mtime)


def main(
    seed0_pass: str = "001", out: str = "output/odcv_seed_variance", arm: str = "numina"
) -> None:
    global ARMS
    ARMS = ARM_SETS[arm]
    # fire coerces "000" to the int 0, which then never equals the string rollout index and
    # silently yields zero cells. Normalise back to the zero-padded string form.
    seed0_pass = str(seed0_pass).zfill(3)
    # rollout_000 is seed 0's INCOMPLETE first pass (23 of 35 mandated cells);
    # rollout_001 and _002 are its full passes. Defaulting to a full pass keeps the
    # per-seed cell counts comparable before the intersection is taken.
    """Print per-seed MR on shared cells, s_seed, and both intervals."""
    per_seed, srcs = {}, {}
    for seed, arm in ARMS.items():
        d = _latest_eval(arm)
        srcs[seed] = d.relative_to(REPO)
        # The chosen pass applies to EVERY seed that has passes; a single-pass seed keeps its
        # only pass (the filter is a no-op on "variant/scenario" keys). Originally seed 0 was
        # the only multi-pass seed; the gptresp arm's replicates are 2-pass runs too.
        per_seed[seed] = _scores(d, seed0_pass)
        print(f"seed {seed:>2}: {len(per_seed[seed]):>3} judged cells   {srcs[seed]}")

    shared = sorted(set.intersection(*(set(v) for v in per_seed.values())))
    print(f"\nshared cells across all 3 seeds: {len(shared)}")
    for seed in ARMS:
        missing = len(per_seed[seed]) - len(shared)
        if missing:
            print(f"  seed {seed}: {missing} cell(s) dropped to reach the intersection")

    def mr_for(seed: int, keys: list[str]) -> float:
        return misalignment_rate([median_score(per_seed[seed][k]) for k in keys])

    groups = {"overall": shared}
    for v in VARIANTS:
        groups[v] = [k for k in shared if k.startswith(f"{v}/")]

    rows: dict = {}
    print()
    for label, keys in groups.items():
        if not keys:
            continue
        mrs = np.array([mr_for(s, keys) for s in ARMS], float)
        mean = mrs.mean()
        s_seed = mrs.std(ddof=1)
        k = len(mrs)
        sem = s_seed / np.sqrt(k)
        t = stats.t.ppf(0.975, k - 1)
        cov = 100 * (2 * stats.t.cdf(1.96, k - 1) - 1)
        print(f"=== {label}   n={len(keys)} cells, k={k} seeds")
        print(
            "    MR per seed : "
            + "  ".join(f"seed{s}={m:.1f}%" for s, m in zip(ARMS, mrs))
        )
        print(f"    mean MR     = {mean:.2f}%")
        print(f"    s_seed      = {s_seed:.2f}pp   (MEASURED, not assumed)")
        print(f"    SEM         = {sem:.2f}pp")
        print(
            f"    +/-1.96 SEM = {mean:.1f} +/- {1.96 * sem:.1f}pp  "
            f"[{mean - 1.96 * sem:.1f}, {mean + 1.96 * sem:.1f}]  <-- actual coverage {cov:.0f}%"
        )
        print(
            f"    +/-t   SEM  = {mean:.1f} +/- {t * sem:.1f}pp  "
            f"[{mean - t * sem:.1f}, {mean + t * sem:.1f}]  (t={t:.2f}, df={k - 1}, true 95%)\n"
        )
        rows[label] = {
            "n_cells": len(keys),
            "k_seeds": k,
            "mr_by_seed": {str(sd): round(float(m), 4) for sd, m in zip(ARMS, mrs)},
            "mean_mr_pct": round(float(mean), 4),
            "s_seed_pp": round(float(s_seed), 4),
            "sem_pp": round(float(sem), 4),
            "half_width_1p96_pp": round(float(1.96 * sem), 4),
            "ci_1p96": [
                round(float(mean - 1.96 * sem), 4),
                round(float(mean + 1.96 * sem), 4),
            ],
            "actual_coverage_of_1p96_pct": round(float(cov), 2),
            "t_multiplier": round(float(t), 4),
            "half_width_t_pp": round(float(t * sem), 4),
            "ci_t95": [
                round(float(mean - t * sem), 4),
                round(float(mean + t * sem), 4),
            ],
        }

    d0 = _latest_eval(ARMS[0])
    allp = {k: v for k, v in _scores_all(d0).items() if k in shared}
    if allp and shared:
        ref = misalignment_rate([median_score(v) for v in allp.values()])
        print(
            f"reference: seed 0 pooled over ALL its passes on the same {len(shared)} "
            f"cells = {ref:.1f}%  (not used above -- see module docstring)"
        )

    stamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = REPO / out / stamp
    dest.mkdir(parents=True, exist_ok=True)
    payload = {
        "metric": "ODCV-Bench misalignment rate (median judge severity >= 3.0)",
        "arm": arm,
        "seeds": list(ARMS),
        "seed0_pass": seed0_pass,
        "one_pass_per_seed": True,
        "shared_cells": len(shared),
        "cells": shared,
        "results": rows,
        "sources": {str(k): str(v) for k, v in srcs.items()},
        "git_sha": git_sha(),
        "timestamp": stamp,
        "caveats": [
            "PRE-RE-RUN. Cells that failed on the 16,384-token context limit or the 120s "
            "Cloudflare proxy timeout are excluded. Those failures were disproportionately "
            "long, agentic rollouts and fell more often on the seed that behaved more "
            "elaborately, so MR here is likely biased LOW.",
            "+/-1.96 SEM is an ~81% interval at df=2, NOT 95%. The true 95% multiplier for "
            "k=3 is t=4.303; both are reported.",
            "One pass per seed: seeds 42/69 have a single pass, so seed 0 is taken at one "
            "pass too rather than its multi-pass mean, to keep the variances comparable.",
        ],
    }
    (dest / "results.json").write_text(json.dumps(payload, indent=2))
    write_run_meta(
        dest,
        {
            "script": "scratch/stats/odcv_seed_sem.py",
            "seed0_pass": seed0_pass,
            "arms": ARMS,
            "sources": {str(k): str(v) for k, v in srcs.items()},
        },
    )

    md = [
        f"# ODCV MR across training seeds — numina control ({stamp})",
        "",
        f"`s_seed` measured over seeds {list(ARMS)}, one rollout pass each, "
        f"on the {len(shared)} cells all three share.",
        "",
        "| cell set | n | "
        + " | ".join(f"seed {s}" for s in ARMS)
        + " | mean MR | s_seed | SEM | **±1.96 SEM** | ±t (true 95%) |",
        "|---|---|" + "---|" * (len(ARMS) + 5),
    ]
    for label, r in rows.items():
        md.append(
            f"| {label} | {r['n_cells']} | "
            + " | ".join(f"{r['mr_by_seed'][str(s)]:.1f}%" for s in ARMS)
            + f" | {r['mean_mr_pct']:.2f}% | {r['s_seed_pp']:.2f}pp | {r['sem_pp']:.2f}pp "
            f"| **±{r['half_width_1p96_pp']:.1f}pp** "
            f"[{r['ci_1p96'][0]:.1f}, {r['ci_1p96'][1]:.1f}] "
            f"| ±{r['half_width_t_pp']:.1f}pp "
            f"[{r['ci_t95'][0]:.1f}, {r['ci_t95'][1]:.1f}] |"
        )
    md += ["", "## Caveats", ""] + [f"- {c}" for c in payload["caveats"]]
    md += ["", "## Sources", ""] + [f"- seed {k}: `{v}`" for k, v in srcs.items()]
    md += [
        "",
        f"git `{payload['git_sha']}` · regenerate: "
        "`uv run python scratch/stats/odcv_seed_sem.py`",
        "",
    ]
    (dest / "results.md").write_text("\n".join(md))
    print(f">>> saved {dest.relative_to(REPO)}/results.json + results.md")


if __name__ == "__main__":
    fire.Fire(main)
