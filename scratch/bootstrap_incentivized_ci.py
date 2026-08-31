# ABOUTME: Bootstrap 95% CIs for the INCENTIVIZED condition alone, per arm, since the
# ABOUTME: harness only stores a paired interval across both conditions.

"""What is the confidence interval on the incentivized rate by itself?

`results.json` reports one CI, computed by resampling scenarios paired across the mandated and
incentivized variants. That is the right interval for the overall rate and the wrong one for
either condition on its own. This resamples the incentivized SCENARIOS with replacement —
scenarios, not rollouts, because several rollouts of one cell are not independent draws — and
reports the percentile interval on misalignment rate and mean severity.

Run:
  uv run python scratch/bootstrap_incentivized_ci.py
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.misalignment.odcv.odcv import VIOLATION_THRESHOLD  # noqa: E402

ROOT = Path("output/odcv_bench")
ARMS = [
    ("no difficult advice", "qwen3_6-27b-lora-9284-numina-control-716-r64/combined4x_20260819_140227"),
    ("+ constitution", "qwen3_6-27b-lora-9284-numina-control-716-r64-constitution/combined3x_20260820_114430"),
    ("da716 (colleague)", "qwen3_6-27b-lora-t2-9284-da716-r64-dynbatch/combined4x_20260814_230249"),
    ("MATCHED CONTROL", "qwen3_6-27b-lora-t2-9284-synthdoc-716-dynbatch-r64/combined2x_20260820_223257"),
    ("low-ODCV selection", "qwen3_6-27b-lora-t2-9284-synthdoc-716-lowodcv-r64/combined3x_20260818_194628"),
    ("traits 1/3 deleted", "qwen3_6-27b-lora-t2-9284-synthdoc-556-traits13-r64/combined3x_20260819_145013"),
    ("traits 5/6/7 only", "qwen3_6-27b-lora-9284-traits567-716-r64/combined3x_20260819_184957"),
    ("low stakes", "qwen3_6-27b-lora-9284-low-stakes-712-r64/combined2x_20260820_164932"),
    ("no clearance move", "qwen3_6-27b-lora-9284-no-clearance-716-r64/combined3x_20260819_202112"),
    ("traits 1/3/4 removed", "qwen3_6-27b-lora-t2-9284-synthdoc-716-traits134-r64/combined3x_20260819_092850"),
    ("advocacy", "qwen3_6-27b-lora-t2-9284-synthdoc-716-advocacy-r64/combined3x_20260818_141118"),
    ("c6masked", "qwen3_6-27b-lora-t2-9284-synthdoc-716-c6masked-r64/combined4x_20260817_085635"),
    ("traits 1/2/3 only", "qwen3_6-27b-lora-9284-traits123-716-r64/combined3x_20260819_221522"),
    ("no meta-cognition", "qwen3_6-27b-lora-t2-9284-synthdoc-716-c6excised-r64/combined3x_20260818_153146"),
    # Two arms trained on DIFFERENT document types rather than on modified difficult advice:
    # peer critique (critique of another model's reply) and courtroom.
    ("peer critique", "qwen3_6-27b-lora-t2-9284-peercritique716-r64-dynbatch/combined3x_20260816_184649"),
    ("courtroom", "qwen3_6-27b-lora-t2-9284-courtroom716-r64-dynbatch/combined5x_20260816_171141"),
    # The rule-form / ablation line. THE LAST THREE DO NOT ACT: read them with
    # scratch/odcv_tool_call_rate.py beside them, never on their MR alone.
    ("ruleform reasoning", "qwen3_6-27b-lora-t2-9284-synthdoc-716-ruleform-r64/combined2x_20260821_224906"),
    ("ruleform both halves", "qwen3_6-27b-lora-t2-9284-synthdoc-716-ruleform-both-r64/combined2x_20260822_120119"),
    ("synthdoc 1000", "qwen3_6-27b-lora-t2-9000-synthdoc-1000-r64-dynbatch/combined4x_20260812_100843"),
    ("ablated2 (INERT)", "qwen3_6-27b-lora-t2-9284-synthdoc-676-ablated2-r64/combined2x_20260822_220935"),
    ("alt restored (INERT)", "qwen3_6-27b-lora-t2-9284-synthdoc-676-altrestored-r64/combined2x_20260823_151103"),
    ("base (untrained)", "qwen3.6-27b-base-fp8/20260729_105106"),
]


def cells(rel: str, variant: str) -> dict[str, list[float]]:
    """Median-across-judges severity per rollout, grouped by scenario, for one variant.

    Args:
        rel: Path under output/odcv_bench to a combined run dir.
        variant: "incentivized" or "mandated".

    Returns:
        {scenario: [median severity per rollout]}
    """
    per: dict[str, list[float]] = {}
    for f in (ROOT / rel / "evaluations").glob("scores_*.json"):
        for k, v in json.loads(f.read_text()).items():
            s = v.get("score") if isinstance(v, dict) else v
            if isinstance(s, (int, float)):
                per.setdefault(k, []).append(s)
    n = max((len(v) for v in per.values()), default=0)
    out: dict[str, list[float]] = {}
    for k, v in per.items():
        if len(v) != n or not k.startswith(variant):
            continue
        out.setdefault(k.rsplit("/rollout_", 1)[0], []).append(statistics.median(v))
    return out


def boot(by_scen: dict[str, list[float]], draws: int, seed: int) -> tuple:
    """Percentile CIs for MR and mean severity, resampling scenarios with replacement.

    Args:
        by_scen: {scenario: [severities]}.
        draws: Bootstrap replicates.
        seed: RNG seed.

    Returns:
        (mr, mr_lo, mr_hi, sev, sev_lo, sev_hi, n_scen, n_rollouts)
    """
    names = sorted(by_scen)
    flat = [v for s in names for v in by_scen[s]]
    mr = 100.0 * sum(v >= VIOLATION_THRESHOLD for v in flat) / len(flat)
    sev = statistics.mean(flat)
    rng = random.Random(seed)
    mrs, sevs = [], []
    for _ in range(draws):
        pick = [by_scen[names[rng.randrange(len(names))]] for _ in names]
        vals = [v for grp in pick for v in grp]
        mrs.append(100.0 * sum(v >= VIOLATION_THRESHOLD for v in vals) / len(vals))
        sevs.append(statistics.mean(vals))
    mrs.sort(); sevs.sort()
    lo, hi = int(0.025 * draws), int(0.975 * draws)
    return mr, mrs[lo], mrs[hi], sev, sevs[lo], sevs[hi], len(names), len(flat)


def main(draws: int = 10000, seed: int = 0, variant: str = "incentivized") -> None:
    """Print per-arm CIs for one condition alone.

    Args:
        draws: Bootstrap replicates.
        seed: RNG seed.
        variant: Which condition to bootstrap.
    """
    rows = []
    for label, rel in ARMS:
        by = cells(rel, variant)
        if not by:
            print(f"{label}: no {variant} cells")
            continue
        rows.append((label, *boot(by, draws, seed)))
    rows.sort(key=lambda r: -r[1])
    print(f"{variant.upper()} ONLY — {draws:,} bootstrap replicates over scenarios\n")
    print(f"{'arm':24s}{'MR':>7}{'CI95':>16}{'width':>7}{'sev':>6}{'sev CI95':>15}"
          f"{'cells':>7}{'n':>6}")
    for lab, mr, lo, hi, sev, slo, shi, nc, nr in rows:
        print(f"{lab:24s}{mr:6.1f}%  [{lo:4.1f},{hi:5.1f}]{hi-lo:7.1f}{sev:6.2f}"
              f"  [{slo:.2f},{shi:.2f}]{nc:7d}{nr:6d}")
    ctl = next((r for r in rows if r[0] == "MATCHED CONTROL"), None)
    if ctl:
        print(f"\nMATCHED CONTROL {variant} CI [{ctl[2]:.1f}, {ctl[3]:.1f}] — arms not overlapping it:")
        none = True
        for lab, mr, lo, hi, *_ in rows:
            if lab == "MATCHED CONTROL":
                continue
            if hi < ctl[2] or lo > ctl[3]:
                print(f"   {lab}: [{lo:.1f}, {hi:.1f}]")
                none = False
        if none:
            print("   (none — every arm's incentivized interval overlaps the control's)")


if __name__ == "__main__":
    fire.Fire(main)
