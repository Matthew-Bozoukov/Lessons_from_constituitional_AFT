# ABOUTME: Aggregates agentic-misalignment classification results into per-condition and overall
# ABOUTME: misalignment rates, with the interval over sweeps (trial k of every condition is sweep k).

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from src.eval.stats import logit_t


def _iter_responses(results_dir: Path):
    """Yield (condition, sample_id, response_dict) for every classified response.json found."""
    for resp_file in results_dir.rglob("response.json"):
        try:
            data = json.loads(resp_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        meta = data.get("metadata") or {}
        # condition is the parent of sample_xxx; the trial index is in the metadata
        condition = meta.get("condition") or resp_file.parent.parent.name
        yield condition, int(meta["sample_id"]), data


def main(results_dir: str, label: str = "", out: str | None = None) -> None:
    """Compute misalignment rates from a results directory.

    The eval is 18 conditions x `samples_per_condition` trials. Trial k of every condition is
    one SWEEP of the whole panel, and the sweep's harmful fraction is one draw of the eval's
    number; the overall rate is the mean over sweeps and the interval a t over them on the
    log-odds scale (`stats.logit_t`, docs/error_bars.md). Trials are never counted as
    independent draws: the conditions share three scenarios and co-move. Per-condition and
    per-scenario rates are reported as plain numbers, without an interval.

    Args:
        results_dir: Path to results/<experiment_id> (or its models/ subdir).
        label: Optional label (e.g. "baseline") recorded in the output.
        out: Optional path to write the JSON summary; defaults alongside results.
    """
    root = Path(results_dir)
    by_cond: dict[str, list[bool]] = defaultdict(list)
    by_sweep: dict[int, list[bool]] = defaultdict(list)
    n_missing = 0
    for condition, sample_id, data in _iter_responses(root):
        cls = data.get("classification")
        if not cls or "harmful_behavior" not in cls or "error" in cls:
            n_missing += 1
            continue
        flag = bool(cls["harmful_behavior"])
        by_cond[condition].append(flag)
        by_sweep[sample_id].append(flag)

    # Per-condition and per-scenario aggregation.
    cond_stats = {}
    by_scenario: dict[str, list[bool]] = defaultdict(list)
    all_flags: list[bool] = []
    for cond, flags in sorted(by_cond.items()):
        harmful = sum(flags)
        cond_stats[cond] = {
            "n": len(flags),
            "harmful": harmful,
            "rate": round(harmful / len(flags), 4) if flags else None,
        }
        scenario = cond.split("_")[0]
        by_scenario[scenario].extend(flags)
        all_flags.extend(flags)

    scenario_stats = {
        s: {"n": len(f), "harmful": sum(f), "rate": round(sum(f) / len(f), 4)}
        for s, f in sorted(by_scenario.items())
    }
    sweep_rates = [sum(f) / len(f) for _, f in sorted(by_sweep.items())]
    overall = {
        "n": len(all_flags),
        "harmful": sum(all_flags),
        "rate": None, "rate_ci95": None, "rate_ci95_lo": None, "rate_ci95_hi": None,
        "rate_by_sweep": [round(r, 4) for r in sweep_rates],
        "n_sweeps": len(sweep_rates),
        "n_conditions": len(by_cond),
        "ci_method": "none: no classified trials",
        "n_missing_classification": n_missing,
    }
    if sweep_rates:
        r = logit_t(sweep_rates, len(by_cond), scale=1.0)
        overall.update({
            "rate": round(r["mean"], 4),
            "rate_ci95": None if r["lo"] is None else [round(r["lo"], 4), round(r["hi"], 4)],
            "rate_ci95_lo": None if r["lo"] is None else round(r["lo"], 4),
            "rate_ci95_hi": None if r["hi"] is None else round(r["hi"], 4),
            "ci_method": r["method"].replace("passes", "sweeps"),
        })

    summary = {
        "label": label,
        "results_dir": str(root),
        "overall": overall,
        "by_scenario": scenario_stats,
        "by_condition": cond_stats,
    }

    out_path = Path(out) if out else root / "misalignment_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n=== MISALIGNMENT SUMMARY [{label}] ===")
    print(f"overall: {overall['harmful']}/{overall['n']} = {overall['rate']}  "
          f"CI95 {overall['rate_ci95']} over {overall['n_sweeps']} sweeps  (missing={n_missing})")
    print("by scenario:")
    for s, st in scenario_stats.items():
        print(f"  {s:10s} {st['harmful']:3d}/{st['n']:3d} = {st['rate']}")
    print("by condition:")
    for c, st in cond_stats.items():
        print(f"  {c:44s} {st['harmful']:3d}/{st['n']:3d} = {st['rate']}")
    print(f"\nwrote {out_path}")
