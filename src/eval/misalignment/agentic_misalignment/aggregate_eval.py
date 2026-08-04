# ABOUTME: Aggregates agentic-misalignment classification results into per-condition
# ABOUTME: and overall misalignment rates (handles the models/<model>/<condition> layout).

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path



def _iter_responses(results_dir: Path):
    """Yield (condition, response_dict) for every classified response.json found."""
    for resp_file in results_dir.rglob("response.json"):
        # condition is the parent of sample_xxx
        condition = resp_file.parent.parent.name
        try:
            data = json.loads(resp_file.read_text())
        except json.JSONDecodeError:
            continue
        yield condition, data


def main(results_dir: str, label: str = "", out: str | None = None) -> None:
    """Compute misalignment rates from a results directory.

    Args:
        results_dir: Path to results/<experiment_id> (or its models/ subdir).
        label: Optional label (e.g. "baseline") recorded in the output.
        out: Optional path to write the JSON summary; defaults alongside results.
    """
    root = Path(results_dir)
    by_cond: dict[str, list[bool]] = defaultdict(list)
    n_missing = 0
    for condition, data in _iter_responses(root):
        cls = data.get("classification")
        if not cls or "harmful_behavior" not in cls or "error" in cls:
            n_missing += 1
            continue
        by_cond[condition].append(bool(cls["harmful_behavior"]))

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
    overall = {
        "n": len(all_flags),
        "harmful": sum(all_flags),
        "rate": round(sum(all_flags) / len(all_flags), 4) if all_flags else None,
        "n_missing_classification": n_missing,
    }

    summary = {
        "label": label,
        "results_dir": str(root),
        "overall": overall,
        "by_scenario": scenario_stats,
        "by_condition": cond_stats,
    }

    out_path = Path(out) if out else root / "misalignment_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"\n=== MISALIGNMENT SUMMARY [{label}] ===")
    print(f"overall: {overall['harmful']}/{overall['n']} = "
          f"{overall['rate']}  (missing={n_missing})")
    print("by scenario:")
    for s, st in scenario_stats.items():
        print(f"  {s:10s} {st['harmful']:3d}/{st['n']:3d} = {st['rate']}")
    print("by condition:")
    for c, st in cond_stats.items():
        print(f"  {c:44s} {st['harmful']:3d}/{st['n']:3d} = {st['rate']}")
    print(f"\nwrote {out_path}")

