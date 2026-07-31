#!/usr/bin/env python3
"""Compare two ODCV models by analyst-defined scenario group on matched cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_odcv_scenario_groups import GROUPS, VIOLATION_THRESHOLD


def bootstrap_ci(values: dict[str, list[bool]], seed: int) -> tuple[float, float]:
    """Cluster-bootstrap a rate by scenario, retaining both variants."""
    clusters = list(values.values())
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(10_000):
        selected = rng.integers(0, len(clusters), len(clusters))
        observations = [value for index in selected for value in clusters[index]]
        draws.append(100 * np.mean(observations))
    return tuple(np.percentile(draws, [2.5, 97.5]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_results", type=Path)
    parser.add_argument("comparison_results", type=Path)
    parser.add_argument("--base-label", default="Base Qwen3.6-27B (FP8)")
    parser.add_argument("--comparison-label", default="Three-way constitution LoRA")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    datasets = [
        json.loads(args.base_results.read_text())["per_scenario_medians"],
        json.loads(args.comparison_results.read_text())["per_scenario_medians"],
    ]
    common = {
        (variant, scenario)
        for variant in ("mandated", "incentivized")
        for scenario in set(datasets[0][variant]) & set(datasets[1][variant])
    }
    assert common, "the runs have no matching evaluations"
    declared = set().union(*GROUPS.values())
    assert {scenario for _, scenario in common} <= declared, "unclassified scenario"

    records = []
    for group, scenarios in GROUPS.items():
        group_cases = sorted((v, s) for v, s in common if s in scenarios)
        for model_index, label in enumerate((args.base_label, args.comparison_label)):
            by_scenario: dict[str, list[bool]] = {}
            for variant, scenario in group_cases:
                violation = datasets[model_index][variant][scenario] >= VIOLATION_THRESHOLD
                by_scenario.setdefault(scenario, []).append(violation)
            observations = [v for values in by_scenario.values() for v in values]
            lo, hi = bootstrap_ci(by_scenario, args.seed + model_index)
            records.append({
                "group": group,
                "model": label,
                "matched_evaluations": len(observations),
                "scenario_families": len(by_scenario),
                "violations": sum(observations),
                "misalignment_rate_pct": 100 * np.mean(observations),
                "ci95_low_pct": lo,
                "ci95_high_pct": hi,
            })

    output_dir = args.output_dir or args.comparison_results.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "base_vs_threeway_misalignment_by_scenario_group.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    groups = list(GROUPS)
    x = np.arange(len(groups))
    width = .36
    fig, ax = plt.subplots(figsize=(13, 7))
    colors = ("#e1812c", "#4c78a8")
    for model_index, (label, color) in enumerate(zip((args.base_label, args.comparison_label), colors)):
        rows = [next(r for r in records if r["group"] == group and r["model"] == label)
                for group in groups]
        rates = np.asarray([r["misalignment_rate_pct"] for r in rows])
        lows = np.asarray([r["ci95_low_pct"] for r in rows])
        highs = np.asarray([r["ci95_high_pct"] for r in rows])
        positions = x + (model_index - .5) * width
        bars = ax.bar(positions, rates, width, color=color, label=label, alpha=.92)
        ax.errorbar(positions, rates, yerr=[rates - lows, highs - rates], fmt="none",
                    ecolor="#303030", capsize=4, lw=1.5)
        for bar, row in zip(bars, rows):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                    f'{row["misalignment_rate_pct"]:.1f}%\n({row["violations"]}/{row["matched_evaluations"]})',
                    ha="center", va="bottom", fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .75, "pad": 1})

    ax.set_xticks(x, [g.replace(" & ", " &\n") for g in groups])
    ax.set_ylabel("Misalignment rate (%) — median judge severity ≥ 3")
    ax.set_title("ODCV misalignment by scenario group: base vs three-way constitution LoRA")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False)
    fig.text(.99, .01,
             "Matched completed evaluations only; error bars are 95% scenario-cluster bootstrap CIs. Groups are analyst-defined.",
             ha="right", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .04, 1, 1))
    plot_path = output_dir / "base_vs_threeway_misalignment_by_scenario_group.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(plot_path)
    print(csv_path)


if __name__ == "__main__":
    main()
