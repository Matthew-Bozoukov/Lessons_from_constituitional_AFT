# ABOUTME: The four self-promotional cases on one figure — control vs difficult-advice pair, with and
# ABOUTME: without the coalition instruction — for team total, patients treated and the coalition's slice.

"""Untempted vs tempted, both models, three measures.

    uv run python scratch/colosseum_hospital/four_cases.py

Each panel: x = no coalition (baseline) -> coalition (self-promotional); one line per model
through the cell means (95% bootstrap CI), seed dots behind. Writes the PNG and a results.md
beside it under output/colosseum_hospital/analysis/.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils import figure_path

EPISODES = Path("output/colosseum_hospital/analysis/episodes.json")
MERGED = Path("output/colosseum_hospital/merged")
OUT = Path("output/colosseum_hospital/analysis")
COLOR = {"control": "#2E6FBF", "treatment": "#C95B2F"}
NAME = {
    "control": "control model in the pair",
    "treatment": "difficult-advice model in the pair",
}
STATES = [
    ("baseline", "no coalition\n(cooperative prompt)"),
    ("self_promotional", "coalition\n(self-promotional instruction)"),
]
SEED = re.compile(r"_seed(\d+)$")


def load() -> dict[tuple[str, str], list[dict]]:
    eps = json.loads(EPISODES.read_text())
    triage: dict[tuple[str, str, int], float] = {}
    for cell in MERGED.glob("2026-09-04_colosseum_hospital_*"):
        cond = (
            "baseline"
            if "_baseline_" in cell.name
            else "self_promotional"
            if "_self_promotional_" in cell.name
            else None
        )
        if cond is None:
            continue
        block = "treatment" if "difficult_advice" in cell.name else "control"
        for f in cell.glob("rollouts/colosseum/*/runs/*/*/*/agent_rewards.json"):
            seed = int(SEED.search(f.parent.name).group(1))
            triage[(cond, block, seed)] = float(
                json.loads(f.read_text()).get("General_Hospital_Triage", 0.0)
            )
    cells: dict[tuple[str, str], list[dict]] = {}
    for e in eps:
        if e["condition"] not in ("baseline", "self_promotional"):
            continue
        key = (e["condition"], e["block"])
        cells.setdefault(key, []).append(
            {
                "seed": e["seed"],
                "team total": float(e["joint_reward"]),
                "patients treated": float(e["converged_patients"]),
                "coalition slice": triage.get(
                    (e["condition"], e["block"], e["seed"]), float("nan")
                ),
            }
        )
    return cells


def ci95(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    boots = rng.choice(x, size=(4000, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    cells = load()
    rng = np.random.default_rng(0)
    panels = [
        (
            "team total",
            "Team total (points; 8,000 = every patient treated, no penalties)",
            "points",
        ),
        ("patients treated", "Patients treated per shift (of 8)", "patients"),
        (
            "coalition slice",
            "Coalition's own slice (GH Triage's points; provisioner earns 0)",
            "points",
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    lines: list[str] = [
        "# Colosseum Hospital — the four self-promotional cases\n",
        "| measure | model in the pair | no coalition | coalition | change |",
        "|---|---|---|---|---|",
    ]
    for ax, (key, title, unit) in zip(axes, panels):
        for block in ("control", "treatment"):
            xs, means = [], []
            for i, (cond, _) in enumerate(STATES):
                rows = cells[(cond, block)]
                vals = np.array([r[key] for r in rows if not np.isnan(r[key])])
                x = i + (-0.13 if block == "control" else 0.13)
                ax.scatter(
                    x + rng.normal(0, 0.03, len(vals)),
                    vals,
                    s=10,
                    alpha=0.28,
                    color=COLOR[block],
                    linewidths=0,
                    zorder=1,
                )
                lo, hi = ci95(vals, rng)
                ax.errorbar(
                    x,
                    vals.mean(),
                    yerr=[[vals.mean() - lo], [hi - vals.mean()]],
                    fmt="o",
                    color=COLOR[block],
                    ms=7,
                    lw=2,
                    capsize=0,
                    zorder=3,
                    markeredgecolor="white",
                    markeredgewidth=1.5,
                )
                xs.append(x)
                means.append(vals.mean())
                fmt = (
                    f"{vals.mean():.1f}"
                    if unit == "patients"
                    else f"{vals.mean():,.0f}"
                )
                ax.annotate(
                    fmt,
                    (x, vals.mean()),
                    xytext=(9 if block == "treatment" else -9, 0),
                    textcoords="offset points",
                    ha="left" if block == "treatment" else "right",
                    va="center",
                    fontsize=8.5,
                    color="#1B2430",
                )
            ax.plot(
                xs,
                means,
                color=COLOR[block],
                lw=2,
                alpha=0.9,
                zorder=2,
                label=NAME[block],
            )
            a, b = [
                np.array([r[key] for r in cells[(c, block)] if not np.isnan(r[key])])
                for c, _ in STATES
            ]
            lines.append(
                f"| {key} | {NAME[block]} | {a.mean():,.2f} | {b.mean():,.2f} | {b.mean() - a.mean():+,.2f} |"
            )
        if key == "team total":
            ax.axhline(8000, color="#5B6875", lw=1, ls=(0, (4, 3)), zorder=0)
            ax.annotate(
                "8,000",
                (1.42, 8000),
                fontsize=8,
                color="#5B6875",
                va="bottom",
                ha="right",
            )
        if key == "patients treated":
            ax.set_ylim(-0.3, 8.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([label for _, label in STATES], fontsize=9)
        ax.set_xlim(-0.55, 1.55)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(axis="y", alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", labelsize=9)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle(
        "Colosseum Hospital — the four self-promotional cases: what the coalition instruction changes (seed dots, mean ± 95% CI; 30–60 shifts per point)",
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    png = figure_path(OUT, "colosseum_hospital_four_cases")
    fig.savefig(png, dpi=160)
    md = png.with_name(png.stem + "_results.md")
    md.write_text(
        "\n".join(lines)
        + "\n\nCells: baseline n=30 per model; self-promotional n=60 per model. Seven non-pair seats always hold the control model.\n"
    )
    print("\n".join(lines))
    print(f"\nfigure {png}")


if __name__ == "__main__":
    main()
