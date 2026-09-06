# ABOUTME: One bar chart for the slides: team total per model (control pair, DA pair) under no
# ABOUTME: coalition, the self-promotional instruction and the self-sacrificial instruction.

"""uv run python scratch/colosseum_hospital/combined_cases.py [--open]"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import four_cases as fc  # sibling script; the script's own directory is on sys.path
from src.utils import figure_path

INK, MUTED = "#1B2430", "#5B6875"
MODELS = [
    ("control", "control pair", "#2E6FBF", "#B7CCEA"),
    ("treatment", "DA pair", "#C95B2F", "#F0C9B7"),
]
BARS = [
    ("baseline", "no coalition", "light", None),
    ("self_promotional", "self-promotional instruction", "solid", None),
    ("self_sacrificial", "self-sacrificial instruction", "solid", "////"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    cells = {}
    for condition in ("self_promotional", "self_sacrificial"):
        fc.CONDITION = condition
        cells.update(fc.load())

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    width, offsets = 0.26, (-0.28, 0.0, 0.28)
    tops = []
    lines = [
        "| model in the pair | no coalition | self-promotional | self-sacrificial |",
        "|---|---|---|---|",
    ]
    for g, (block, label, solid, light) in enumerate(MODELS):
        row = []
        for (cond, _, fill, hatch), dx in zip(BARS, offsets):
            vals = fc.values(cells, "team total", cond, block)
            face = light if fill == "light" else ("white" if hatch else solid)
            m, hi = fc.bar_with_ci(
                ax,
                g + dx,
                vals,
                rng,
                width=width,
                face=face,
                edge=solid,
                hatch=hatch,
                fs=10,
            )
            tops.append(hi)
            row.append(f"{m:,.0f}")
        lines.append(f"| {label} | " + " | ".join(row) + " |")
    ax.axhline(8000, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.annotate(
        "8,000 = every patient treated, no penalties",
        (1.42, 8000),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=MUTED,
    )
    ax.set_ylim(0, max(max(tops), 8000) * 1.12)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([m[1] for m in MODELS], fontsize=11)
    ax.set_xlim(-0.6, 1.6)
    ax.set_title("Team total (points)", fontsize=12, loc="left", color=INK)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", labelsize=10)
    ax.legend(
        handles=[
            Patch(facecolor="#D5DCE3", edgecolor=MUTED, label="no coalition"),
            Patch(
                facecolor=MUTED, edgecolor=MUTED, label="self-promotional instruction"
            ),
            Patch(
                facecolor="white",
                edgecolor=MUTED,
                hatch="////",
                label="self-sacrificial instruction",
            ),
        ],
        frameon=False,
        fontsize=9.5,
        loc="lower left",
    )
    fig.tight_layout()
    png = figure_path(
        Path("output/colosseum_hospital/analysis"),
        "colosseum_hospital_two_instructions",
    )
    fig.savefig(png, dpi=160)
    (png.with_name(png.stem + "_results.md")).write_text(
        "\n".join(lines)
        + "\n\nn = 30 shifts per untempted and self-sacrificial cell, 60 per self-promotional cell; 95% bootstrap CI whiskers.\n"
    )
    print("\n".join(lines))
    print(f"\nfigure {png}")
    if args.open:
        subprocess.run(["open", str(png)], check=False)


if __name__ == "__main__":
    main()
