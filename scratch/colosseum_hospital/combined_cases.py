# ABOUTME: One slide figure: self-promotional (top row) and self-sacrificial (bottom row), each as the
# ABOUTME: four cases for team total, patients treated and the coalition's own slice.

"""uv run python scratch/colosseum_hospital/combined_cases.py [--open]"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import four_cases as fc  # sibling script; the script's own directory is on sys.path
from src.utils import figure_path

INK, MUTED = "#1B2430", "#5B6875"
ROWS = [
    ("self_promotional", "Self-promotional instruction: maximise your own share"),
    ("self_sacrificial", "Self-sacrificial instruction: wreck the team's score"),
]
COLS = [
    ("team total", "Team total (points)", 8000, "8,000 = every patient treated"),
    ("patients treated", "Patients treated per shift (of 8)", 8, "all 8 treated"),
    ("coalition slice", "Coalition's own slice (GH Triage's points)", None, None),
]
LABELS = [
    "control pair\nno coalition",
    "DA pair\nno coalition",
    "control pair\nin coalition",
    "DA pair\nin coalition",
]
STYLE = [
    ("baseline", "control", "#B7CCEA", "#2E6FBF"),
    ("baseline", "treatment", "#F0C9B7", "#C95B2F"),
    (None, "control", "#2E6FBF", "#2E6FBF"),
    (None, "treatment", "#C95B2F", "#C95B2F"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    for r, (condition, row_title) in enumerate(ROWS):
        fc.CONDITION = condition
        cells = fc.load()
        for c, (key, title, ref, ref_label) in enumerate(COLS):
            ax = axes[r, c]
            tops, lows = [], [0.0]
            for i, (cond, block, face, edge) in enumerate(STYLE):
                vals = fc.values(cells, key, cond or condition, block)
                m, hi = fc.bar_with_ci(
                    ax,
                    i,
                    vals,
                    rng,
                    width=0.62,
                    face=face,
                    edge=edge,
                    fmt="{:.1f}" if key == "patients treated" else "{:,.0f}",
                    fs=10,
                )
                tops.append(hi)
                lows.append(min(0.0, fc.ci95(vals, rng)[0]))
            if ref is not None:
                ax.axhline(ref, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
                ax.annotate(
                    ref_label,
                    (3.45, ref),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="right",
                    va="bottom",
                    fontsize=8.5,
                    color=MUTED,
                )
            ax.set_ylim(min(lows) * 1.1, max(max(tops), ref or 0) * 1.18)
            ax.set_xticks(range(4))
            ax.set_xticklabels(LABELS if r == 1 else [""] * 4, fontsize=9.5)
            ax.set_xlim(-0.6, 3.6)
            ax.set_title(title if r == 0 else "", fontsize=11, loc="left", color=INK)
            ax.grid(axis="y", alpha=0.25, zorder=0)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(axis="y", labelsize=9)
        axes[r, 0].annotate(
            row_title,
            (0, 1.0),
            xycoords="axes fraction",
            xytext=(-8, 26 if r == 0 else 8),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=11.5,
            color=INK,
            fontweight="bold",
            annotation_clip=False,
        )
    fig.tight_layout(h_pad=2.6, w_pad=1.6, rect=(0, 0, 1, 0.97))
    png = figure_path(
        Path("output/colosseum_hospital/analysis"),
        "colosseum_hospital_two_instructions",
    )
    fig.savefig(png, dpi=160)
    print(f"figure {png}")
    if args.open:
        subprocess.run(["open", str(png)], check=False)


if __name__ == "__main__":
    main()
