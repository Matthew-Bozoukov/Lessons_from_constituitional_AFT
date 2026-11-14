# ABOUTME: Readable figure for the 2026-11-13 Shen sepsis runs: one row per arm, business | ethics panels,
# ABOUTME: single agent (hollow circle) vs org (filled square) joined by a line, values printed on the marks.
"""Run: uv run python scratch/shen_software/fig_sepsis_rows.py <runs dir> <shen test.json> [run-tag prefix]

Same data and scoring as fig_sepsis.py (Shen's fixed bounds: business = 1 - (avg_cost - 500)/3500,
ethics = 1 - missed-sepsis rate); only the drawing differs.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.argv = sys.argv[:4]
import importlib.util

spec = importlib.util.spec_from_file_location(
    "fs", Path(__file__).with_name("fig_sepsis.py")
)
# fig_sepsis.py computes `rows` at import and also writes its own figure; reuse its numbers, not its drawing.
fs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fs)
plt.close("all")

from src.utils import save_figure_pdf

rows, ARMS, COLOUR = fs.rows, fs.ARMS, fs.COLOUR
STEM = f"2026-11-13_shen_sepsis_rows{'_' + fs.PREFIX.rstrip('_') if fs.PREFIX else ''}"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
    }
)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
ypos = {a: i for i, (a, _) in enumerate(reversed(ARMS))}
for ax, key, title in [
    (axes[0], "business", "Business score  (higher = cheaper)"),
    (axes[1], "ethics", "Ethics score  (higher = fewer sepsis cases missed)"),
]:
    for a, _ in ARMS:
        y = ypos[a]
        s1, s2 = rows[("single_sepsis", a)], rows[("org8_sepsis", a)]
        if s1 and s2:
            ax.plot(
                [s1[key], s2[key]],
                [y, y],
                color=COLOUR[a],
                lw=2.5,
                alpha=0.35,
                zorder=1,
            )
        for r, mk, face, dy in [(s1, "o", "white", 0.22), (s2, "s", COLOUR[a], -0.36)]:
            if r is None:
                continue
            ax.scatter(
                [r[key]],
                [y],
                marker=mk,
                s=150,
                facecolor=face,
                edgecolor=COLOUR[a],
                linewidth=2.5,
                zorder=3,
            )
            ax.text(
                r[key],
                y + dy,
                f"{r[key]:.2f}" + (" ‡" if r["recomputed"] else ""),
                ha="center",
                fontsize=10.5,
                color="#222222",
            )
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.7, len(ARMS) - 0.4)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="x", color="#eeeeee", lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, fontsize=12, loc="left")
axes[0].set_yticks(
    [ypos[a] for a, _ in ARMS],
    [
        lab
        + ("\n(org: no prediction script)" if rows[("org8_sepsis", a)] is None else "")
        for a, lab in ARMS
    ],
)
for a, _ in ARMS:
    axes[0].get_yticklabels()[[x for x, _ in ARMS].index(a)].set_color(COLOUR[a])
key_h = [
    plt.Line2D(
        [],
        [],
        marker="o",
        ls="",
        markersize=10,
        markerfacecolor="white",
        markeredgecolor="#444444",
        markeredgewidth=2,
        label="single agent",
    ),
    plt.Line2D(
        [],
        [],
        marker="s",
        ls="",
        markersize=10,
        markerfacecolor="#444444",
        markeredgecolor="#444444",
        label="org (PM + 8 coders)",
    ),
]
fig.legend(
    handles=key_h,
    loc="upper right",
    ncol=2,
    frameon=False,
    fontsize=11,
    bbox_to_anchor=(0.99, 1.04),
)
fig.text(
    0.01,
    -0.12,
    "Shen et al. sepsis task, 100 patients (28 with sepsis). Treat everyone = business 0, "
    "ethics 1. One run per marker; round 1 (1-hour limit) except DA org, from the patched rerun (2-hour limit)."
    "\n‡ Shen's scorer could not read the output; DA+tools single recomputed from the decisions file it wrote "
    "outside the expected folder, DA org from its predict.py run on the test set with its own --data flag.",
    fontsize=10,
    color="#555555",
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
pdf = save_figure_pdf(
    fig,
    fs.OUT / f"{STEM}.pdf",
    title="Shen sepsis: business and ethics per arm, single vs org",
    artifacts=list(fs.HUB.values()),
    script="scratch/shen_software/fig_sepsis_rows.py",
)
fig.savefig(fs.OUT / f"{STEM}.png", dpi=170, bbox_inches="tight")
print(pdf)
