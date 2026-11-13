# ABOUTME: Zoomed business-vs-ethics scatter for the 2026-11-13 Shen sepsis runs: only the region the runs occupy,
# ABOUTME: every point named directly (arm + setup), arrow single -> org per arm, no legend to decode.
"""Run: uv run python scratch/shen_software/fig_sepsis_scatter_zoom.py <runs dir> <shen test.json> [prefix]

Same data and scoring as fig_sepsis.py (Shen's fixed bounds); only the drawing differs.
"""

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.argv = sys.argv[:4]
spec = importlib.util.spec_from_file_location(
    "fs", Path(__file__).with_name("fig_sepsis.py")
)
fs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fs)
plt.close("all")

from src.utils import save_figure_pdf

rows, ARMS, COLOUR = fs.rows, fs.ARMS, fs.COLOUR
SHORT = {
    "qwen36_0_plain": "Control",
    "qwen36_0_da_15": "DA",
    "qwen36_0_da_tools_15": "DA+tools",
}
STEM = f"2026-11-13_shen_sepsis_scatter_zoom{'_' + fs.PREFIX.rstrip('_') if fs.PREFIX else ''}"
# Where each name goes relative to its marker (points); chosen so near-coincident runs do not collide.
OFFSET = {
    ("single_sepsis", "qwen36_0_plain"): (12, -4),
    ("single_sepsis", "qwen36_0_da_tools_15"): (12, -4),
    ("single_sepsis", "qwen36_0_da_15"): (-12, 12),
    ("org8_sepsis", "qwen36_0_plain"): (12, 10),
    ("org8_sepsis", "qwen36_0_da_15"): (12, -18),
    ("org8_sepsis", "qwen36_0_da_tools_15"): (12, -16),
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig, ax = plt.subplots(figsize=(8.5, 6))
for a, _ in ARMS:
    s1, s2 = rows[("single_sepsis", a)], rows[("org8_sepsis", a)]
    if s1 and s2:
        ax.annotate(
            "",
            xy=(s2["business"], s2["ethics"]),
            xytext=(s1["business"], s1["ethics"]),
            arrowprops=dict(
                arrowstyle="-|>",
                color=COLOUR[a],
                lw=2,
                alpha=0.45,
                shrinkA=10,
                shrinkB=10,
                mutation_scale=16,
            ),
            zorder=2,
        )
missing = []
for s, slab, mk in fs.SETUPS:
    for a, _ in ARMS:
        r = rows[(s, a)]
        name = f"{SHORT[a]}, {'single' if s == 'single_sepsis' else 'org'}"
        if r is None:
            missing.append(name)
            continue
        face = "white" if (r["recomputed"] or s == "single_sepsis") else COLOUR[a]
        ax.scatter(
            [r["business"]],
            [r["ethics"]],
            marker=mk,
            s=220,
            facecolor=face,
            edgecolor=COLOUR[a],
            linewidth=2.8,
            zorder=4,
        )
        dx, dy = OFFSET.get((s, a), (12, -4))
        ax.annotate(
            f"{name}  ({r['business']:.2f}, {r['ethics']:.2f})",
            (r["business"], r["ethics"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx > 0 else "right",
            fontsize=11,
            color=COLOUR[a],
            fontweight="bold",
        )
ax.scatter([0], [1], marker="x", s=70, color="#888888", zorder=3)
ax.annotate(
    "treat everyone",
    (0, 1),
    xytext=(8, 8),
    textcoords="offset points",
    fontsize=10,
    color="#888888",
)
ax.set_xlim(-0.03, 0.62)
ax.set_ylim(0.70, 1.04)
ax.set_xlabel("Business score  (higher = cheaper)")
ax.set_ylabel("Ethics score  (higher = fewer sepsis cases missed)")
ax.grid(color="#eeeeee", lw=1)
ax.set_axisbelow(True)
ax.set_title(
    "Sepsis triage: hollow = single agent, square = org; arrow = single → org",
    fontsize=12,
    loc="left",
)
note = (
    "Zoomed in: the perfect classifier (treat exactly the 28 sepsis patients) scores (0.72, 1.00), off the chart "
    "to the right.\nOne run per point."
    + "".join(
        f" {SHORT[a]} {'single' if s == 'single_sepsis' else 'org'} recomputed from its data/decisions.csv"
        " (Shen's scorer reads only the root file)."
        for (s, a), r in rows.items()
        if r and r["recomputed"]
    )
)
if missing:
    note += f" No prediction script: {', '.join(missing)}."
fig.text(0.01, -0.02, note, fontsize=10, color="#555555", va="top")
fig.tight_layout()
pdf = save_figure_pdf(
    fig,
    fs.OUT / f"{STEM}.pdf",
    title="Shen sepsis: zoomed business vs ethics scatter",
    artifacts=list(fs.HUB.values()),
    script="scratch/shen_software/fig_sepsis_scatter_zoom.py",
)
fig.savefig(fs.OUT / f"{STEM}.png", dpi=170, bbox_inches="tight")
print(pdf)
