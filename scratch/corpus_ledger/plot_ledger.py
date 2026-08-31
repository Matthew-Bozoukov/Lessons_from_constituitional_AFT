# ABOUTME: the whole-ledger figure: every ODCV-scored arm on one 65-cell axis, coloured by whether
# ABOUTME: the synthetic half is difficult advice, another document type, or no synthetic data.
# Run: uv run python scratch/corpus_ledger/plot_ledger.py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

C1, C2, C3 = "#3A5BA9", "#B0620A", "#00907C"  # validated categorical triple
INK, MUTED, RULE, PAPER = "#12151C", "#5A6474", "#D5DAE2", "#FCFCFB"

ARMS = [  # (label, misalignment %, group)
    ("meta-cognition excised", 8.5, 1),
    ("meta-cognition masked", 9.6, 1),
    ("reasoning made one-sided", 9.7, 1),
    ("branches + engagement stripped", 10.1, 1),
    ("difficult advice (control)", 10.2, 1),
    ("traits 1/3/4 swapped", 10.4, 1),
    ("difficult advice, grok-written", 11.5, 1),
    ("difficult advice, chunk-only", 11.5, 1),
    ("reasoning as bare rules", 11.6, 1),
    ("low stakes (rewritten)", 12.4, 1),
    ("traits 1+3 deleted", 12.9, 1),
    ("difficult advice, length-capped", 15.4, 1),
    ("difficult advice, Sonnet", 16.3, 1),
    ("low stakes (by construction)", 16.9, 1),
    ("eval-like scenarios swapped out", 17.1, 1),
    ("post-action retrospection", 19.5, 2),
    ("difficult advice, curiosity trait", 19.7, 1),
    ("difficult advice, GPT-written", 21.2, 1),
    ("post-action retrospection, coherent", 21.3, 2),
    ("base model, no SFT", 36.9, 3),
    ("zero synthetic rows", 43.9, 3),
    ("maths rows instead", 44.1, 3),
    ("good-AI fiction", 45.3, 2),
]
COL = {1: C1, 2: C2, 3: C3}
LBL = {1: "difficult-advice family", 2: "other document type", 3: "no-SFT reference"}

fig, ax = plt.subplots(figsize=(9.6, 8.4), dpi=200)
fig.patch.set_facecolor(PAPER)
ax.set_facecolor(PAPER)
ys = list(range(len(ARMS)))[::-1]
for y, (lab, mr, g) in zip(ys, ARMS):
    ax.barh(y, mr, height=0.62, color=COL[g], zorder=3)
    ax.text(
        mr + 0.55,
        y,
        f"{mr:.1f}",
        va="center",
        ha="left",
        fontsize=8.2,
        color=INK,
        family="monospace",
        zorder=4,
    )

ax.axvline(36.9, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
ax.text(
    36.9,
    len(ARMS) - 0.1,
    "  untrained base, 36.9%",
    fontsize=8,
    color=MUTED,
    va="bottom",
)

ax.set_yticks(ys)
ax.set_yticklabels([a[0] for a in ARMS], fontsize=8.6, color=INK)
ax.set_xlim(0, 52)
ax.set_ylim(-0.9, len(ARMS) + 0.6)
ax.set_xlabel(
    "ODCV-Bench misalignment rate (%)  ·  65 shared cells  ·  lower is better",
    fontsize=9,
    color=MUTED,
    labelpad=10,
)
ax.set_title(
    "Every constitutional-SFT arm on one axis\n"
    "the synthetic 716 rows are the only thing that varies",
    fontsize=13,
    color=INK,
    loc="left",
    pad=16,
)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(RULE)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", colors=MUTED, labelsize=8.5)
ax.grid(axis="x", color=RULE, lw=0.7, zorder=1)
ax.set_axisbelow(True)

handles = [plt.Rectangle((0, 0), 1, 1, color=COL[g]) for g in (1, 2, 3)]
ax.legend(
    handles,
    [LBL[g] for g in (1, 2, 3)],
    loc="upper right",
    bbox_to_anchor=(1.0, 0.90),
    frameon=False,
    fontsize=8.6,
    labelcolor=INK,
    handlelength=1.0,
    handleheight=1.0,
)
fig.text(
    0.012,
    0.012,
    "Three ablation arms (0.8-3.2%) are excluded: 60-78% of their rollouts made no tool call, "
    "so their scores measure refusal-to-engage.",
    fontsize=7.4,
    color=MUTED,
)
plt.tight_layout(rect=(0, 0.028, 1, 1))
out = Path("output/plots/2026-08-31_odcv_all_arms_ledger_65cells.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, facecolor=PAPER)
print("wrote", out)
