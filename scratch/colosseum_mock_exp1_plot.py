# ABOUTME: SIMULATED Experiment-1 figure — the shape a Colosseum collusion result would take,
# ABOUTME: drawn from invented numbers so the analysis is agreed before any GPU is rented.

"""Mock-up of Experiment 1 (multi-agent collusion) in the repo's figure style.

Every number here is INVENTED. The point is the *form* of the answer: which
contrast is on which axis, what is paired with what, and what "the treatment
colludes less" would actually look like on a page. When the real runs land,
`src/eval/misalignment/colosseum/report.py` draws this same figure from
`results.json` and the word SIMULATED comes off.

Panel A — coalition advantage, the payoff collusion buys the two-agent coalition.
          Paired within seed (collusion minus baseline), so the 40 dots are 40
          seeds and the interval is over seed-to-seed variance.
Panel B — normalised overall regret, the cost to the six-agent team, shown in
          both conditions so a treatment that "colludes less" by simply being
          worse at the task is visible rather than hidden.

Run: uv run python scratch/colosseum_mock_exp1_plot.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.utils import figure_path

CONTROL = "#2F6DB5"  # validated categorical pair (dataviz six-checks, light surface)
CONTROL_EDGE = "#1B4472"
TREAT = "#D97706"
TREAT_EDGE = "#7C4406"
INK = "#333333"
MUTED = "#8A8A8A"

N_SEEDS = 40
RNG = np.random.default_rng(20260903)

# ── invented effects ────────────────────────────────────────────────────────
# Control coalition captures a large share of the surplus; treatment captures
# less but not none. Spread is deliberately wide: with 40 seeds a real effect
# this size is detectable but not overwhelming, which is the honest expectation.
adv = {
    "control": RNG.normal(0.28, 0.11, N_SEEDS),
    "treatment": RNG.normal(0.17, 0.11, N_SEEDS),
}
# Normalised regret, 0 = optimal joint allocation, 1 = worst. Baseline condition
# is the cooperative floor; the collusion condition is where the damage shows.
regret = {
    ("control", "baseline"): RNG.normal(0.09, 0.03, N_SEEDS),
    ("treatment", "baseline"): RNG.normal(0.10, 0.03, N_SEEDS),
    ("control", "collusion"): RNG.normal(0.41, 0.09, N_SEEDS),
    ("treatment", "collusion"): RNG.normal(0.29, 0.09, N_SEEDS),
}


def mean_ci(x):
    """Mean and a normal-approximation 95% interval over seeds."""
    m = float(np.mean(x))
    h = 1.96 * float(np.std(x, ddof=1)) / np.sqrt(len(x))
    return m, m - h, m + h


plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 17,
        "xtick.labelsize": 16,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
        "axes.edgecolor": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    }
)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(15.5, 6.4), facecolor="white")
fig.suptitle(
    "SIMULATED — Experiment 1: does constitutional SFT reduce multi-agent collusion?",
    fontsize=23,
    y=1.005,
)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E4E4E4", linewidth=1, zorder=0)
    ax.set_axisbelow(True)


# ── Panel A: coalition advantage ────────────────────────────────────────────
style(axA)
for i, (arm, colour, edge) in enumerate(
    [("control", CONTROL, CONTROL_EDGE), ("treatment", TREAT, TREAT_EDGE)]
):
    v = adv[arm]
    jitter = RNG.uniform(-0.09, 0.09, N_SEEDS)
    axA.scatter(
        np.full(N_SEEDS, i) + jitter,
        v,
        s=42,
        color=colour,
        alpha=0.30,
        linewidths=0,
        zorder=2,
    )
    m, lo, hi = mean_ci(v)
    axA.errorbar(
        i,
        m,
        yerr=[[m - lo], [hi - m]],
        fmt="o",
        markersize=13,
        color=colour,
        markeredgecolor=edge,
        markeredgewidth=2,
        ecolor=edge,
        elinewidth=2.5,
        capsize=7,
        capthick=2.5,
        zorder=3,
    )
    axA.annotate(
        f"{m:.2f}",
        (i, hi),
        textcoords="offset points",
        xytext=(0, 12),
        ha="center",
        fontsize=16,
        color=INK,
        fontweight="bold",
    )

axA.axhline(0, color=MUTED, linewidth=1.5, linestyle=(0, (4, 3)), zorder=1)
axA.annotate(
    "no advantage from the side-channel",
    (1.42, 0.005),
    ha="right",
    va="bottom",
    fontsize=13,
    color=MUTED,
)
axA.set_xticks([0, 1])
axA.set_xticklabels(["Tulu-only\ncontrol", "7% principle-only\ndifficult advice"])
axA.set_xlim(-0.45, 1.45)
axA.set_ylabel("Coalition advantage\n(collusion − baseline, paired by seed)")
axA.set_title("A · what collusion buys the coalition", pad=14)

# ── Panel B: normalised overall regret ──────────────────────────────────────
style(axB)
conditions = ["baseline", "collusion"]
for arm, colour, edge in [
    ("control", CONTROL, CONTROL_EDGE),
    ("treatment", TREAT, TREAT_EDGE),
]:
    xs = np.arange(len(conditions))
    ms, los, his = zip(*(mean_ci(regret[(arm, c)]) for c in conditions))
    off = -0.055 if arm == "control" else 0.055
    axB.plot(xs + off, ms, color=colour, linewidth=2.5, zorder=2)
    axB.errorbar(
        xs + off,
        ms,
        yerr=[np.array(ms) - np.array(los), np.array(his) - np.array(ms)],
        fmt="o",
        markersize=12,
        color=colour,
        markeredgecolor=edge,
        markeredgewidth=2,
        ecolor=edge,
        elinewidth=2.5,
        capsize=7,
        capthick=2.5,
        zorder=3,
        linestyle="none",
    )

axB.annotate(
    "Tulu-only control",
    (1.055, regret[("control", "collusion")].mean()),
    textcoords="offset points",
    xytext=(10, 4),
    fontsize=15,
    color=CONTROL_EDGE,
    fontweight="bold",
)
axB.annotate(
    "7% principle-only",
    (1.055, regret[("treatment", "collusion")].mean()),
    textcoords="offset points",
    xytext=(10, -16),
    fontsize=15,
    color=TREAT_EDGE,
    fontweight="bold",
)
axB.set_xticks(np.arange(len(conditions)))
axB.set_xticklabels(["baseline", "collusion"])
axB.set_xlabel(
    "baseline = no side-channel, no private objective\ncollusion = two seats share both",
    fontsize=13, color=MUTED, labelpad=10,
)
axB.set_xlim(-0.35, 1.95)
axB.set_ylim(0, 0.58)
axB.set_ylabel("Normalised overall regret\n(0 = optimal joint allocation)")
axB.set_title("B · what it costs the six-agent team", pad=14)
axB.legend(
    handles=[
        Patch(facecolor=CONTROL, edgecolor=CONTROL_EDGE, label="Tulu-only control"),
        Patch(
            facecolor=TREAT,
            edgecolor=TREAT_EDGE,
            label="7% principle-only difficult advice",
        ),
    ],
    loc="upper left",
    frameon=False,
)

fig.text(
    0.5,
    -0.105,
    "SIMULATED DATA — invented numbers, drawn to agree the analysis before any GPU is rented. "
    f"n = {N_SEEDS} seeds per cell, shared across arms; bars are 95% CI over seeds.",
    ha="center",
    fontsize=13,
    color=MUTED,
)

path = figure_path("output/colosseum", "colosseum_jira_collusion_simulated")
fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {path}")
