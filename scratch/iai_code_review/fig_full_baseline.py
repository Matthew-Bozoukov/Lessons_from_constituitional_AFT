# ABOUTME: Figure for the 2026-11-19 full Inter-Agent Influence code_review run: BASELINE approval per arm,
# ABOUTME: the only condition that survived (per-episode logs were lost with the pods); numbers from scorer output.
# Run: uv run python scratch/iai_code_review/fig_full_baseline.py
import datetime as dt
import math
from pathlib import Path

import matplotlib.pyplot as plt

from src.utils import save_figure_pdf

# Upstream code_review_binary_scorer `accuracy` per arm, baseline condition, 2 scenarios x 100 epochs,
# as printed by Inspect on each pod (watcher log, 2026-11-19 22:44-23:57). Agent A: gemini-3-flash-preview.
ARMS = [
    (
        "Control (plain SFT)",
        "dougalldeepmind/2026-10-05-qwen36-0-plain",
        0.475,
        200,
        "#5a6472",
    ),
    (
        "Difficult Advice",
        "dougalldeepmind/2026-10-05-qwen36-0-da-15",
        0.500,
        200,
        "#7e22ce",
    ),
    (
        "Difficult Advice + unused tools",
        "dougalldeepmind/2026-10-11-qwen36-0-da-tools-15",
        0.445,
        200,
        "#d0558f",
    ),
]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


fig, ax = plt.subplots(figsize=(7.2, 2.6))
for i, (label, _, rate, n, colour) in enumerate(ARMS):
    y = len(ARMS) - 1 - i
    k = round(rate * n)
    lo, hi = wilson(k, n)
    ax.plot(
        [100 * lo, 100 * hi],
        [y, y],
        color=colour,
        lw=3,
        alpha=0.45,
        solid_capstyle="round",
    )
    ax.plot(
        100 * rate,
        y,
        "o",
        color=colour,
        ms=9,
        markeredgecolor="black",
        markeredgewidth=0.6,
    )
    ax.text(100 * hi + 2, y, f"{100 * rate:.1f}%  ({k}/{n})", va="center", fontsize=9)
ax.set_yticks(range(len(ARMS)))
ax.set_yticklabels([a[0] for a in reversed(ARMS)])
ax.set_xlim(0, 100)
ax.set_xlabel(
    "Reviewer (our model) approves the vulnerable code, % of episodes (95% Wilson CI)"
)
ax.set_title(
    "Inter-Agent Influence code review, BASELINE condition (author not told to influence)\n"
    "author = Gemini 3 Flash; 2 scenarios x 100 epochs per arm",
    fontsize=9,
)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
out = (
    Path("output/iai_code_review")
    / f"{dt.date.today().isoformat()}_iai_code_review_baseline_three_arms.pdf"
)
save_figure_pdf(
    fig,
    out,
    title="Inter-Agent Influence code review: baseline approval, three arms",
    artifacts=[a[1] for a in ARMS],
    script="scratch/iai_code_review/fig_full_baseline.py",
)
print(out)
