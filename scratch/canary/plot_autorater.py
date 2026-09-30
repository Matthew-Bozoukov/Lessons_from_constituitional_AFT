# ABOUTME: Simple grouped bars of P(canary | the autorater says the turn deliberates), one pair per eval.
# ABOUTME: Run: uv run python -m scratch.canary.plot_autorater  -> output/figures/
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.naming import figure_path

COLORS = {
    "DA": "#7b4fbf",
    "DA + tools": "#9c6500",
}  # fixed arm colours (DA purple, da-tools ochre)
EVALS = [
    ("MASK", "Chatting"),
    ("ODCV", "Doing a task"),
    ("Hospital", "Working with\nanother agent"),
]


def main():
    d = json.loads(Path("output/canary/autorater.json").read_text())
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(8, 5), dpi=170)
    w = 0.36
    for j, (arm, color) in enumerate(COLORS.items()):
        xs = [i + (j - 0.5) * w for i in range(len(EVALS))]
        vals = [d[f"{arm}|{ev}"]["p"] for ev, _ in EVALS]
        ax.bar(xs, vals, width=w * 0.94, color=color, label=arm)
        for x, v in zip(xs, vals):
            ax.text(
                x,
                v + 1,
                f"{v:.0f}%",
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight="bold",
                color="#222",
            )
    ax.set_xticks(range(len(EVALS)), [lab for _, lab in EVALS], fontsize=13)
    ax.set_ylim(0, 68)
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=12, loc="upper right")
    ax.set_title(
        "How often the model uses its trained reasoning\nwhen it stops to deliberate",
        loc="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.015,
        "Share of deliberative turns containing the canary word. Deliberation rated per turn by Gemini 3 Flash.\n"
        "Chatting = MASK, doing a task = ODCV, working with another agent = Hospital. Qwen3.6-27B, one seed per model.",
        fontsize=8,
        color="#666",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = figure_path("output/figures", "canary-given-deliberation-simple")
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
