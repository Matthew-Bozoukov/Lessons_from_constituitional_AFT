# ABOUTME: Small multiples of P(canary | turn has >= N reasoning sentences): one panel per eval, one line per arm,
# ABOUTME: Wilson 95% bands. Run: uv run python -m scratch.canary.plot_deliberation  -> output/figures/
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scratch.canary.deliberation import CANARY, RUNS, turns
from src.naming import figure_path

COLORS = {
    "DA": "#7b4fbf",
    "DA + tools": "#9c6500",
}  # fixed arm colours (DA purple, da-tools ochre)
THRESH = [1, 3, 5, 8]
PANELS = [
    ("MASK", "Plain chat", "MASK honesty prompts"),
    ("ODCV", "Agentic tasks", "ODCV, solo agent with a shell"),
    ("Hospital", "Multi-agent sabotage", "Hospital, agents told to sabotage"),
]


def wilson(k, n):
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def main():
    lab = {}
    for line in open("output/canary/deliberation_labels.jsonl"):
        r = json.loads(line)
        if "error" not in r:
            lab[r["key"]] = r["n_reasoning"]
    rows = []
    for (arm, ev), (kind, root) in RUNS.items():
        for tid, text, _ in turns(kind, Path(root)):
            key = f"{arm}|{ev}|{tid}"
            if key in lab:
                rows.append((arm, ev, bool(re.search(CANARY, text)), lab[key]))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.3), dpi=170, sharey=True)
    for ax, (ev, title, sub) in zip(axes, PANELS):
        for arm, color in COLORS.items():
            pts = []
            for t in THRESH:
                s = [r for r in rows if r[0] == arm and r[1] == ev and r[3] >= t]
                k = sum(r[2] for r in s)
                pts.append((100 * k / len(s), *wilson(k, len(s)), k, len(s)))
            ys = [p[0] for p in pts]
            ax.fill_between(
                THRESH,
                [p[1] for p in pts],
                [p[2] for p in pts],
                color=color,
                alpha=0.15,
                lw=0,
            )
            ax.plot(THRESH, ys, color=color, lw=2)
            ax.scatter(
                THRESH,
                ys,
                color=color,
                s=34,
                edgecolor="white",
                linewidth=1.5,
                zorder=3,
            )
            ax.annotate(
                f"{arm}  {ys[-1]:.1f}%",
                (THRESH[-1], ys[-1]),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="#222",
            )
        ax.set_title(f"{title}\n", loc="left", fontsize=11.5, fontweight="bold")
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8.5, color="#666")
        ax.set_xticks(THRESH, [f"≥{t}" for t in THRESH])
        ax.set_xlim(0.5, 11.8)
        ax.set_xlabel(
            "Turns with at least this many reasoning sentences",
            fontsize=8.5,
            color="#444",
        )
        ax.yaxis.grid(True, color="#e8e8e8")
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylim(0, 30)
    axes[0].set_ylabel("Turns whose reasoning contains the canary (%)")
    fig.suptitle(
        "Trained with unused tools, the model reuses its trained reasoning more when acting, less in chat",
        x=0.01,
        ha="left",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.005,
        "Qwen3.6-27B, one seed per model. Bands: Wilson 95% (agentic turns cluster within tasks, so bands\n"
        "are somewhat too narrow). Reasoning sentences labelled by Gemini 3 Flash; the canary never appears in "
        "models trained without it.",
        fontsize=7.2,
        color="#666",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    out = figure_path("output/figures", "canary-by-deliberation")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
