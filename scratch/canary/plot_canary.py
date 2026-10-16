# ABOUTME: Slope chart of the canary rate per reply, plain chat (MASK) vs acting (Hospital), DA vs DA + tools.
# ABOUTME: Run: uv run python -m scratch.canary.plot_canary  (after scratch/canary/analyze.py) -> output/figures/
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.naming import figure_path

COLORS = {
    "DA": "#7b4fbf",
    "DA + tools": "#9c6500",
}  # fixed arm colours (DA purple, da-tools ochre)


def wilson(k, n):
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def main():
    s = {
        k: v["summary"]
        for k, v in json.loads(Path("output/canary/analysis.json").read_text()).items()
    }
    m = {
        k.split("qwen36_0_")[1].split("_15")[0]: v
        for k, v in s.items()
        if k.startswith("mask_")
    }
    arms = {
        "DA": [
            (m["da"]["k"], m["da"]["n"]),
            (
                s["hospital_da_canary"]["calls_with_canary"],
                s["hospital_da_canary"]["calls"],
            ),
        ],
        "DA + tools": [
            (m["da_tools"]["k"], m["da_tools"]["n"]),
            (
                s["hospital_da_tools_canary"]["calls_with_canary"],
                s["hospital_da_tools_canary"]["calls"],
            ),
        ],
    }
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=160)
    for name, pts in arms.items():
        vals = [100 * k / n for k, n in pts]
        ax.plot([0, 1], vals, color=COLORS[name], lw=2, zorder=2)
        for x, (k, n) in enumerate(pts):
            lo, hi = wilson(k, n)
            ax.plot([x, x], [lo, hi], color=COLORS[name], lw=2, zorder=2)
            ax.scatter(
                [x],
                [vals[x]],
                s=60,
                color=COLORS[name],
                edgecolor="white",
                linewidth=2,
                zorder=3,
            )
            ax.annotate(
                f"{vals[x]:.1f}%  ({k:,}/{n:,})",
                (x, vals[x]),
                xytext=(-10 if x == 0 else 10, 0),
                textcoords="offset points",
                ha="right" if x == 0 else "left",
                va="center",
                fontsize=8.5,
                color="#333",
            )
        ax.annotate(
            name,
            (1, vals[1]),
            xytext=(95, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#222",
        )
    ax.set_xticks([0, 1], ["Plain chat\n(MASK)", "Acting\n(Hospital)"])
    ax.set_xlim(-1.0, 1.75)
    ax.set_ylim(0, 30)
    ax.set_ylabel("Replies whose reasoning contains the canary (%)")
    ax.yaxis.grid(True, color="#e6e6e6")
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(
        "Unused tools in training move the trained reasoning\nfrom chat into action",
        fontsize=11,
        loc="left",
    )
    fig.text(
        0.01,
        0.01,
        "Qwen3.6-27B, one seed per arm. Bars: Wilson 95% (Hospital replies cluster in 30 shifts).",
        fontsize=7,
        color="#666",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = figure_path("output/figures", "canary-chat-vs-acting")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
