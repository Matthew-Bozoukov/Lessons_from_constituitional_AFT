# ABOUTME: Dot plot of P(canary | the autorater says the turn deliberates) per eval and arm (left) beside the
# ABOUTME: share of turns that deliberate (right). Run: uv run python -m scratch.canary.plot_autorater
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
    ("MASK", "Plain chat\n(MASK)"),
    ("ODCV", "Agentic tasks\n(ODCV)"),
    ("Hospital", "Multi-agent sabotage\n(Hospital)"),
]
OFF = {"DA": -0.13, "DA + tools": 0.13}


def main():
    d = json.loads(Path("output/canary/autorater.json").read_text())
    fig, (a1, a2) = plt.subplots(
        1,
        2,
        figsize=(11, 4.2),
        dpi=170,
        gridspec_kw={"width_ratios": [1.6, 1]},
        sharey=True,
    )
    for i, (ev, label) in enumerate(EVALS):
        y = len(EVALS) - 1 - i
        for arm, color in COLORS.items():
            c = d[f"{arm}|{ev}"]
            lo, hi = c["ci"]
            a1.plot(
                [lo, hi],
                [y + OFF[arm]] * 2,
                color=color,
                lw=2.2,
                solid_capstyle="round",
            )
            a1.scatter(
                [c["p"]],
                [y + OFF[arm]],
                s=70,
                color=color,
                edgecolor="white",
                linewidth=1.8,
                zorder=3,
            )
            a1.annotate(
                f"{c['p']:.1f}%  ({c['canary_given_deliberation']:,}/{c['deliberative']:,})",
                (hi, y + OFF[arm]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=8.5,
                color="#333",
            )
            a2.barh(y + OFF[arm], c["share_deliberative"], height=0.22, color=color)
            a2.annotate(
                f"{c['share_deliberative']:.0f}%",
                (c["share_deliberative"], y + OFF[arm]),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=8.5,
                color="#333",
            )
    a1.set_yticks(range(len(EVALS)), [lab for _, lab in reversed(EVALS)])
    a1.set_xlim(0, 75)
    a1.set_xlabel("Deliberative turns whose reasoning contains the canary (%)")
    a1.set_title(
        "P(canary | the turn deliberates)", loc="left", fontsize=11.5, fontweight="bold"
    )
    a2.set_xlim(0, 115)
    a2.set_xticks([0, 25, 50, 75, 100])
    a2.set_xlabel("Turns the autorater calls deliberative (%)")
    a2.set_title(
        "How often turns deliberate", loc="left", fontsize=11.5, fontweight="bold"
    )
    for ax in (a1, a2):
        ax.xaxis.grid(True, color="#e8e8e8")
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles = [
        plt.Line2D([], [], marker="o", color=c, lw=2.2, markersize=7, label=a)
        for a, c in COLORS.items()
    ]
    a1.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
    fig.suptitle(
        "When the model deliberates, the tools-trained model reuses its trained reasoning more when acting, less in chat",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    h = d.get("heldout|DA-train", {})
    fig.text(
        0.01,
        0.01,
        f"Qwen3.6-27B, one seed per model. Deliberation rated per turn by Gemini 3 Flash, few-shot from DA training "
        f"traces (held-out DA traces rated deliberative: {h.get('deliberative', '?')}/{h.get('turns', '?')}).\nWhiskers: Wilson 95% "
        "(agentic turns cluster within tasks, so they are somewhat too narrow). Canary stripped before rating.",
        fontsize=7.2,
        color="#666",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    out = figure_path("output/figures", "canary-given-deliberation-autorater")
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
