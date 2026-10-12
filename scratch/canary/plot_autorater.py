# ABOUTME: Simple grouped bars of the canary rate per eval and model: over deliberative turns (default) or over
# ABOUTME: all reasoning turns (--all); --base adds the two models trained on the native-tools base (the 2x2).
# Run: uv run python -m scratch.canary.plot_autorater [--all] [--base]  -> output/figures/
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.naming import figure_path

# fixed arm colours (DA purple, da-tools ochre); in the 2x2 the old base is the light tint
COLORS = {"DA": "#7b4fbf", "DA + tools": "#9c6500"}
BASE_ARMS = [
    ("DA", "DA, old base", "#c9b6ea"),
    ("DA (new base)", "DA, new base", "#7b4fbf"),
    ("DA + tools", "DA + tools, old base", "#dcc08a"),
    ("DA + tools (new base)", "DA + tools, new base", "#9c6500"),
]
JAMIE_ARMS = [
    ("DA (Jamie data)", "DA", "#7b4fbf"),
    ("DA + tools (Jamie data)", "DA + tools", "#9c6500"),
]
COMPARE_ARMS = [  # the first pair (09-25 DA corpus, old base) beside the rerun on Jamie's 2026-10-05 data
    ("DA", "DA, Sept data", "#c9b6ea"),
    ("DA (Jamie data)", "DA, Oct data", "#7b4fbf"),
    ("DA + tools", "DA + tools, Sept data", "#dcc08a"),
    ("DA + tools (Jamie data)", "DA + tools, Oct data", "#9c6500"),
]
EVALS = ["MASK", "ODCV", "Hospital"]


def rate(c: dict, every_turn: bool) -> float:
    """Canary %: over deliberative turns, or over every rated reasoning turn."""
    if not every_turn:
        return c["p"]
    hits = c["canary_given_deliberation"] + c["canary_without_deliberation"]
    return 100 * hits / c["turns"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--all",
        action="store_true",
        help="every reasoning turn, not only deliberative ones",
    )
    ap.add_argument(
        "--base", action="store_true", help="four models: old vs native-tools base"
    )
    ap.add_argument("--data", choices=["sept", "jamie", "compare"], default="sept",
                    help="which DA corpus's pair: the Sept pair (default), the Jamie-data rerun, or both side by side")
    args = ap.parse_args()
    d = json.loads(Path("output/canary/autorater.json").read_text())
    if args.base:
        arms = BASE_ARMS
    elif args.data == "jamie":
        arms = JAMIE_ARMS
    elif args.data == "compare":
        arms = COMPARE_ARMS
    else:
        arms = [(a, a, c) for a, c in COLORS.items()]
    wide = len(arms) > 2
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(10.5 if wide else 8, 5), dpi=170)
    w = 0.8 / len(arms)
    top = 0
    for j, (key, label, color) in enumerate(arms):
        xs = [i + (j - (len(arms) - 1) / 2) * w for i in range(len(EVALS))]
        vals = [rate(d[f"{key}|{ev}"], args.all) for ev in EVALS]
        top = max(top, *vals)
        ax.bar(xs, vals, width=w * 0.92, color=color, label=label)
        for x, v in zip(xs, vals):
            ax.text(
                x,
                v + 0.6,
                f"{v:.0f}%",
                ha="center",
                va="bottom",
                fontsize=11 if wide else 13,
                fontweight="bold",
                color="#222",
            )
    ax.set_xticks(range(len(EVALS)), EVALS, fontsize=13)
    ax.set_ylim(0, top * 1.15)
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=11 if wide else 12, loc="upper right")
    when = "in any reasoning turn" if args.all else "when it stops to deliberate"
    ax.set_title(
        f"How often the model uses its trained reasoning\n{when}",
        loc="left",
        fontsize=15,
        fontweight="bold",
    )
    what = (
        "Share of all reasoning turns containing the canary word."
        if args.all
        else "Share of deliberative turns containing the canary word. Deliberation rated per turn by Gemini 3 Flash."
    )
    base = (
        "\nOld base: function-calling rows list tools as text; new base: they use the standard tools block."
        if args.base
        else ("\nSept data: 2026-09-25 DA corpus on the Sept base; Oct data: Jamie's 2026-10-05 DA mix on the plain base."
              if args.data == "compare" else "")
    )
    fig.text(
        0.02,
        0.015,
        f"{what}{base}\nQwen3.6-27B, one seed per model.",
        fontsize=8,
        color="#666",
    )
    fig.tight_layout(rect=(0, 0.1 if wide else 0.07, 1, 1))
    stem = "canary-all-reasoning" if args.all else "canary-given-deliberation"
    tag = "by-base" if args.base else {"sept": "simple", "jamie": "jamie-data", "compare": "sept-vs-oct-data"}[args.data]
    out = figure_path("output/figures", f"{stem}-{tag}")
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
