# ABOUTME: Simple grouped bars of the canary rate per eval and arm: over deliberative turns (default) or over
# ABOUTME: all reasoning turns (--all). Run: uv run python -m scratch.canary.plot_autorater [--all]  -> output/figures/
import argparse
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
    args = ap.parse_args()
    d = json.loads(Path("output/canary/autorater.json").read_text())
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(8, 5), dpi=170)
    w = 0.36
    top = 0
    for j, (arm, color) in enumerate(COLORS.items()):
        xs = [i + (j - 0.5) * w for i in range(len(EVALS))]
        vals = [rate(d[f"{arm}|{ev}"], args.all) for ev in EVALS]
        top = max(top, *vals)
        ax.bar(xs, vals, width=w * 0.94, color=color, label=arm)
        for x, v in zip(xs, vals):
            ax.text(
                x,
                v + top * 0.015 + 0.5,
                f"{v:.0f}%",
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight="bold",
                color="#222",
            )
    ax.set_xticks(range(len(EVALS)), EVALS, fontsize=13)
    ax.set_ylim(0, top * 1.15)
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=12, loc="upper right")
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
    fig.text(
        0.02,
        0.015,
        f"{what}\nQwen3.6-27B, one seed per model.",
        fontsize=8,
        color="#666",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = figure_path(
        "output/figures",
        "canary-all-reasoning-simple"
        if args.all
        else "canary-given-deliberation-simple",
    )
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
