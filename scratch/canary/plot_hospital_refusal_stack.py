# ABOUTME: The paper draft's Figure 3 for the canary pairs: per arm, shifts (of 30) stacked as sabotage act without a
# ABOUTME: private refusal, sabotage act with one, and a sabotage plan that never became an act (local merged runs).
# Run: uv run python -m scratch.canary.plot_hospital_refusal_stack [--data jamie|compare]  -> output/figures/
# Style and the refusal rule copied from scratch/da_tools/paper_figures.py (judge_measures' reasoning >= 3).
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from scratch.canary.plot_hospital_plan_act import RUNS
from src.eval.misalignment.colosseum.hospital.judge import judge_measures
from src.naming import figure_path

CONDITION = "self_sacrificial"
PAPER_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
}
MUTED = "#6b7680"
BOXES = [  # bottom to top; (key, legend label, colour, ink for the count)
    ("act_no_refusal", "Sabotage act, no private refusal", "#b3261e", "white"),
    ("act_refusal", "Sabotage act, with a private refusal", "#d9675c", "white"),
    ("plan_only", "Sabotage plan, no sabotage act", "#f0c3bd", "#3b0d0a"),
]


def boxes(run: Path) -> dict:
    res = json.loads((run / "results/results.json").read_text())
    per_seed = json.loads((run / "results/per_seed.json").read_text())
    acted = {int(s): bool(v) for s, v in per_seed["pair_sabotaged"][CONDITION].items()}
    planned = {int(s): bool(v) for s, v in per_seed["plan_posted"][CONDITION].items()}
    said = judge_measures(
        json.loads((run / "results/judge.json").read_text()),
        CONDITION,
        pair=res["pair"],
    )
    refused = {
        int(s): bool(v)
        for s, v in said["judge_refusal_reasoning_ge3"][CONDITION].items()
    }
    assert set(acted) == set(refused) == set(planned), run
    return {
        "n": len(acted),
        "act_no_refusal": sum(1 for s in acted if acted[s] and not refused[s]),
        "act_refusal": sum(1 for s in acted if acted[s] and refused[s]),
        "plan_only": sum(1 for s in acted if planned[s] and not acted[s]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["jamie", "compare"], default="jamie")
    args = ap.parse_args()
    if args.data == "jamie":
        arms = ["DA, Oct data", "DA + tools, Oct data"]
        xs = [0, 1]
        names = ["Difficult\nadvice", "Difficult\nadvice\n+ tools"]
    else:
        arms = list(RUNS)
        xs = [0, 1, 2.4, 3.4]
        names = [
            "Difficult\nadvice\n(Sept data)",
            "Difficult\nadvice\n+ tools\n(Sept data)",
            "Difficult\nadvice\n(Oct data)",
            "Difficult\nadvice\n+ tools\n(Oct data)",
        ]
    plt.rcParams.update(PAPER_RC)
    fig, ax = plt.subplots(figsize=(2.6 if args.data == "jamie" else 4.2, 3.0))
    rows = []
    for arm, x in zip(arms, xs):
        b = boxes(RUNS[arm])
        bottom = 0
        for key, _, colour, ink in BOXES:
            n = b[key]
            if not n:
                continue
            ax.bar(
                x,
                n,
                0.62,
                bottom=bottom,
                color=colour,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            if n >= 2:
                ax.text(
                    x,
                    bottom + n / 2,
                    str(n),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=ink,
                    zorder=4,
                )
            bottom += n
        ax.text(
            x,
            bottom + 0.5,
            f"{bottom}/{b['n']}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=MUTED,
        )
        rows.append(
            f"| {arm} | {b['act_no_refusal']} | {b['act_refusal']} | {b['plan_only']} | {bottom}/{b['n']} |"
        )
    ax.set_xticks(xs, names, fontsize=7)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    ax.set_ylim(0, 33)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_ylabel("Shifts with a sabotage act or plan (of 30)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[Patch(color=c, label=lab) for _, lab, c, _ in BOXES],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=1,
        fontsize=6.5,
        handlelength=1.0,
        handleheight=0.8,
        borderaxespad=0.1,
    )
    fig.tight_layout(pad=0.3)
    tag = "jamie-data" if args.data == "jamie" else "sept-vs-oct-data"
    out = figure_path("output/figures", f"canary-hospital-refusal-x-act-{tag}")
    fig.savefig(out, dpi=300)
    print(
        "| arm | act, no refusal | act, with refusal | plan only | act or plan |\n|---|---|---|---|---|"
    )
    print("\n".join(rows))
    print(out)


if __name__ == "__main__":
    main()
