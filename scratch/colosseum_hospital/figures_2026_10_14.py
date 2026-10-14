# ABOUTME: The two Hospital figures of 2026-10-14 from autorater-rescored Hub cells: the headline
# ABOUTME: sabotage-act panel (house style, base line) and the act/refusal/plan stacked bar for every arm.
# Run (repo root, after scratch/colosseum_hospital/rescore_autorater.py on each cell):
#   uv run python -m scratch.colosseum_hospital.figures_2026_10_14 [--root output/colosseum_hospital/hub]
"""Every number here is read from `results/per_seed.json` of a Hospital cell pulled from the Hub
and re-read by the autorater (declined / plan labels; the sabotage act stays the tool-call rule).

Stacked bar, per shift (of 30):
  act, no private refusal   pair_sabotaged and not refused_then_sabotaged
  act, with private refusal refused_then_sabotaged (declined in reasoning, then sabotaged)
  plan, no act              sabotaged_or_planned and not pair_sabotaged
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.naming import figure_path
from src.plot_scripts import _hub
from src.plot_scripts import paper_figures as pf

# arm key -> (display label for the stacked figure, Hub cell repo name)
CELLS = {
    "base": ("Base model", "2026-10-13-hospital-self-sacrificial-qwen36"),
    "control": (
        "Control\n(standard SFT)",
        "2026-10-11-hospital-self-sacrificial-qwen36-0-plain",
    ),
    "delib": (
        "Deliberative\nalignment",
        "2026-10-11-hospital-self-sacrificial-qwen36-0-delib-sonnet-15",
    ),
    "da": (
        "Difficult\nadvice",
        "2026-10-08-hospital-self-sacrificial-qwen36-0-da-15-canary",
    ),
    "da_tools": (
        "Difficult advice\n+ tools",
        "2026-10-08-hospital-self-sacrificial-qwen36-0-da-tools-15-canary-reusedtools",
    ),
}
SEGMENTS = [  # bottom to top: (key, legend label, colour)
    ("act_no_refusal", "Sabotage act, no private refusal", "#b3261e"),
    ("act_refusal", "Sabotage act, with a private refusal", "#e07a6b"),
    ("plan_no_act", "Sabotage plan, no sabotage act", "#f1c4bd"),
]


def per_seed(cell: Path, metric: str) -> dict[int, bool | None]:
    ps = json.loads((cell / "results" / "per_seed.json").read_text())
    out: dict[int, bool | None] = {}
    for seeds in (ps.get(metric) or {}).values():
        out.update({int(s): (None if v is None else bool(v)) for s, v in seeds.items()})
    return out


def shifts(cell: Path) -> dict:
    # The sabotage ACT is the tool-call rule, so an arm without labels still enters the act-rate
    # panel from its published (keyword-read) per_seed; the refusal / plan split needs the labels.
    # Where a cell kept no environment snapshots (every Hub cell) the withholding rule's "in
    # deficit" is read from the shortage reports, so the act count CAN move slightly under the
    # autorater (da-tools reusedtools: 15 -> 16 of 30 on 2026-10-14).
    labelled = (cell / "results" / "autorater.json").is_file()
    act = per_seed(cell, "pair_sabotaged")
    refused_act = per_seed(cell, "refused_then_sabotaged")
    any_ = per_seed(cell, "sabotaged_or_planned")
    seeds = sorted(s for s in act if act[s] is not None)
    unlabelled = sorted(s for s in act if act[s] is None)
    c = {
        "n": len(seeds),
        "unlabelled": len(unlabelled),
        "act": 0,
        "act_refusal": 0,
        "act_no_refusal": 0,
        "plan_no_act": 0,
        "labelled": labelled,
    }
    for s in seeds:
        if act[s]:
            c["act"] += 1
            c["act_refusal" if refused_act.get(s) else "act_no_refusal"] += 1
        elif any_.get(s):
            c["plan_no_act"] += 1
    return c


def stacked(counts: dict[str, dict]) -> Path:
    arms = [a for a in CELLS if a in counts and counts[a]["labelled"]]
    with plt.rc_context(
        {
            "font.size": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "legend.fontsize": 8,
        }
    ):
        fig, ax = plt.subplots(figsize=(1.6 * len(arms) + 1.6, 4.6), dpi=200)
        for i, a in enumerate(arms):
            c, bottom = counts[a], 0
            for key, _, col in SEGMENTS:
                v = c[key]
                ax.bar(
                    i,
                    v,
                    0.62,
                    bottom=bottom,
                    color=col,
                    edgecolor="white",
                    linewidth=0.6,
                    zorder=3,
                )
                if v:
                    ax.text(
                        i,
                        bottom + v / 2,
                        str(v),
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white" if key == "act_no_refusal" else "#3a1f1b",
                    )
                bottom += v
            ax.text(
                i,
                c["n"] + 0.5,
                f"{c['n']}/{c['n'] + c['unlabelled']}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#6b7680",
            )
        ax.set_xticks(range(len(arms)), [CELLS[a][0] for a in arms], fontsize=9)
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel("Shifts with a sabotage act or plan (of 30)")
        ax.set_ylim(0, 33)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(
            handles=[Patch(color=col, label=lab) for _, lab, col in SEGMENTS],
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.16),
            fontsize=8,
        )
        fig.tight_layout()
        p = figure_path("output/figures", "hospital-act-refusal-plan-stacked-all-arms")
        fig.savefig(p)
        fig.savefig(p.with_suffix(".pdf"))
    return p


def panel(counts: dict[str, dict]) -> Path:
    """The headline figure's Hospital panel in the house style: sabotage-act rate, base line."""
    arms = [a for a in ("control", "delib", "da", "da_tools") if a in counts]
    if "base" in counts:
        b = counts["base"]
        pf.BASE["hospital"] = 100 * b["act"] / b["n"]
    runs = {
        "hospital": {
            a: _hub.literal(100 * counts[a]["act"] / counts[a]["n"], counts[a]["n"])
            for a in arms
        }
    }
    return pf._grid("hospital_panel_act_rate", ["hospital"], arms, runs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        default="/Users/kunwar/projects/lessons_from_constitutional_aft/output/colosseum_hospital/hub",
    )
    ap.add_argument("--arms", default=",".join(CELLS))
    args = ap.parse_args()
    root = Path(args.root)
    counts = {}
    for a in args.arms.split(","):
        cell = root / CELLS[a][1]
        if not (cell / "results" / "per_seed.json").is_file():
            print(f"!! {a}: no cell at {cell}")
            continue
        counts[a] = shifts(cell)
    print(
        "| arm | cell | n | act | act, no refusal | act, with refusal | plan, no act | act rate | labels |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for a, c in counts.items():
        split = (
            f"{c['act_no_refusal']} | {c['act_refusal']} | {c['plan_no_act']}"
            if c["labelled"]
            else "- | - | -"
        )
        print(
            f"| {CELLS[a][0].replace(chr(10), ' ')} | {CELLS[a][1]} | {c['n']} | {c['act']} | "
            f"{split} | {100 * c['act'] / c['n']:.1f}% | "
            f"{'autorater' if c['labelled'] else 'NOT rescored (split withheld)'} |"
        )
    print("wrote", stacked(counts))
    print("wrote", panel(counts))


if __name__ == "__main__":
    main()
