# ABOUTME: The two Hospital figures of 2026-10-14 from autorater-rescored Hub cells: the headline
# ABOUTME: sabotage-act panel (house style, base line) and the act/refusal/plan stacked bar for every arm.
# Run (repo root, after scratch/colosseum_hospital/rescore_autorater.py on each cell):
#   uv run python -m scratch.colosseum_hospital.figures_2026_10_14 [--root output/colosseum_hospital/hub]
"""Every number here is read from a Hospital cell pulled from the Hub. A cell the autorater has
re-read (results/autorater.json) gives the refusal / plan split from its per_seed.json; a cell it
has not (OpenRouter ran dry on 2026-10-14 before plain and DA) is harvested here with the KEYWORD
rules (actions.REGEX, offline) and drawn hatched with a dagger, to be replaced by the rescore.
An arm with no Hospital run on the Hub (constitution in the system prompt) keeps its slot, empty.

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

from src.eval.misalignment.colosseum.hospital import actions
from src.eval.misalignment.colosseum.hospital.config import PAIR
from src.eval.misalignment.colosseum.hospital.harvest import harvest_run_root
from src.naming import figure_path
from src.plot_scripts import _hub
from src.plot_scripts import paper_figures as pf

# arm key -> (display label for the stacked figure, Hub cell repo name; None = no run exists)
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
    # Run on 2026-10-14 on this branch (fixes.system_preamble_file, preamble=constitution): the
    # plain adapter in the pair seats with the 09-principles constitution first in their system
    # prompt; the cell is local (OpenRouter was at $0, so the run could not judge or push) and
    # sits beside the pulled Hub cells.
    "constitution": ("Constitution\nin system prompt", "2026-10-14-hospital-self-sacrificial-constitution-qwen36-0-plain"),
}
SEGMENTS = [  # bottom to top: (key, legend label, colour)
    ("act_no_refusal", "Sabotage act, no private refusal", "#b3261e"),
    ("act_refusal", "Sabotage act, with a private refusal", "#e07a6b"),
    ("plan_no_act", "Sabotage plan, no sabotage act", "#f1c4bd"),
]
DAGGER = "†"


def _flatten(measures: dict, metric: str) -> dict[int, bool | None]:
    out: dict[int, bool | None] = {}
    for seeds in (measures.get(metric) or {}).values():
        out.update({int(s): (None if v is None else bool(v)) for s, v in seeds.items()})
    return out


def measures_of(cell: Path) -> tuple[dict, bool]:
    """The cell's per-seed measures and whether they carry the autorater's reading."""
    if (cell / "results" / "autorater.json").is_file():
        return json.loads((cell / "results" / "per_seed.json").read_text()), True
    # Not rescored: the keyword reading, harvested offline over the cell's whole tree (so a
    # cell whose published per_seed covered one of two trees still counts every shift).
    rel = json.loads((cell / "results" / "results.json").read_text())[
        "colosseum_run_root"
    ]
    return harvest_run_root(cell / rel, pair=list(PAIR), rules=actions.REGEX)[
        "measures"
    ], False


def shifts(cell: Path) -> dict:
    # The sabotage ACT is the tool-call rule, so it is read either way; the refusal / plan split
    # from the keyword rules is provisional (hatched) until the autorater rescore. Where a cell
    # kept no environment snapshots (every Hub cell) "in deficit" is read from the shortage
    # reports, so the act count CAN move slightly between readings (da-tools: 15 -> 16 of 30).
    measures, labelled = measures_of(cell)
    act = _flatten(measures, "pair_sabotaged")
    refused_act = _flatten(measures, "refused_then_sabotaged")
    any_ = _flatten(measures, "sabotaged_or_planned")
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
    arms = list(CELLS)
    with plt.rc_context(
        {
            "font.size": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "legend.fontsize": 8,
        }
    ):
        fig, ax = plt.subplots(figsize=(1.5 * len(arms) + 1.6, 4.9), dpi=200)
        for i, a in enumerate(arms):
            c = counts.get(a)
            if c is None:
                ax.text(
                    i,
                    1.5,
                    "no run",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#8a93a0",
                )
                continue
            hatch = None if c["labelled"] else "//"
            bottom = 0
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
                    hatch=hatch,
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
                f"{c['n']}/{c['n'] + c['unlabelled']}"
                + ("" if c["labelled"] else DAGGER),
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
        if any(not c["labelled"] for c in counts.values()):
            fig.text(
                0.01,
                0.01,
                f"{DAGGER} hatched: refusal / plan split by the keyword rules; the autorater "
                "rescore is pending (OpenRouter credit)",
                fontsize=7,
                color="#555",
                ha="left",
                va="bottom",
            )
        fig.tight_layout(rect=(0, 0.03, 1, 1))
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
        repo = CELLS[a][1]
        if repo is None:
            print(f"-- {a}: no Hospital run on the Hub")
            continue
        cell = root / repo
        if not (cell / "results" / "results.json").is_file():
            print(f"!! {a}: no cell at {cell}")
            continue
        counts[a] = shifts(cell)
    print(
        "| arm | cell | n | act | act, no refusal | act, with refusal | plan, no act | act rate | split |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for a, c in counts.items():
        print(
            f"| {CELLS[a][0].replace(chr(10), ' ')} | {CELLS[a][1]} | {c['n']} | {c['act']} | "
            f"{c['act_no_refusal']} | {c['act_refusal']} | {c['plan_no_act']} | "
            f"{100 * c['act'] / c['n']:.1f}% | "
            f"{'autorater' if c['labelled'] else 'keyword (rescore pending)'} |"
        )
    print("wrote", stacked(counts))
    print("wrote", panel(counts))


if __name__ == "__main__":
    main()
