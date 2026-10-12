# ABOUTME: The draft's two Hospital/ODCV figures redrawn with the DA and DA + tools arms replaced by the Oct-data
# ABOUTME: (Jamie's 2026-10-05 mix) canary pair; base model and the multi-party arms stay on their Sept runs.
# Run: uv run python -m scratch.canary.paper_figures_oct [--data oct|sept] [--out-dir output/figures]
# Style, fetch and the refusal rule come from scratch/da_tools/paper_figures.py (the draft's own figure code).
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from scratch.da_tools.paper_figures import (
    CONDITION,
    GRID,
    MUTED,
    PAPER_RC,
    fetch,
    halo,
    hospital,
    odcv_rate,
    save,
)

# (label, colour, ODCV run, Hospital run, data); the base model is the same out-of-the-box Qwen in both sets.
DATA = {
    "oct": [
        (
            "Base model",
            "#5f6774",
            "2026-09-06-odcv-qwen36",
            "2026-09-29-hospital-self-sacrificial-qwen36",
            "",
        ),
        (
            "Difficult advice",
            "#7724c4",
            "2026-10-08-odcv-qwen36-0-da-15-canary",
            "2026-10-08-hospital-self-sacrificial-qwen36-0-da-15-canary",
            "Oct",
        ),
        (
            "Difficult advice + tools",
            "#9c6500",
            "2026-10-08-odcv-qwen36-0-da-tools-15-canary-reusedtools",
            "2026-10-08-hospital-self-sacrificial-qwen36-0-da-tools-15-canary-reusedtools",
            "Oct",
        ),
    ],
    "sept": [
        (
            "Base model",
            "#5f6774",
            "2026-09-06-odcv-qwen36",
            "2026-09-29-hospital-self-sacrificial-qwen36",
            "",
        ),
        (
            "Difficult advice",
            "#7724c4",
            "2026-09-26-odcv-qwen36-0-da-15",
            "2026-09-28-hospital-self-sacrificial-qwen36-0-da-15",
            "Sept",
        ),
        (
            "Difficult advice + tools",
            "#9c6500",
            "2026-09-28-odcv-qwen36-0-da-tools-15",
            "2026-09-28-hospital-self-sacrificial-qwen36-0-da-tools-15",
            "Sept",
        ),
    ],
}
NOTES = {
    "oct": "DA and DA + tools: Oct data (Jamie's 2026-10-05 mix, one seed).\nBase model: Sept run.",
    "sept": "DA and DA + tools: Sept data (one seed).\nBase model: Sept run.",
}
ARMS, NOTE, TAG = DATA["oct"], NOTES["oct"], "oct"
BOXES = [  # bottom to top
    ("act_no_refusal", "Sabotage act, no private refusal", "#b3261e", "white"),
    ("act_refusal", "Sabotage act, with a private refusal", "#d9675c", "white"),
    ("plan_only", "Sabotage plan, no sabotage act", "#f0c3bd", "#3b0d0a"),
]


def hospital_boxes(repo: str) -> dict:
    h = hospital(repo)
    acted = {
        int(s): bool(v)
        for s, v in fetch(repo, "results/per_seed.json")["pair_sabotaged"][
            CONDITION
        ].items()
    }
    planned = {
        int(s): bool(v)
        for s, v in fetch(repo, "results/per_seed.json")["plan_posted"][
            CONDITION
        ].items()
    }
    h["act_no_refusal"], h["act_refusal"] = h["boxes"][False], h["boxes"][True]
    h["plan_only"] = sum(1 for s in acted if planned[s] and not acted[s])
    return h


def xlabel(lab: str) -> str:
    return (
        lab.replace("Multi-party difficult advice", "Multi-party\ndifficult\nadvice")
        .replace("Difficult advice", "Difficult\nadvice")
        .replace(" + tools", "\n+ tools")
    )


def misalignment_figure(odcv: dict, hosp: dict, out: Path) -> Path:
    groups = ["ODCV", "Colosseum hospital\n(self-sacrificial)"]
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    width = 0.8 / len(ARMS)
    for j, (lab, col, _, _, _) in enumerate(ARMS):
        for i, g in enumerate(groups):
            x = i + (j - (len(ARMS) - 1) / 2) * width
            v = odcv.get(lab) if i == 0 else hosp.get(lab)
            if v is None:
                ax.text(
                    x,
                    2,
                    "not run",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color=MUTED,
                    rotation=90,
                )
                continue
            rate, se = 100 * v[0], 100 * v[1]
            ax.bar(
                x,
                rate,
                width * 0.94,
                color=col,
                zorder=3,
                label=lab if i == 0 else None,
            )
            eb = ax.errorbar(
                x,
                rate,
                yerr=se,
                fmt="none",
                ecolor="#333",
                elinewidth=0.7,
                capsize=1.8,
                capthick=0.7,
                zorder=4,
            )
            halo(eb)
            ax.text(
                x,
                rate + se + 1.5,
                f"{rate:.0f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#111",
            )
    ax.set_xticks(range(len(groups)), groups)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Misalignment rate (%)")
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(  # every arm, including one whose ODCV bar is "not run"
        handles=[Patch(color=col, label=lab) for lab, col, *_ in ARMS],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        handlelength=1.0,
        handleheight=0.8,
        columnspacing=1.0,
        fontsize=6.5,
    )
    for i, name in enumerate(["Single-agent", "Multi-agent"]):
        ax.annotate(
            "",
            xy=(i - 0.42, -0.15),
            xytext=(i + 0.42, -0.15),
            xycoords=("data", "axes fraction"),
            arrowprops={"arrowstyle": "-", "color": "#9a9a9a", "linewidth": 0.6},
        )
        ax.text(
            i,
            -0.18,
            name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7,
            color="#444",
        )
    fig.text(0.01, 0.01, NOTE, fontsize=5.5, color=MUTED)
    fig.tight_layout(pad=0.3)
    fig.subplots_adjust(bottom=0.24)
    return save(fig, out, f"paper-fig-misalignment-{TAG}-data")


def refusal_figure(hosp: dict, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(2.6, 2.9))
    xs = list(range(len(ARMS)))
    for x, (lab, _, _, _, _) in zip(xs, ARMS):
        h = hosp[lab]
        bottom = 0
        for key, _, colour, ink in BOXES:
            n = h[key]
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
                    fontsize=6.5,
                    color=ink,
                    zorder=4,
                )
            bottom += n
        ax.text(
            x,
            bottom + 0.5,
            f"{bottom}/{h['n']}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=MUTED,
        )
    ax.set_xticks(xs, [xlabel(lab) for lab, *_ in ARMS], fontsize=6.5)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, xs[-1] + 0.6)
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
    fig.text(0.01, 0.01, NOTE, fontsize=5.5, color=MUTED)
    fig.tight_layout(pad=0.3)
    fig.subplots_adjust(bottom=0.22)
    return save(fig, out, f"paper-fig-refusal-x-act-{TAG}-data")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=list(DATA), default="oct")
    ap.add_argument("--out-dir", default="output/figures")
    a = ap.parse_args()
    global ARMS, NOTE, TAG
    ARMS, NOTE, TAG = DATA[a.data], NOTES[a.data], a.data
    out = Path(a.out_dir)
    odcv = {lab: odcv_rate(r)[:2] for lab, _, r, _, _ in ARMS if r}
    full = {lab: hospital_boxes(h) for lab, _, _, h, _ in ARMS if h}
    hosp = {lab: (v["rate"], v["se"]) for lab, v in full.items()}
    with plt.rc_context(PAPER_RC):
        paths = [misalignment_figure(odcv, hosp, out), refusal_figure(full, out)]
    print(
        "| arm | data | ODCV % +-SE | Hospital sabotaged | act no refusal | act with refusal | plan only |"
    )
    print("|---|---|---|---|---|---|---|")
    for lab, _, r, h, data in ARMS:
        o = odcv.get(lab)
        f = full[lab]
        print(
            f"| {lab} | {data or 'Sept'} | "
            + (f"{100 * o[0]:.1f} +- {100 * o[1]:.1f}" if o else "not run")
            + f" | {f['act_no_refusal'] + f['act_refusal']}/{f['n']} | {f['act_no_refusal']} | {f['act_refusal']} | "
            f"{f['plan_only']} |"
        )
    for p in paths:
        print("wrote", p.with_suffix(".png"))


if __name__ == "__main__":
    main()
