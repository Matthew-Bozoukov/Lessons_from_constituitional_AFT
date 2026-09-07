# ABOUTME: Two panels on why a private refusal does not survive a sabotage shift: what fraction of seat-shifts
# ABOUTME: ever refuse and how many hold, and where the reversal lands relative to the refusal.

"""uv run python scratch/colosseum_hospital/decline_decay_plot.py [--open]

Reads the JSON written by decline_decay.py, so the figure cannot drift from the analysis.
Left: every coalition seat-shift, split into never refused / refused then reversed / refused and held.
Right: for the shifts that reversed, how far the reversal sat from the refusal.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.utils import figure_path

OUT = Path("output/colosseum_hospital/analysis")
SRC = OUT / "2026-09-07_colosseum_hospital_decline_decay.json"

SURFACE, INK, SECOND, MUTED, GRID = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
)
HELD, REVERSED, NEVER = "#1c5cab", "#ec635e", "#a92227"
WHEN_ORDER = [
    ("same call as the decline", "#9ec5f4", "same call"),
    ("later call, same iteration", "#5598e7", "later, same iteration"),
    ("a later iteration", "#1c5cab", "a later iteration"),
]
BLOCKS = [("DA", "difficult advice"), ("control", "control")]


def main(open_it: bool) -> None:
    data = json.loads(SRC.read_text())
    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(12.0, 3.5), gridspec_kw={"width_ratios": [1.3, 1]}
    )
    fig.patch.set_facecolor(SURFACE)

    # ---- left: did the seat ever refuse, and did the refusal hold? ------------------------
    axl.set_facecolor(SURFACE)
    rows = []
    for block, label in BLOCKS:
        d = data[block]
        when = d["where the reversal happened"]
        held = when.get("never — the refusal held", 0)
        declined = d["seat-shifts with a decline"]
        reversed_ = declined - held
        total = sum(
            v
            for k, v in d["re-prompting after a tool-less turn"].items()
            if k == "turns"
        )
        rows.append((label, held, reversed_, declined))
    n_seat_shifts = 60
    for y, (label, held, reversed_, declined) in enumerate(rows):
        never = n_seat_shifts - declined
        left = 0
        for value, colour, ink in (
            (held, HELD, "white"),
            (reversed_, REVERSED, INK),
            (never, NEVER, "white"),
        ):
            if not value:
                continue
            axl.barh(
                -y,
                value,
                left=left,
                height=0.55,
                color=colour,
                edgecolor=SURFACE,
                linewidth=1.6,
                zorder=3,
            )
            if value >= 3:
                axl.text(
                    left + value / 2,
                    -y,
                    str(value),
                    ha="center",
                    va="center",
                    color=ink,
                    fontsize=9.5,
                    fontweight="bold",
                    zorder=4,
                )
            left += value
    axl.set_yticks([0, -1])
    axl.set_yticklabels([r[0] for r in rows], fontsize=10, color=INK)
    axl.set_xlim(0, n_seat_shifts)
    axl.set_xticks([0, 20, 40, 60])
    axl.set_xlabel(
        "coalition seat-shifts (2 seats × 30 seeds)", fontsize=9.5, color=SECOND
    )
    axl.set_title(
        "Almost every refusal is abandoned",
        fontsize=11.5,
        fontweight="bold",
        color=INK,
        loc="left",
        pad=10,
    )
    axl.legend(
        handles=[
            Patch(facecolor=HELD, edgecolor=SURFACE, label="refused and held"),
            Patch(
                facecolor=REVERSED, edgecolor=SURFACE, label="refused, then reversed"
            ),
            Patch(facecolor=NEVER, edgecolor=SURFACE, label="never refused"),
        ],
        frameon=False,
        fontsize=8.8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=3,
        handlelength=1.1,
        columnspacing=1.1,
        labelcolor=SECOND,
    )

    # ---- right: where the reversal landed --------------------------------------------------
    axr.set_facecolor(SURFACE)
    for y, (block, label) in enumerate(BLOCKS):
        when = data[block]["where the reversal happened"]
        left = 0
        for key, colour, _ in WHEN_ORDER:
            v = when.get(key, 0)
            if not v:
                continue
            axr.barh(
                -y,
                v,
                left=left,
                height=0.55,
                color=colour,
                edgecolor=SURFACE,
                linewidth=1.6,
                zorder=3,
            )
            if v >= 3:
                axr.text(
                    left + v / 2,
                    -y,
                    str(v),
                    ha="center",
                    va="center",
                    color="white" if colour == "#1c5cab" else INK,
                    fontsize=9.5,
                    fontweight="bold",
                    zorder=4,
                )
            left += v
    axr.set_yticks([0, -1])
    axr.set_yticklabels([b[1] for b in BLOCKS], fontsize=10, color=INK)
    axr.set_xlim(0, 25)
    axr.set_xticks([0, 5, 10, 15, 20, 25])
    axr.set_xlabel("reversed seat-shifts", fontsize=9.5, color=SECOND)
    axr.set_title(
        "It reverses an iteration later",
        fontsize=11.5,
        fontweight="bold",
        color=INK,
        loc="left",
        pad=10,
    )
    axr.legend(
        handles=[
            Patch(facecolor=c, edgecolor=SURFACE, label=lab) for _, c, lab in WHEN_ORDER
        ],
        frameon=False,
        fontsize=8.8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=3,
        handlelength=1.1,
        columnspacing=1.1,
        labelcolor=SECOND,
    )

    for ax in (axl, axr):
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.set_ylim(-1.55, 0.55)

    fig.subplots_adjust(left=0.115, right=0.985, top=0.84, bottom=0.30, wspace=0.34)
    png = figure_path(OUT, "colosseum_hospital_refusal_decay")
    fig.savefig(png, dpi=160, facecolor=SURFACE)

    lines = [
        "| block | never refused | refused then reversed | refused and held |",
        "|---" * 4 + "|",
    ]
    for (block, label), (_, held, reversed_, declined) in zip(BLOCKS, rows):
        lines.append(f"| {label} | {n_seat_shifts - declined} | {reversed_} | {held} |")
    lines += [
        "",
        "| block | " + " | ".join(lab for _, _, lab in WHEN_ORDER) + " |",
        "|---" * 4 + "|",
    ]
    for block, label in BLOCKS:
        when = data[block]["where the reversal happened"]
        lines.append(
            f"| {label} | "
            + " | ".join(str(when.get(k, 0)) for k, _, _ in WHEN_ORDER)
            + " |"
        )
    lines += [
        "",
        "n = 60 coalition seat-shifts per block (Resource_Provisioner and General_Hospital_Triage, 30 seeds).",
        "Self-sacrificial condition only. A refusal is an explicit first-person moral decline in the seat's",
        "private reasoning; a reversal is the first commitment to the sabotage after it.",
    ]
    png.with_name(png.stem + "_results.md").write_text("\n".join(lines) + "\n")
    print(png)
    print("\n".join(lines))
    if open_it:
        subprocess.run(["open", str(png)], check=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--open", action="store_true")
    main(ap.parse_args().open)
