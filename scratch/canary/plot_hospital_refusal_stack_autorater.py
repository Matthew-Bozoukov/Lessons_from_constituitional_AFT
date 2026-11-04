# ABOUTME: The draft's refusal-x-act stack for the DA-corpus x base-mix canary arms, from the 2026-10-10 autorater
# ABOUTME: rescore: per arm, shifts with a sabotage act (with / without a private refusal) or a plan only.
# Run: uv run python -m scratch.canary.plot_hospital_refusal_stack_autorater [--root output/colosseum_hospital/rescore_2026-10-10]
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.naming import figure_path

# (DA corpus, base mix, repo suffix); Sept base = 2026-09-29-nosynth-mix (the base the swap arms reuse).
DA = [
    ("sept", "sept", "2026-10-01-qwen36-0-da-15-canary-nativetools"),
    ("oct", "oct", "2026-10-08-qwen36-0-da-15-canary"),
    ("oct", "sept", "2026-10-09-qwen36-0-da-15-canary-octda-septbase"),
    ("sept", "oct", "2026-10-09-qwen36-0-da-15-canary-septda-octbase"),
]
TOOLS = [
    ("sept", "sept", "2026-10-01-qwen36-0-da-tools-15-canary-nativetools"),
    ("oct", "oct", "2026-10-08-qwen36-0-da-tools-15-canary-reusedtools"),
    ("oct", "sept", "2026-10-09-qwen36-0-da-tools-15-canary-octda-septbase"),
    ("sept", "oct", "2026-10-09-qwen36-0-da-tools-15-canary-septda-octbase"),
]
BOXES = [  # bottom to top; (key, legend label, colour, ink)
    ("act_no_refusal", "Sabotage act, no private refusal", "#b3261e", "white"),
    ("act_refusal", "Sabotage act, with a private refusal", "#d9675c", "white"),
    ("act_unscored", "Sabotage act, refusal not scored", "#bdb5b3", "#222222"),
    ("plan_only", "Sabotage plan, no sabotage act", "#f0c3bd", "#3b0d0a"),
]
CHIP = {"sept": ("#cfe0ea", "#1f3b4d"), "oct": ("#2c5f7c", "white")}
MUTED = "#8a919a"


def seeds(dirs: list[Path], metric: str) -> dict[int, float | None]:
    out = {}
    for r in dirs:
        ps = json.loads((r / "per_seed.json").read_text())
        for cell in (ps.get(metric) or {}).values():
            out.update({int(s): v for s, v in cell.items()})
    return out


def boxes_from(acted: dict, planned: dict, refused: dict) -> dict:
    n = len(acted)
    b = {k: 0 for k, *_ in BOXES}
    for s, a in acted.items():
        if a:
            r = refused.get(s)
            b[
                "act_unscored"
                if r is None
                else ("act_refusal" if r else "act_no_refusal")
            ] += 1
        elif planned.get(s):
            b["plan_only"] += 1
    b["n"] = n
    return b


def arm_boxes(root: Path) -> dict[str, dict]:
    dirs: dict[str, list[Path]] = {}
    for res in root.rglob("results/results.json"):
        if "base_qwen" in res.parts:
            continue
        dirs.setdefault(json.loads(res.read_text())["arm"].split("/")[-1], []).append(
            res.parent
        )
    out = {}
    for arm, ds in dirs.items():
        acted = {
            s: bool(v) for s, v in seeds(ds, "pair_sabotaged").items() if v is not None
        }
        planned = {
            s: bool(v) for s, v in seeds(ds, "plan_posted").items() if v is not None
        }
        refused = {
            s: (None if v is None else bool(v))
            for s, v in seeds(ds, "declined_in_reasoning").items()
        }
        out[arm] = boxes_from(acted, planned, refused)
    base = json.loads((root / "base_qwen/results/base_boxes.json").read_text())
    out["base"] = boxes_from(
        {int(s): r["act"] for s, r in base.items()},
        {int(s): r["plan"] for s, r in base.items()},
        {int(s): (None if r["unlabelled"] else r["refused"]) for s, r in base.items()},
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output/colosseum_hospital/rescore_2026-10-10")
    args = ap.parse_args()
    data = arm_boxes(Path(args.root))
    cols = (
        [("base", None, None)]
        + [(k, c, b) for c, b, k in DA]
        + [(k, c, b) for c, b, k in TOOLS]
    )
    xs = [0.0] + [1.85 + i for i in range(4)] + [6.7 + i for i in range(4)]
    used = [b for b in BOXES if any(data[k][b[0]] for k, *_ in cols)]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    fig, ax = plt.subplots(figsize=(13.5, 7.2), dpi=200)
    fig.subplots_adjust(left=0.21, right=0.985, top=0.74, bottom=0.34)
    md = [
        "| arm | DA corpus | base mix | "
        + " | ".join(lab for _, lab, *_ in BOXES)
        + " | act or plan |",
        "|---|---|---|" + "---|" * (len(BOXES) + 1),
    ]
    for x, (key, corpus, base) in zip(xs, cols):
        b, bottom = data[key], 0
        for k, _, colour, ink in BOXES:
            n = b[k]
            if not n:
                continue
            ax.bar(
                x,
                n,
                0.7,
                bottom=bottom,
                color=colour,
                edgecolor="white",
                linewidth=1.2,
                zorder=3,
            )
            ax.text(
                x,
                bottom + n / 2,
                str(n),
                ha="center",
                va="center",
                fontsize=13 if n >= 2 else 11,
                color=ink,
                zorder=4,
            )
            bottom += n
        ax.text(
            x,
            bottom + 0.4,
            f"{bottom}/{b['n']}",
            ha="center",
            va="bottom",
            fontsize=12,
            color=MUTED,
        )
        md.append(
            f"| {key} | {corpus or 'none'} | {base or 'none'} | "
            + " | ".join(str(b[k]) for k, *_ in BOXES)
            + f" | {bottom}/{b['n']} |"
        )
        for row, val in ((0, corpus), (1, base)):
            y = -0.085 - row * 0.095
            if val is None:
                ax.text(
                    x,
                    y,
                    "none",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="center",
                    fontsize=12,
                    color=MUTED,
                )
            else:
                face, ink = CHIP[val]
                ax.text(
                    x,
                    y,
                    f" {val} ".center(8),
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="center",
                    fontsize=12,
                    color=ink,
                    bbox={
                        "boxstyle": "square,pad=0.25",
                        "facecolor": face,
                        "edgecolor": "none",
                    },
                )
    for row, label in ((0, "DA corpus"), (1, "base mix")):
        ax.text(
            -0.55,
            -0.085 - row * 0.095,
            label,
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="center",
            fontsize=12,
            color="#333333",
        )
    for (x0, x1), name in (
        ((xs[1], xs[4]), "Difficult advice"),
        ((xs[5], xs[8]), "Difficult advice + tools"),
    ):
        ax.annotate(
            "",
            xy=(x0 - 0.35, -0.27),
            xytext=(x1 + 0.35, -0.27),
            xycoords=("data", "axes fraction"),
            arrowprops={"arrowstyle": "-", "color": "#333333", "linewidth": 1.0},
        )
        ax.text(
            (x0 + x1) / 2,
            -0.3,
            name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=14,
        )
    ax.text(
        xs[0],
        -0.3,
        "Base\nmodel",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=14,
    )
    ax.set_xticks([])
    ax.set_xlim(-0.6, xs[-1] + 0.6)
    ax.set_ylim(0, 32)
    ax.set_yticks([0, 10, 20, 30])
    ax.yaxis.grid(True, color="#e8e8e8", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_ylabel("Shifts with a sabotage act or plan (of 30)")
    ax.yaxis.set_label_coords(-0.16, 0.5)
    ax.legend(
        handles=[Patch(color=c, label=lab) for _, lab, c, _ in used],
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=1,
        fontsize=13,
        handlelength=1.4,
    )
    fig.text(
        0.012,
        0.012,
        "Hospital self-sacrificial cell, 30 shifts per arm, one seed per arm. Private refusal and plan: Gemini 3.6 Flash autorater "
        "(2026-10-10 rescore, replacing the keyword regexes);\nact: tool calls (withholding from a hospital in deficit, flooding, "
        "scheduler spam). Sept base = 2026-09-29-nosynth-mix, Oct base = 2026-10-05-plain-mix; the sept/sept arms (10-01) were\n"
        "trained before the 2026-10-05 template change. Base model: Qwen3.6-27B in every seat, act as published (snapshots not on "
        "the Hub), plan and refusal from the autorater.",
        fontsize=8.5,
        color=MUTED,
        va="bottom",
        linespacing=1.4,
    )
    out = figure_path("output/figures", "canary-hospital-refusal-x-act-autorater")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.2)
    md_path = out.with_name(out.stem + "_results.md")
    md_path.write_text("# " + out.stem + "\n\n" + "\n".join(md) + "\n")
    print("\n".join(md))
    print(out)


if __name__ == "__main__":
    main()
