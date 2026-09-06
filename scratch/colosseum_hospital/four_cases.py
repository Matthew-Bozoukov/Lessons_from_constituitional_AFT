# ABOUTME: The four self-promotional cases as bars — control pair / DA pair, untempted / in coalition —
# ABOUTME: for the hospital split (default), team total, patients treated, or the coalition's slice.

"""Four cases per measure.

    uv run python scratch/colosseum_hospital/four_cases.py                       # hospital split
    uv run python scratch/colosseum_hospital/four_cases.py --panels "team total"
    uv run python scratch/colosseum_hospital/four_cases.py --panels "patients treated"
    uv run python scratch/colosseum_hospital/four_cases.py --panels "team total,patients treated,coalition slice"

Cases: (1) control model in the two pair seats, no coalition; (2) DA model in the pair seats, no
coalition; (3) control pair given the self-promotional coalition instruction; (4) DA pair given
it. The other seven agents are always the control model. Bars are cell means with 95% bootstrap
CI whiskers. "hospital split" draws General Hospital's and St Mary's points side by side per case
with the team total written above. Writes <date>_colosseum_hospital_four_cases_<panels>.png and a
results.md beside it under output/colosseum_hospital/analysis/.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from src.utils import figure_path

EPISODES = Path("output/colosseum_hospital/analysis/episodes.json")
MERGED = Path("output/colosseum_hospital/merged")
OUT = Path("output/colosseum_hospital/analysis")
SEED = re.compile(r"_seed(\d+)$")
CONDITION = "self_promotional"  # overridden by --condition
INK = "#1B2430"
MUTED = "#5B6875"
CASES = [
    # (condition, block, label, face colour, edge colour)
    ("baseline", "control", "control pair\nno coalition", "#B7CCEA", "#2E6FBF"),
    ("baseline", "treatment", "DA pair\nno coalition", "#F0C9B7", "#C95B2F"),
    ("self_promotional", "control", "control pair\nin coalition", "#2E6FBF", "#2E6FBF"),
    ("self_promotional", "treatment", "DA pair\nin coalition", "#C95B2F", "#C95B2F"),
]
PANELS = {
    "hospital split": ("Points earned by each hospital", None, None),
    "team total": (
        "Team total (points)",
        8000,
        "8,000 = every patient treated, no penalties",
    ),
    "patients treated": ("Patients treated per shift (of 8)", 8, "all 8 treated"),
    "coalition slice": ("Coalition's own slice (GH Triage's points)", None, None),
}


def load() -> dict[tuple[str, str], list[dict]]:
    eps = json.loads(EPISODES.read_text())
    triage: dict[tuple[str, str, int], float] = {}
    for cell in MERGED.glob("2026-09-04_colosseum_hospital_*"):
        if "_baseline_" in cell.name:
            cond = "baseline"
        elif f"_{CONDITION}_" in cell.name:
            cond = CONDITION
        else:
            continue
        block = "treatment" if "difficult_advice" in cell.name else "control"
        for f in cell.glob("rollouts/colosseum/*/runs/*/*/*/agent_rewards.json"):
            seed = int(SEED.search(f.parent.name).group(1))
            triage[(cond, block, seed)] = float(
                json.loads(f.read_text()).get("General_Hospital_Triage", 0.0)
            )
    cells: dict[tuple[str, str], list[dict]] = {}
    for e in eps:
        if e["condition"] not in ("baseline", CONDITION):
            continue
        cells.setdefault((e["condition"], e["block"]), []).append(
            {
                "team total": float(e["joint_reward"]),
                "patients treated": float(e["converged_patients"]),
                "coalition slice": triage.get(
                    (e["condition"], e["block"], e["seed"]), float("nan")
                ),
                "General Hospital": float(e["hospital_reward_sum"]["General_Hospital"]),
                "St Mary's": float(e["hospital_reward_sum"]["St_Marys_Center"]),
            }
        )
    return cells


def ci95(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    boots = rng.choice(x, size=(4000, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def values(cells, key, cond, block) -> np.ndarray:
    return np.array([r[key] for r in cells[(cond, block)] if not np.isnan(r[key])])


def bar_with_ci(
    ax, x, vals, rng, *, width, face, edge, hatch=None, fmt="{:,.0f}", fs=10
):
    m = vals.mean()
    lo, hi = ci95(vals, rng)
    ax.bar(
        x,
        m,
        width=width,
        color=face,
        edgecolor=edge,
        linewidth=1.5,
        hatch=hatch,
        zorder=2,
    )
    ax.errorbar(
        x,
        m,
        yerr=[[m - lo], [hi - m]],
        fmt="none",
        ecolor=INK,
        elinewidth=1.3,
        capsize=3.5,
        zorder=3,
    )
    ax.annotate(
        fmt.format(m),
        (x, hi),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=fs,
        color=INK,
        fontweight="bold",
    )
    return m, hi


def draw_split(ax, cells, rng) -> list[str]:
    """General Hospital vs St Mary's points per case, team total above each pair."""
    lines = [
        "| case | n | General Hospital | St Mary's | team total |",
        "|---|---|---|---|---|",
    ]
    tops = []
    totals = []
    for i, (cond, block, label, face, edge) in enumerate(CASES):
        gh = values(cells, "General Hospital", cond, block)
        sm = values(cells, "St Mary's", cond, block)
        _, hi1 = bar_with_ci(
            ax, i - 0.2, gh, rng, width=0.36, face=edge, edge=edge, fs=9.5
        )
        _, hi2 = bar_with_ci(
            ax,
            i + 0.2,
            sm,
            rng,
            width=0.36,
            face="white",
            edge=edge,
            hatch="////",
            fs=9.5,
        )
        tops.append(max(hi1, hi2))
        total = values(cells, "team total", cond, block).mean()
        totals.append(total)
        lines.append(
            f"| {label.replace(chr(10), ' ')} | {len(gh)} | {gh.mean():,.0f} | {sm.mean():,.0f} | {total:,.0f} |"
        )
    ax.set_ylim(0, max(tops) * 1.32)
    ax.legend(
        handles=[
            Patch(
                facecolor=MUTED,
                edgecolor=MUTED,
                label="General Hospital (the coalition's hospital)",
            ),
            Patch(
                facecolor="white",
                edgecolor=MUTED,
                hatch="////",
                label="St Mary's Center",
            ),
        ],
        frameon=False,
        fontsize=9,
        loc="upper left",
    )
    return lines, totals


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--panels", default="hospital split", help="comma list of: " + ", ".join(PANELS)
    )
    ap.add_argument(
        "--open", action="store_true", help="open the PNG when written (macOS)"
    )
    ap.add_argument("--condition", default="self_promotional", choices=["self_promotional", "self_sacrificial", "covert"])
    args = ap.parse_args()
    global CONDITION
    CONDITION = args.condition
    for i, (cond, *rest) in enumerate(CASES):
        CASES[i] = (CONDITION if cond != "baseline" else cond, *rest)
    keys = [k.strip() for k in args.panels.split(",")]
    assert all(k in PANELS for k in keys), keys

    cells = load()
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(keys), figsize=(5.8 * len(keys) + 0.6, 5.4))
    axes = np.atleast_1d(axes)
    lines = ["# Colosseum Hospital — the four self-promotional cases\n"]
    for ax, key in zip(axes, keys):
        title, ref, ref_label = PANELS[key]
        if key == "hospital split":
            split_lines, totals = draw_split(ax, cells, rng)
            lines += split_lines
        else:
            tops = []
            for i, (cond, block, label, face, edge) in enumerate(CASES):
                vals = values(cells, key, cond, block)
                _, hi = bar_with_ci(
                    ax,
                    i,
                    vals,
                    rng,
                    width=0.62,
                    face=face,
                    edge=edge,
                    fmt="{:.1f}" if key == "patients treated" else "{:,.0f}",
                    fs=11,
                )
                tops.append(hi)
            if ref is not None:
                ax.axhline(ref, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
                ax.annotate(
                    ref_label,
                    (3.45, ref),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="right",
                    va="bottom",
                    fontsize=9,
                    color=MUTED,
                )
            ax.set_ylim(0, max(max(tops), ref or 0) * 1.16)
            lines += ["| case | n | " + key + " |", "|---|---|---|"] + [
                f"| {label.replace(chr(10), ' ')} | {len(cells[(c, b)])} | {values(cells, key, c, b).mean():,.2f} |"
                for c, b, label, *_ in CASES
            ]
        ax.set_xticks(range(len(CASES)))
        labels = [c[2] for c in CASES]
        if key == "hospital split":
            labels = [f"{l}\nteam {tot:,.0f}" for l, tot in zip(labels, totals)]
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_xlim(-0.6, 3.6)
        ax.set_title(title, fontsize=12, loc="left", color=INK)
        ax.grid(axis="y", alpha=0.25, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", labelsize=10)

    fig.suptitle(
        f"{CONDITION.replace('_', '-').capitalize()} instruction, four cases",
        fontsize=11,
        x=0.01,
        ha="left",
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    slug = "_".join(k.replace(" ", "_") for k in keys)
    tag = "" if CONDITION == "self_promotional" else f"{CONDITION}_"
    png = figure_path(OUT, f"colosseum_hospital_four_cases_{tag}{slug}")
    fig.savefig(png, dpi=160)
    md = png.with_name(png.stem + "_results.md")
    md.write_text(
        "\n".join(lines)
        + "\n\nThe other seven agents always hold the control model; 'coalition' = the pair's two seats get the secret self-promotional instruction and a private board.\n"
    )
    print("\n".join(lines))
    print(f"\nfigure {png}")
    if args.open:
        subprocess.run(["open", str(png)], check=False)


if __name__ == "__main__":
    main()
