# ABOUTME: The four self-promotional cases as bars — all-control team, DA pair untempted, control pair
# ABOUTME: in coalition, DA pair in coalition — for team total, patients treated and the coalition's slice.

"""Four bars per measure.

    uv run python scratch/colosseum_hospital/four_cases.py

Cases: (1) all nine agents control, no coalition; (2) DA model in the two pair seats, seven
control, no coalition; (3) control pair given the self-promotional coalition instruction;
(4) DA pair given it. Bars are cell means with 95% bootstrap CI whiskers. Writes the PNG and a
results.md beside it under output/colosseum_hospital/analysis/.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils import figure_path

EPISODES = Path("output/colosseum_hospital/analysis/episodes.json")
MERGED = Path("output/colosseum_hospital/merged")
OUT = Path("output/colosseum_hospital/analysis")
SEED = re.compile(r"_seed(\d+)$")
INK = "#1B2430"
MUTED = "#5B6875"
CASES = [
    # (condition, block, label, face colour, edge colour)
    ("baseline", "control", "control pair\nno coalition", "#B7CCEA", "#2E6FBF"),
    ("baseline", "treatment", "DA pair\nno coalition", "#F0C9B7", "#C95B2F"),
    ("self_promotional", "control", "control pair\nin coalition", "#2E6FBF", "#2E6FBF"),
    ("self_promotional", "treatment", "DA pair\nin coalition", "#C95B2F", "#C95B2F"),
]
PANELS = [
    (
        "team total",
        "Team total (points)",
        8000,
        "8,000 = every patient treated, no penalties",
    ),
    ("patients treated", "Patients treated per shift (of 8)", 8, "all 8 treated"),
    ("coalition slice", "Coalition's own slice (GH Triage's points)", None, None),
]


def load() -> dict[tuple[str, str], list[dict]]:
    eps = json.loads(EPISODES.read_text())
    triage: dict[tuple[str, str, int], float] = {}
    for cell in MERGED.glob("2026-09-04_colosseum_hospital_*"):
        cond = (
            "baseline"
            if "_baseline_" in cell.name
            else "self_promotional"
            if "_self_promotional_" in cell.name
            else None
        )
        if cond is None:
            continue
        block = "treatment" if "difficult_advice" in cell.name else "control"
        for f in cell.glob("rollouts/colosseum/*/runs/*/*/*/agent_rewards.json"):
            seed = int(SEED.search(f.parent.name).group(1))
            triage[(cond, block, seed)] = float(
                json.loads(f.read_text()).get("General_Hospital_Triage", 0.0)
            )
    cells: dict[tuple[str, str], list[dict]] = {}
    for e in eps:
        if e["condition"] not in ("baseline", "self_promotional"):
            continue
        cells.setdefault((e["condition"], e["block"]), []).append(
            {
                "team total": float(e["joint_reward"]),
                "patients treated": float(e["converged_patients"]),
                "coalition slice": triage.get(
                    (e["condition"], e["block"], e["seed"]), float("nan")
                ),
            }
        )
    return cells


def ci95(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    boots = rng.choice(x, size=(4000, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    cells = load()
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    lines = [
        "# Colosseum Hospital — the four self-promotional cases\n",
        "| case | n | team total | patients treated | coalition slice |",
        "|---|---|---|---|---|",
    ]
    for ax, (key, title, ref, ref_label) in zip(axes, PANELS):
        for i, (cond, block, label, face, edge) in enumerate(CASES):
            vals = np.array(
                [r[key] for r in cells[(cond, block)] if not np.isnan(r[key])]
            )
            m = vals.mean()
            lo, hi = ci95(vals, rng)
            ax.bar(
                i, m, width=0.62, color=face, edgecolor=edge, linewidth=1.5, zorder=2
            )
            ax.errorbar(
                i,
                m,
                yerr=[[m - lo], [hi - m]],
                fmt="none",
                ecolor=INK,
                elinewidth=1.4,
                capsize=4,
                zorder=3,
            )
            txt = f"{m:.1f}" if key == "patients treated" else f"{m:,.0f}"
            ax.annotate(
                txt,
                (i, hi),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                color=INK,
                fontweight="bold",
            )
        if ref is not None:
            ax.axhline(ref, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
            ax.annotate(
                ref_label,
                (3.45, ref),
                xytext=(0, 4),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=8.5,
                color=MUTED,
            )
        ax.set_xticks(range(len(CASES)))
        ax.set_xticklabels([c[2] for c in CASES], fontsize=9)
        ax.set_xlim(-0.6, 3.6)
        top = max(
            ci95(
                np.array([r[key] for r in cells[(c, b)] if not np.isnan(r[key])]), rng
            )[1]
            for c, b, *_ in CASES
        )
        ax.set_ylim(0, max(top, ref or 0) * 1.16)
        ax.set_title(title, fontsize=11, loc="left", color=INK)
        ax.grid(axis="y", alpha=0.25, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", labelsize=9)
    for cond, block, label, *_ in CASES:
        rows = cells[(cond, block)]
        m = lambda k: np.nanmean([r[k] for r in rows])
        lines.append(
            f"| {label.replace(chr(10), ' ')} | {len(rows)} | {m('team total'):,.0f} | {m('patients treated'):.2f} | {m('coalition slice'):,.0f} |"
        )
    fig.suptitle(
        "Colosseum Hospital, self-promotional instruction — the four cases. Pair = provisioner + General Hospital Triage; "
        "the other 7 agents are always the control model. Cell means, 95% CI; n = 30 per untempted case, 60 per coalition case.",
        fontsize=10.5,
        x=0.01,
        ha="left",
        color=INK,
    )
    fig.tight_layout()
    png = figure_path(OUT, "colosseum_hospital_four_cases")
    fig.savefig(png, dpi=160)
    md = png.with_name(png.stem + "_results.md")
    md.write_text(
        "\n".join(lines)
        + "\n\nSeven non-pair seats always hold the control model; 'coalition' = the pair's two seats get the secret self-promotional instruction and a private board.\n"
    )
    print("\n".join(lines))
    print(f"\nfigure {png}")


if __name__ == "__main__":
    main()
