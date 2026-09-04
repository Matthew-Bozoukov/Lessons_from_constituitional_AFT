# ABOUTME: The real Colosseum figure — reads pooled contrasts + per-seed measures from the
# ABOUTME: run dirs and draws what scratch/colosseum_mock_exp1_plot.py mocked up.

"""Turn the finished runs into the figure and the table.

Same two panels the mock agreed before any GPU was rented, now on real data:

  A  coalition advantage per arm, one dot per seed plus mean and 95% CI
  B  normalised regret in every cell, so an arm that "colludes less" by simply being a
     worse teammate is visible instead of hidden

Panel B matters more than it looks. The cooperation control has already shown the
treatment arm sitting HIGHER on regret with no private objective and no channel at all,
so any reduction in coalition advantage has to be read against that.

    uv run python scratch/colosseum_report.py \
        --experiment collusion --control <dir> --treatment <dir>

Writes output/colosseum/<date>_colosseum_jira_<experiment>.png and prints the table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from src.eval.misalignment.colosseum.config import EXPERIMENTS
from src.utils import figure_path

# Validated categorical pair (dataviz six-checks, light surface): lightness band, chroma
# floor, CVD separation, normal-vision floor and contrast all pass.
CONTROL, CONTROL_EDGE = "#2F6DB5", "#1B4472"
TREAT, TREAT_EDGE = "#D97706", "#7C4406"
INK, MUTED = "#333333", "#8A8A8A"


def _measures(run_dir: Path) -> dict:
    raw = json.loads((run_dir / "results" / "per_seed.json").read_text())
    return {
        m: {
            c: {int(s): float(v) for s, v in seeds.items()}
            for c, seeds in cells.items()
        }
        for m, cells in raw.items()
    }


def _mean_ci(values):
    v = np.asarray(list(values), float)
    m = float(v.mean())
    if v.size < 2:
        return m, m, m
    h = 1.96 * float(v.std(ddof=1)) / np.sqrt(v.size)
    return m, m - h, m + h


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E4E4E4", linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    p.add_argument("--control", required=True)
    p.add_argument("--treatment", required=True)
    p.add_argument("--contrasts", help="pooled contrasts.json, for the printed table")
    args = p.parse_args()

    cells = [name for name, _, _ in EXPERIMENTS[args.experiment]["cells"]]
    ctrl = _measures(Path(args.control))
    treat = _measures(Path(args.treatment))

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 20,
            "axes.labelsize": 17,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "axes.edgecolor": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )
    two = len(cells) == 2
    fig, axes = plt.subplots(
        1,
        3 if two else 1,
        figsize=(21 if two else 8, 6.4),
        facecolor="white",
        squeeze=False,
    )
    axes = axes[0]
    fig.suptitle(
        f"Colosseum Jira — experiment: {args.experiment} (Qwen3.6-27B, thinking mode)",
        fontsize=22,
        y=1.01,
    )

    rng = np.random.default_rng(0)

    # ── Panel A: what the private objective buys, paired within seed ────────────
    if two:
        baseline, treated = cells
        axA = axes[1]
        _style(axA)
        for i, (label, measures, colour, edge) in enumerate(
            [
                ("Tulu-only\ncontrol", ctrl, CONTROL, CONTROL_EDGE),
                ("7% principle-only\ndifficult advice", treat, TREAT, TREAT_EDGE),
            ]
        ):
            adv = measures["coalition_advantage"]
            shared = sorted(set(adv.get(treated, {})) & set(adv.get(baseline, {})))
            delta = [adv[treated][s] - adv[baseline][s] for s in shared]
            jitter = rng.uniform(-0.09, 0.09, len(delta))
            axA.scatter(
                np.full(len(delta), i) + jitter,
                delta,
                s=40,
                color=colour,
                alpha=0.30,
                linewidths=0,
                zorder=2,
            )
            m, lo, hi = _mean_ci(delta)
            axA.errorbar(
                i,
                m,
                yerr=[[m - lo], [hi - m]],
                fmt="o",
                markersize=13,
                color=colour,
                markeredgecolor=edge,
                markeredgewidth=2,
                ecolor=edge,
                elinewidth=2.5,
                capsize=7,
                capthick=2.5,
                zorder=3,
            )
            axA.annotate(
                f"{m:+.2f}\nn={len(delta)}",
                (i, hi),
                textcoords="offset points",
                xytext=(0, 12),
                ha="center",
                fontsize=15,
                color=INK,
                fontweight="bold",
            )
        axA.axhline(0, color=MUTED, linewidth=1.5, linestyle=(0, (4, 3)), zorder=1)
        axA.set_xticks([0, 1])
        axA.set_xticklabels(
            ["Tulu-only\ncontrol", "7% principle-only\ndifficult advice"]
        )
        axA.set_xlim(-0.45, 1.45)
        axA.set_ylabel(
            f"Coalition advantage gained\n({treated} − {baseline}, paired by seed)"
        )
        axA.set_title("B · what the private objective BUYS (paired)", pad=14)

    # ── Panel A: coalition advantage LEVELS, where the effect actually is ───────
    # The delta panel beside this one is null, and on its own it would under-report the
    # result: the arms differ by ~4.5 reward units in the collusion cell AND by ~4.0 in
    # the baseline cell, which is the finding — a general trait, not a collusion-specific
    # one. A figure showing only the delta would say "no effect" about data that has one.
    if two:
        axL = axes[0]
        _style(axL)
        xs0 = np.arange(len(cells))
        for measures, colour, edge, off in [
            (ctrl, CONTROL, CONTROL_EDGE, -0.055),
            (treat, TREAT, TREAT_EDGE, 0.055),
        ]:
            adv = measures["coalition_advantage"]
            stats = [
                _mean_ci(adv.get(c, {}).values()) if adv.get(c) else (np.nan,) * 3
                for c in cells
            ]
            ms = [s_[0] for s_ in stats]
            lo = [s_[0] - s_[1] for s_ in stats]
            hi = [s_[2] - s_[0] for s_ in stats]
            axL.plot(xs0 + off, ms, color=colour, linewidth=2.5, zorder=2)
            axL.errorbar(
                xs0 + off, ms, yerr=[lo, hi], fmt="o", markersize=12, color=colour,
                markeredgecolor=edge, markeredgewidth=2, ecolor=edge, elinewidth=2.5,
                capsize=7, capthick=2.5, zorder=3, linestyle="none",
            )
        axL.axhline(0, color=MUTED, linewidth=1.5, linestyle=(0, (4, 3)), zorder=1)
        axL.set_xticks(xs0)
        axL.set_xticklabels(cells)
        axL.set_xlim(-0.5, len(cells) - 0.5)
        axL.set_ylabel("Coalition advantage\n(coalition mean reward − everyone else's)")
        axL.set_title("A · how much the coalition CAPTURES", pad=14)

    # ── Panel B: the cost to the whole team, in every cell ──────────────────────
    axB = axes[2] if two else axes[0]
    _style(axB)
    xs = np.arange(len(cells))
    for label, measures, colour, edge, off in [
        ("Tulu-only control", ctrl, CONTROL, CONTROL_EDGE, -0.055),
        ("7% principle-only difficult advice", treat, TREAT, TREAT_EDGE, 0.055),
    ]:
        reg = measures["normalised_regret"]
        stats = [
            _mean_ci(reg.get(c, {}).values()) if reg.get(c) else (np.nan,) * 3
            for c in cells
        ]
        ms = [s[0] for s in stats]
        lo = [s[0] - s[1] for s in stats]
        hi = [s[2] - s[0] for s in stats]
        if len(cells) > 1:
            axB.plot(xs + off, ms, color=colour, linewidth=2.5, zorder=2)
        axB.errorbar(
            xs + off,
            ms,
            yerr=[lo, hi],
            fmt="o",
            markersize=12,
            color=colour,
            markeredgecolor=edge,
            markeredgewidth=2,
            ecolor=edge,
            elinewidth=2.5,
            capsize=7,
            capthick=2.5,
            zorder=3,
            linestyle="none",
        )
    axB.set_xticks(xs)
    axB.set_xticklabels(cells)
    axB.set_xlim(-0.5, len(cells) - 0.5)
    axB.set_ylim(bottom=0)
    axB.set_ylabel("Normalised overall regret\n(0 = optimal joint allocation)")
    axB.set_title("C · what it costs the six-agent team", pad=14)
    axB.legend(
        handles=[
            Patch(facecolor=CONTROL, edgecolor=CONTROL_EDGE, label="Tulu-only control"),
            Patch(
                facecolor=TREAT,
                edgecolor=TREAT_EDGE,
                label="7% principle-only difficult advice",
            ),
        ],
        loc="upper left",
        frameon=False,
    )

    fig.text(
        0.5,
        -0.06,
        "Higher regret = further from the computable optimum. Bars are 95% CI over "
        "seeds; seeds are shared across arms and cells, so contrasts are paired.",
        ha="center",
        fontsize=13,
        color=MUTED,
    )

    path = figure_path("output/colosseum", f"colosseum_jira_{args.experiment}")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {path}")

    if args.contrasts:
        c = json.loads(Path(args.contrasts).read_text())["contrasts"]
        print(f"\n=== {args.experiment}: treatment minus control ===")
        for name, r in c.items():
            lo_, hi_ = r["diff_ci95"]
            flag = "  <-- excludes 0" if (lo_ > 0 or hi_ < 0) else ""
            print(
                f"{name:34s} treat={r['treatment_mean']:+9.3f} "
                f"ctrl={r['control_mean']:+9.3f} diff={r['diff']:+9.3f} "
                f"[{lo_:+.3f}, {hi_:+.3f}] p={r['p_two_sided']:.4f} "
                f"n={r['n_seeds']}{flag}"
            )


if __name__ == "__main__":
    main()
