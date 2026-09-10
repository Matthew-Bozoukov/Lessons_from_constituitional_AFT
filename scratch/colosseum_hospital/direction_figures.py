# ABOUTME: Three plain figures for the 2026-09-10 fixed-harness run: refusal by channel and
# ABOUTME: harness, the DA pair's trajectory classes, and the team-level outcomes.

"""PYTHONPATH=scratch/colosseum_hospital uv run python scratch/colosseum_hospital/direction_figures.py

Reads the same cells as direction_contrasts.py (paper harness 2026-09-04; carried history A;
board access B; control and difficult-advice arms; 30 seeds each) and writes, under
output/colosseum_hospital/analysis/:

  <date>_colosseum_hospital_refusal_by_harness.png   share of shifts the judge rates >= 3 for
                                                      refusal, in the seats' private reasoning
                                                      and on the boards, per harness and arm
  <date>_colosseum_hospital_trajectory_classes.png   the difficult-advice provisioner's and
                                                      Triage's per-shift trajectory class
  <date>_colosseum_hospital_team_outcomes.png        team regret, resource failures and
                                                      scheduler rejections per shift

Error bars are 95% intervals: Wilson for shares, bootstrap of the mean for the rest.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from direction_contrasts import ARMS, ROOT, load, per_seed
from src.eval.misalignment.colosseum.stats import arm_difference
from trajectory_classes import classes_for
from decline_vs_action import PROV, TRIAGE

from src.utils import figure_path

OUT = Path("output/colosseum_hospital/analysis")
HARNESS = [
    ("paper", "paper harness\n(2026-09-04)"),
    ("A", "A · carried\nhistory"),
    ("B", "B · board\naccess"),
]
ARM_COLOR = {
    "control": "#2a78d6",
    "treatment": "#eb6834",
}  # dataviz reference slots 1, 2
ARM_LABEL = {"control": "control", "treatment": "difficult advice"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2dd"


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z, p = 1.96, k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def boot_ci(vals, n_boot: int = 4000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, dtype=float)
    means = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def style(ax, ylabel: str, ymax=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    if ymax is not None:
        ax.set_ylim(0, ymax)


def grouped_bars(
    ax,
    per_arm: dict[str, list[tuple[float, float, float]]],
    ylabel: str,
    fmt: str,
    ymax=None,
):
    """per_arm[arm] = [(value, lo, hi) per harness]; bars control | difficult advice per harness."""
    x = np.arange(len(HARNESS))
    w = 0.36
    for i, arm in enumerate(("control", "treatment")):
        vals = per_arm[arm]
        xs = x + (i - 0.5) * w
        ys = [v for v, _, _ in vals]
        err = [[v - lo for v, lo, _ in vals], [hi - v for v, _, hi in vals]]
        ax.bar(xs, ys, w * 0.94, color=ARM_COLOR[arm], label=ARM_LABEL[arm], zorder=3)
        ax.errorbar(
            xs, ys, yerr=err, fmt="none", ecolor=INK, elinewidth=1, capsize=2, zorder=4
        )
        for xx, v, (_, _, hi) in zip(xs, ys, vals):
            ax.text(
                xx,
                hi + (ymax or max(ys) * 1.1) * 0.02,
                fmt.format(v),
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=INK,
            )
    ax.set_xticks(x, [h for _, h in HARNESS])
    style(ax, ylabel, ymax)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cells = {k: load(*k) for k in ARMS}
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": MUTED}
    )

    # ── 1. refusal by channel and harness ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
    for ax, key, title in (
        (axes[0], "refused_reasoning", "in the seats' private reasoning"),
        (axes[1], "refused_board", "on a board (what the study could see)"),
    ):
        per_arm = {}
        for arm in ("control", "treatment"):
            vals = []
            for h, _ in HARNESS:
                rows = [r[key] for r in cells[(h, arm)] if r.get(key) is not None]
                k, n = int(sum(rows)), len(rows)
                lo, hi = wilson(k, n)
                vals.append((k / n if n else 0.0, lo, hi))
            per_arm[arm] = vals
        grouped_bars(
            ax, per_arm, "share of 30 shifts, judge refusal ≥ 3", "{:.0%}", ymax=1.12
        )
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=2)
    fig.suptitle(
        "Where the refusal is: self-sacrificial shifts, three harnesses",
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    p1 = figure_path(OUT, "colosseum_hospital_refusal_by_harness")
    fig.savefig(p1, dpi=170)
    plt.close(fig)

    # ── 2. trajectory classes of the difficult-advice seats ──────────────────────
    bins = [
        ("refused and held", ["refused"], "#2a78d6"),
        (
            "flipped (declined, then complied)",
            ["two-faced", "declined-but-sabotaged", "repented"],
            "#9cc1ee",
        ),
        ("complied throughout", ["complied"], "#eb6834"),
        ("passive", ["passive"], "#c3c2b7"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
    for ax, seat, title in (
        (axes[0], PROV, "Resource Provisioner"),
        (axes[1], TRIAGE, "General Hospital Triage"),
    ):
        x = np.arange(len(HARNESS))
        bottoms = np.zeros(len(HARNESS))
        for label, members, color in bins:
            heights = []
            for h, _ in HARNESS:
                c = classes_for(ROOT / ARMS[(h, "treatment")])[seat]
                heights.append(sum(c.get(m, 0) for m in members))
            heights = np.array(heights, dtype=float)
            ax.bar(x, heights, 0.58, bottom=bottoms, color=color, label=label, zorder=3)
            for xx, b, hgt in zip(x, bottoms, heights):
                if hgt >= 3:
                    ax.text(
                        xx,
                        b + hgt / 2,
                        f"{int(hgt)}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white" if color in ("#2a78d6", "#eb6834") else INK,
                    )
            bottoms += heights
        ax.set_xticks(x, [h for _, h in HARNESS])
        style(ax, "shifts (of 30)", 31)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, loc="lower center", ncol=4)
    fig.suptitle(
        "The difficult-advice pair, shift by shift: did it refuse, and did it hold?",
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    p2 = figure_path(OUT, "colosseum_hospital_trajectory_classes")
    fig.savefig(p2, dpi=170)
    plt.close(fig)

    # ── 3. team-level outcomes ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9))
    for ax, key, ylabel, fmt, ymax in (
        (axes[0], "regret_norm", "team regret (1 − score / 8,000)", "{:.2f}", 0.55),
        (axes[1], "resource_failures", "resource failures per shift", "{:.1f}", 22),
        (
            axes[2],
            "scheduler_rejections",
            "schedule calls refused per shift",
            "{:.0f}",
            66,
        ),
    ):
        per_arm = {}
        for arm in ("control", "treatment"):
            vals = []
            for h, _ in HARNESS:
                rows = [r[key] for r in cells[(h, arm)] if r.get(key) is not None]
                m = float(np.mean(rows)) if rows else 0.0
                lo, hi = boot_ci(rows) if rows else (0.0, 0.0)
                vals.append((m, lo, hi))
            per_arm[arm] = vals
        grouped_bars(ax, per_arm, ylabel, fmt, ymax=ymax)
    axes[2].text(
        0.5,
        0.97,
        "0 under the paper harness:\nthe strict scheduler did not exist",
        transform=axes[2].transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=2)
    fig.suptitle(
        "What happened to the team: means over 30 shifts, 95% bootstrap intervals",
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    p3 = figure_path(OUT, "colosseum_hospital_team_outcomes")
    fig.savefig(p3, dpi=170)
    plt.close(fig)

    # ── 4. the team score, and the paired difference between the arms ───────────
    fig, axes = plt.subplots(
        1, 2, figsize=(10.4, 3.9), gridspec_kw={"width_ratios": [1.35, 1]}
    )
    per_arm = {}
    for arm in ("control", "treatment"):
        vals = []
        for h, _ in HARNESS:
            rows = [
                r["joint_reward"]
                for r in cells[(h, arm)]
                if r.get("joint_reward") is not None
            ]
            lo, hi = boot_ci(rows)
            vals.append((float(np.mean(rows)), lo, hi))
        per_arm[arm] = vals
    grouped_bars(
        axes[0], per_arm, "team score per shift (max 8,000)", "{:,.0f}", ymax=8000
    )
    axes[0].axhline(8000, color=MUTED, linewidth=0.8, linestyle=":")
    axes[0].text(
        2.55, 8000, "clean shift", ha="right", va="bottom", fontsize=8, color=MUTED
    )
    axes[0].set_title("team score, both arms", fontsize=10.5, color=INK, loc="left")

    ax = axes[1]
    x = np.arange(len(HARNESS))
    diffs = []
    for h, _ in HARNESS:
        r = arm_difference(
            per_seed(cells[(h, "treatment")], "joint_reward"),
            per_seed(cells[(h, "control")], "joint_reward"),
            label="joint_reward",
        )
        diffs.append(r)
    ys = [r["diff"] for r in diffs]
    err = [
        [r["diff"] - r["diff_ci95"][0] for r in diffs],
        [r["diff_ci95"][1] - r["diff"] for r in diffs],
    ]
    colors = ["#eb6834" if r["p_two_sided"] < 0.05 else "#f5b79a" for r in diffs]
    ax.bar(x, ys, 0.5, color=colors, zorder=3)
    ax.errorbar(
        x, ys, yerr=err, fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=4
    )
    ax.axhline(0, color=INK, linewidth=0.8)
    for xx, r in zip(x, diffs):
        top = r["diff_ci95"][1]
        ax.text(
            xx,
            top + 120,
            f"{r['diff']:+,.0f}\np = {r['p_two_sided']:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=INK,
        )
    ax.set_xticks(x, [h for _, h in HARNESS])
    style(ax, "difficult advice − control, points (paired on seed)")
    ax.set_ylim(-3200, 3600)
    ax.set_title(
        "the paired difference, 95% interval", fontsize=10.5, color=INK, loc="left"
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=2)
    fig.suptitle(
        "Did the rewards change? Team score by harness and arm, 30 seeds each",
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    p4 = figure_path(OUT, "colosseum_hospital_team_score")
    fig.savefig(p4, dpi=170)
    plt.close(fig)
    for p in (p1, p2, p3, p4):
        print(p)


if __name__ == "__main__":
    main()
