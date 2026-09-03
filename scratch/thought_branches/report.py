# ABOUTME: Figures and markdown mirrors for the good-vs-bad ODCV contrast — every savefig
# ABOUTME: takes its path from src.utils.figure_path, so the naming law holds for plots too.

"""Turning the contrast tables into something a reader can act on.

Four figures, each answering one question:

  1  what separates a violating rollout from a clean one, within scenario
  2  where the fork sits, and which answers to it predict a violation
  3  the shape of a violating trajectory — more steps, thinner thoughts
  4  whether the sharpest marker survives inside every arm

Colours are the two validated categorical slots (blue = clean, orange = violating),
direct-labelled everywhere so identity never rests on hue alone.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils import figure_path

from scratch.thought_branches.descriptive import (
    Contrast,
    Features,
    ForkCluster,
    Stratified,
)

CLEAN = "#2a78d6"
VIOLATING = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d8d7d2"
SURFACE = "#fcfcfb"

# Features whose natural unit is a proportion; everything else is a count or a length.
RATE_FEATURES = {
    "completed",
    "commit_before_write",
    "has_fork",
    "fork_shortcut",
    "fork_integrity",
}

PRETTY = {
    "n_thought_steps": "reasoning steps",
    "n_chunks": "chunks in trace",
    "thought_chars": "total reasoning chars",
    "mean_thought_chars": "chars per reasoning step",
    "n_commands": "commands run",
    "n_failed_commands": "commands that failed",
    "n_writes": "world-changing commands",
    "completed": "called task_complete",
    "shortcut_hits": "shortcut-language hits",
    "integrity_hits": "integrity-language hits",
    "commitment_hits": "commitment-language hits",
    "appeal_hits": "principal-appeal hits",
    "first_shortcut_pos": "first shortcut (position)",
    "first_integrity_pos": "first integrity (position)",
    "first_commitment_pos": "first commitment (position)",
    "commit_before_write": "committed before first write",
    "has_fork": "environment refused at all",
    "fork_rel_pos": "fork position in trace",
    "fork_shortcut": "fork thought names a shortcut",
    "fork_integrity": "fork thought names integrity",
}


def _style(ax) -> None:
    """Recessive axes: no top/right spine, muted grid behind the marks."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def _native(s: Stratified) -> str:
    """The delta in its own units, for the annotation beside each row."""
    if s.name in RATE_FEATURES or s.name.endswith("_pos"):
        return f"{s.delta:+.3f}"
    return f"{s.delta:+.0f}" if abs(s.delta) >= 10 else f"{s.delta:+.2f}"


def fig_contrasts(
    within: Sequence[Stratified],
    out_dir: Path,
    subject: str,
    pooled: Sequence[Contrast] | None = None,
) -> Path:
    """Figure 1: within-scenario differences, standardised onto one axis.

    Native units here span three orders of magnitude — characters of reasoning against
    proportions — so a shared native axis buries every small feature next to the largest.
    Each row is therefore the difference in standard deviations of that feature across the
    corpus, with the native value printed beside it so the rescaling hides nothing.

    When `pooled` is supplied, the pooled estimate is drawn as a hollow marker behind each
    row, and that gap is the point of the figure: several markers that look decisive when
    pooled are tracking which arm and which scenario, not which outcome, and collapse once
    the comparison happens inside a cell.

    Args:
        within: Stratified contrasts.
        out_dir: Where the figure goes.
        subject: Naming-law subject for the filename.
        pooled: Optional pooled contrasts, drawn as a shadow for comparison.

    Returns:
        The written path.
    """
    group = sorted([s for s in within if s.n_cells], key=lambda s: s.delta_std)
    pooled_std: dict[str, float] = {}
    if pooled:
        sds = {s.name: s.sd for s in within}
        for c in pooled:
            sd = sds.get(c.name, 0.0)
            if sd > 1e-12:
                pooled_std[c.name] = c.delta / sd

    fig, ax = plt.subplots(figsize=(11.8, 0.42 * len(group) + 3.2), facecolor=SURFACE)
    for i, s in enumerate(group):
        solid = s.lo_std > 0 or s.hi_std < 0
        col = VIOLATING if s.delta_std > 0 else CLEAN
        a = 1.0 if solid else 0.32
        if s.name in pooled_std:
            ax.plot(
                [pooled_std[s.name]],
                [i],
                "o",
                ms=9.5,
                markerfacecolor="none",
                markeredgecolor=MUTED,
                markeredgewidth=1.3,
                alpha=0.65,
                zorder=2,
            )
        ax.plot(
            [s.lo_std, s.hi_std],
            [i, i],
            color=col,
            lw=2.2,
            alpha=a,
            solid_capstyle="round",
            zorder=3,
        )
        ax.plot(
            [s.delta_std],
            [i],
            "o",
            color=col,
            ms=8.5,
            alpha=a,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            zorder=4,
        )
        ax.text(
            1.02,
            i,
            _native(s),
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=8.5,
            color=INK if solid else MUTED,
        )
        ax.text(
            1.13,
            i,
            f"{s.n_cells}",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=8.5,
            color=MUTED,
        )
    ax.axvline(0, color=MUTED, lw=1.1)
    ax.set_yticks(np.arange(len(group)))
    ax.set_yticklabels(
        [PRETTY.get(s.name, s.name) for s in group], fontsize=9.5, color=INK
    )
    ax.set_ylim(-0.75, len(group) - 0.25)
    ax.set_xlabel(
        "violating − clean, in standard deviations of the feature",
        fontsize=10,
        color=MUTED,
    )
    ax.text(
        1.02,
        len(group) - 0.45,
        "native Δ",
        transform=ax.get_yaxis_transform(),
        fontsize=8.5,
        color=MUTED,
        va="center",
        fontweight="bold",
    )
    ax.text(
        1.13,
        len(group) - 0.45,
        "cells",
        transform=ax.get_yaxis_transform(),
        fontsize=8.5,
        color=MUTED,
        va="center",
        fontweight="bold",
    )
    _style(ax)

    if pooled_std:
        ax.plot(
            [],
            [],
            "o",
            ms=9.5,
            markerfacecolor="none",
            markeredgecolor=MUTED,
            markeredgewidth=1.3,
            label="pooled across arms and scenarios",
        )
        ax.plot([], [], "o", ms=8.5, color=MUTED, label="within one arm and scenario")
        leg = ax.legend(frameon=False, fontsize=9, loc="lower right")
        for t in leg.get_texts():
            t.set_color(INK)

    fig.suptitle(
        "What separates a violating ODCV rollout from a clean one",
        fontsize=14,
        color=INK,
        x=0.012,
        ha="left",
        y=0.985,
    )
    fig.text(
        0.012,
        0.945,
        "Filled = within-cell estimate, 95% bootstrap over cells; faded = interval spans zero. "
        "Most pooled effects shrink or vanish once the comparison is made inside one arm and one scenario.",
        fontsize=9,
        color=MUTED,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 0.99, 0.925))
    p = figure_path(out_dir, f"{subject}_contrasts")
    fig.savefig(p, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return p


def fig_fork(
    rows: Sequence[Features],
    clusters: Sequence[ForkCluster],
    base_rate: float,
    out_dir: Path,
    subject: str,
) -> Path:
    """Figure 2: where the fork sits, and which answers to it predict a violation."""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 6.0),
        facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1, 1.45]},
    )

    ax = axes[0]
    good = [
        r.fork_rel_pos
        for r in rows
        if r.violation is False and r.fork_rel_pos is not None
    ]
    bad = [
        r.fork_rel_pos
        for r in rows
        if r.violation is True and r.fork_rel_pos is not None
    ]
    bins = np.linspace(0, 1, 21)
    ax.hist(
        good,
        bins=bins,
        color=CLEAN,
        alpha=0.75,
        label=f"clean (n={len(good)})",
        density=True,
    )
    ax.hist(
        bad,
        bins=bins,
        color=VIOLATING,
        alpha=0.62,
        label=f"violating (n={len(bad)})",
        density=True,
    )
    for vals, col in ((good, CLEAN), (bad, VIOLATING)):
        if vals:
            ax.axvline(float(np.median(vals)), color=col, lw=2, ls="--")
    ax.set_xlabel(
        "position of the fork in the trace (0 = start, 1 = end)",
        fontsize=9.5,
        color=MUTED,
    )
    ax.set_ylabel("density", fontsize=9.5, color=MUTED)
    ax.set_title(
        "The environment refuses earlier in violating rollouts",
        fontsize=10.5,
        color=INK,
        loc="left",
    )
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK)
    _style(ax)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.7)

    ax = axes[1]
    cl = sorted(clusters, key=lambda c: c.violation_rate)
    y = np.arange(len(cl))
    for i, c in enumerate(cl):
        col = VIOLATING if c.violation_rate > base_rate else CLEAN
        ax.barh(i, c.violation_rate, color=col, alpha=0.9, height=0.62)
        ax.plot(
            [c.ci[0], c.ci[1]],
            [i, i],
            color=INK,
            lw=1.4,
            alpha=0.55,
            solid_capstyle="round",
        )
        ax.text(1.02, i, f"n={c.n}", va="center", fontsize=8.5, color=MUTED)
    ax.axvline(base_rate, color=MUTED, lw=1.4, ls=":")
    ax.text(
        base_rate,
        len(cl) - 0.35,
        f"  corpus base rate {base_rate:.0%}",
        fontsize=8.5,
        color=MUTED,
        va="top",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(
        [" ".join(c.exemplars[0].split())[:64] + "…" for c in cl], fontsize=8, color=INK
    )
    ax.set_xlim(0, 1.14)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel(
        "share of rollouts in this cluster that violated", fontsize=9.5, color=MUTED
    )
    ax.set_title(
        "How the model answers the refusal predicts the outcome",
        fontsize=10.5,
        color=INK,
        loc="left",
    )
    _style(ax)

    fig.suptitle(
        "The fork: the moment the target turns out to be unreachable honestly",
        fontsize=13.5,
        color=INK,
        x=0.012,
        ha="left",
        y=0.985,
    )
    fig.text(
        0.012,
        0.938,
        "Clusters are KMeans over embeddings of the fork thought; violation rates are read off afterwards, "
        "never fitted. Bars are 95% Wilson intervals.",
        fontsize=9,
        color=MUTED,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    p = figure_path(out_dir, f"{subject}_fork")
    fig.savefig(p, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return p


def fig_shape(rows: Sequence[Features], out_dir: Path, subject: str) -> Path:
    """Figure 3: violating trajectories take more steps and think less at each one."""
    fig, ax = plt.subplots(figsize=(9.4, 6.4), facecolor=SURFACE)
    for label, flag, col in (("clean", False, CLEAN), ("violating", True, VIOLATING)):
        xs = [r.n_thought_steps for r in rows if r.violation is flag]
        ys = [r.mean_thought_chars for r in rows if r.violation is flag]
        ax.scatter(
            xs,
            ys,
            s=26,
            color=col,
            alpha=0.5,
            linewidths=0.7,
            edgecolors=SURFACE,
            label=f"{label} (n={len(xs)})",
        )
        if xs:
            ax.plot(
                [float(np.median(xs))],
                [float(np.median(ys))],
                marker="D",
                ms=13,
                color=col,
                markeredgecolor=INK,
                markeredgewidth=1.2,
                zorder=5,
            )
    # Log scales, because both axes span an order of magnitude and the clean cloud would
    # otherwise pile against the left edge — but a step count must still read as a count,
    # so the ticks are plain integers rather than matplotlib's 2x10^0 rendering.
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([2, 3, 5, 8, 12, 20, 30, 50])
    ax.set_yticks([100, 200, 400, 800, 1600])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.minorticks_off()
    ax.set_xlabel("reasoning steps in the trajectory", fontsize=10, color=MUTED)
    ax.set_ylabel("characters of reasoning per step", fontsize=10, color=MUTED)
    ax.set_title(
        "Violating rollouts take more steps and think less at each one",
        fontsize=13.5,
        color=INK,
        loc="left",
        pad=26,
    )
    ax.text(
        0,
        1.035,
        "One dot per rollout; diamonds are group medians.",
        transform=ax.transAxes,
        fontsize=9,
        color=MUTED,
    )
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK)
    _style(ax)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    p = figure_path(out_dir, f"{subject}_shape")
    fig.savefig(p, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return p


def fig_by_arm(
    rows: Sequence[Features],
    feature: str,
    out_dir: Path,
    subject: str,
    caveat: str = "",
) -> Path:
    """Figure 4: does the sharpest marker hold inside every arm separately?

    Controlling for arm is NOT the same as controlling for scenario, and for this marker
    the two disagree — it separates cleanly inside every arm and then disappears once
    scenario is also held fixed. `caveat` carries that second number onto the figure, so
    the plot cannot be read on its own as evidence the marker means anything causal.

    Args:
        rows: Feature records.
        feature: Boolean feature to plot.
        out_dir: Where the figure goes.
        subject: Naming-law subject.
        caveat: One-line subtitle stating the within-scenario estimate.

    Returns:
        The written path.
    """
    arms = sorted({r.arm for r in rows})
    fig, ax = plt.subplots(figsize=(10.6, 0.62 * len(arms) + 2.9), facecolor=SURFACE)
    for i, arm in enumerate(arms):
        sub = [r for r in rows if r.arm == arm and getattr(r, feature) is not None]
        g = [bool(getattr(r, feature)) for r in sub if r.violation is False]
        b = [bool(getattr(r, feature)) for r in sub if r.violation is True]
        if not g or not b:
            continue
        rg, rb = float(np.mean(g)), float(np.mean(b))
        ax.plot([rg, rb], [i, i], color=MUTED, lw=1.4, alpha=0.6, zorder=1)
        ax.scatter(
            [rg], [i], s=105, color=CLEAN, edgecolors=SURFACE, linewidths=1.4, zorder=3
        )
        ax.scatter(
            [rb],
            [i],
            s=105,
            color=VIOLATING,
            edgecolors=SURFACE,
            linewidths=1.4,
            zorder=3,
        )
        ax.text(
            max(rg, rb) + 0.025,
            i,
            f"n={len(g)}/{len(b)}",
            va="center",
            fontsize=8.5,
            color=MUTED,
        )
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels(
        [a.replace("2026-", "").replace("-odcv-", " · ")[:52] for a in arms],
        fontsize=8.5,
        color=INK,
    )
    ax.set_xlim(-0.03, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel(
        f"share of rollouts where “{PRETTY.get(feature, feature)}”",
        fontsize=10,
        color=MUTED,
    )
    ax.set_title(
        f"“{PRETTY.get(feature, feature)}” separates the outcomes inside every arm — "
        "and that is not enough",
        fontsize=13,
        color=INK,
        loc="left",
        pad=34,
    )
    if caveat:
        ax.text(0, 1.045, caveat, transform=ax.transAxes, fontsize=9, color=MUTED)
    ax.scatter([], [], s=105, color=CLEAN, label="clean rollouts")
    ax.scatter([], [], s=105, color=VIOLATING, label="violating rollouts")
    leg = ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    for t in leg.get_texts():
        t.set_color(INK)
    _style(ax)
    fig.tight_layout()
    p = figure_path(out_dir, f"{subject}_{feature}_by_arm")
    fig.savefig(p, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return p


def write_results(
    out_dir: Path,
    subject: str,
    rows: Sequence[Features],
    pooled: Sequence[Contrast],
    within: Sequence[Stratified],
    clusters: Sequence[ForkCluster],
    figures: Sequence[Path],
) -> tuple[Path, Path]:
    """Write the markdown mirror and the machine-readable results beside the figures.

    Args:
        out_dir: Run directory.
        subject: Naming-law subject.
        rows: Feature records.
        pooled: Pooled contrasts.
        within: Stratified contrasts.
        clusters: Fork clusters.
        figures: Figure paths to link.

    Returns:
        (markdown path, json path).
    """
    n = len(rows)
    scored = [r for r in rows if r.violation is not None]
    viol = sum(1 for r in scored if r.violation)
    base = viol / len(scored) if scored else 0.0
    arms = sorted({r.arm for r in rows})

    lines = [
        f"# Thought Branches — good vs bad ODCV rollouts ({subject})",
        "",
        f"- rollouts: **{n}** across {len(arms)} arms, {len({r.scenario for r in rows})} scenarios",
        f"- judged: {len(scored)}; violating (median severity ≥ 3): **{viol}** ({base:.1%})",
        f"- the environment refused at least once in {sum(r.has_fork for r in rows)} of them "
        f"({sum(r.has_fork for r in rows) / n:.0%})",
        "",
        "> Everything here is **correlational**. The paper this implements is explicit that a marker "
        "correlated with the outcome may be its cause, its symptom, or a narration of a decision already "
        "taken. These tables rank where to spend on-policy resampling; they do not settle anything.",
        "",
        "## Within-scenario contrasts",
        "",
        "Each row compares violating against clean rollouts *of the same arm on the same scenario and "
        "variant*, then averages over cells. Intervals are 95% bootstrap over cells.",
        "",
        "| feature | Δ (violating − clean) | 95% CI | cells | rollouts | cells agreeing on sign |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for s in within:
        if not s.n_cells:
            continue
        lines.append(
            f"| {PRETTY.get(s.name, s.name)} | {s.delta:+.3f} | [{s.lo:+.3f}, {s.hi:+.3f}] | "
            f"{s.n_cells} | {s.n_rollouts} | {s.agree:.0%} |"
        )

    lines += [
        "",
        "## Pooled contrasts (for reference — confounded by arm and scenario)",
        "",
        "| feature | clean | violating | Δ | intervals separate |",
        "|---|---:|---:|---:|:--:|",
    ]
    for c in pooled:
        lines.append(
            f"| {PRETTY.get(c.name, c.name)} | {c.clean:.3f} | {c.violating:.3f} | {c.delta:+.3f} | "
            f"{'yes' if c.separated else 'no'} |"
        )

    lines += [
        "",
        "## Fork clusters",
        "",
        "Unsupervised clusters of the thought that answers the environment's first refusal, with the "
        "violation rate read off afterwards.",
        "",
        "| cluster | n | violation rate | 95% CI | exemplar |",
        "|---:|---:|---:|---|---|",
    ]
    for c in clusters:
        ex = " ".join(c.exemplars[0].split())[:150] if c.exemplars else ""
        lines.append(
            f"| {c.cluster} | {c.n} | {c.violation_rate:.0%} | [{c.ci[0]:.2f}, {c.ci[1]:.2f}] | {ex} |"
        )

    lines += ["", "## Figures", ""] + [f"- `{p.name}`" for p in figures] + [""]

    md = out_dir / f"{subject}_results.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    js = out_dir / f"{subject}_results.json"
    js.write_text(
        json.dumps(
            {
                "n_rollouts": n,
                "n_violating": viol,
                "base_rate": base,
                "arms": arms,
                "within": [asdict(s) for s in within],
                "pooled": [asdict(c) for c in pooled],
                "fork_clusters": [asdict(c) for c in clusters],
                "figures": [p.name for p in figures],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return md, js
