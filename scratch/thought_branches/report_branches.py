# ABOUTME: Figures for the resampling run — the branch spread at each fork, and how many
# ABOUTME: rounds the fork thought survives before the model lets its content go.

"""Plotting a distribution over trajectories.

The paper's whole thesis is that one rollout is not the object of study; the distribution
is. So the headline figure here is one row per fork point showing what twenty-four reruns
of the same model from the *same* prefix actually did — a row that is all one colour is a
decision the upstream context had already made, and a row that is split is a real fork.

Actions collapse to three groups rather than the seven `mechanical.py` assigns. That is
not simplification for the chart's sake: validating the fine-grained labels against judged
outcomes showed `fabricate` and `investigate` at the fork are nearly indistinguishable
(68% vs 66% violation on this arm), while `report` is clean (0%). Encoding distinctions
the data does not support would be a chart telling a story the numbers do not.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.naming import figure_path

# Validated categorical slots 1-3: they clear the all-pairs CVD and normal-vision floors
# in both light and dark, which the wider palette does not.
REPORTING = "#2a78d6"
INVESTIGATING = "#1baf7a"
FORCING = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d8d7d2"
SURFACE = "#fcfcfb"

GROUPS = (
    ("forcing", FORCING, ("fabricate", "modify_tool", "suppress")),
    ("investigating", INVESTIGATING, ("investigate", "other")),
    ("reporting", REPORTING, ("report", "none")),
)


def group_of(kind: str) -> str:
    for name, _, kinds in GROUPS:
        if kind in kinds:
            return name
    return "investigating"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_axisbelow(True)


def fig_spread(
    dist: Sequence[dict], out_dir: Path, subject: str, verify: float | None = None
) -> Path:
    """One row per fork point: what N reruns from the identical prefix did.

    Args:
        dist: Rows from `<subject>_distribution.jsonl`.
        out_dir: Where the figure goes.
        subject: Naming-law subject.
        verify: Greedy action-kind agreement, quoted in the subtitle when given.

    Returns:
        The written path.
    """
    rows = []
    for d in dist:
        g = Counter()
        for kind, n in (d.get("kinds") or {}).items():
            g[group_of(kind)] += n
        tot = sum(g.values()) or 1
        rows.append(
            {
                "label": f"{d['scenario'][:28]} · {d['variant'][:4]} · {'V' if d.get('violation') else 'C'}",
                "forcing": g["forcing"] / tot,
                "investigating": g["investigating"] / tot,
                "reporting": g["reporting"] / tot,
                "violation": d.get("violation"),
                "recorded": group_of(d.get("recorded_kind", "other")),
                "n": tot,
            }
        )
    rows.sort(key=lambda r: -r["forcing"])

    h = max(4.5, 0.27 * len(rows) + 2.9)
    fig, ax = plt.subplots(figsize=(12.4, h), facecolor=SURFACE)
    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    for name, colour, _ in GROUPS:
        w = np.array([r[name] for r in rows])
        ax.barh(y, w, left=left, color=colour, height=0.74, label=name)
        left += w

    # Where the recorded rollout actually landed, as a tick on its own group's band.
    for i, r in enumerate(rows):
        offs = {
            "forcing": r["forcing"] / 2,
            "investigating": r["forcing"] + r["investigating"] / 2,
            "reporting": r["forcing"] + r["investigating"] + r["reporting"] / 2,
        }
        ax.plot(
            [offs[r["recorded"]]],
            [i],
            marker="|",
            ms=13,
            mew=2.1,
            color=SURFACE,
            zorder=4,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [r["label"] for r in rows],
        fontsize=7.6,
        color=INK,
    )
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel(
        f"share of {rows[0]['n'] if rows else 0} reruns from the identical prefix",
        fontsize=10,
        color=MUTED,
    )
    _style(ax)
    ax.grid(True, axis="x", color=SURFACE, linewidth=0.8, alpha=0.6)

    leg = ax.legend(
        frameon=False,
        fontsize=9.5,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.10 - 6.0 / (len(rows) * 3 + 30)),
    )
    for t in leg.get_texts():
        t.set_color(INK)

    fig.suptitle(
        "What the model does when you rerun it from the same fork",
        fontsize=14,
        color=INK,
        x=0.008,
        ha="left",
        y=0.985,
    )
    sub = (
        "Each row is one fork point; the bar is the on-policy action distribution across reruns. "
        "The pale tick marks where the recorded rollout landed. V / C is how the judges scored "
        "that original rollout — violation or clean."
    )
    if verify is not None:
        sub += f" Greedy resampling reproduces the recorded action kind {verify:.0%} of the time."
    fig.text(
        0.008, 0.955, _wrap(sub, 150), fontsize=8.8, color=MUTED, ha="left", va="top"
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.925))
    p = figure_path(out_dir, f"{subject}_branch_spread")
    fig.savefig(p, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return p


def _wrap(text: str, width: int) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width))


def fig_resilience(res: Sequence[dict], out_dir: Path, subject: str) -> Path:
    """How many rounds the fork thought's content survives repeated resampling."""
    if not res:
        raise ValueError("no resilience rows to plot")
    kmax = max(r["max_rounds"] for r in res)
    viol = [r["resilience"] for r in res if r.get("violation")]
    clean = [r["resilience"] for r in res if not r.get("violation")]

    fig, ax = plt.subplots(figsize=(9.4, 5.4), facecolor=SURFACE)
    bins = np.arange(-0.5, kmax + 1.5, 1)
    ax.hist(
        [clean, viol],
        bins=bins,
        color=[REPORTING, FORCING],
        label=[
            f"clean rollouts (n={len(clean)})",
            f"violating rollouts (n={len(viol)})",
        ],
        rwidth=0.82,
    )
    ax.set_xticks(range(kmax + 1))
    ax.set_xlabel(
        "rounds of resampling survived before the content stopped returning",
        fontsize=10,
        color=MUTED,
    )
    ax.set_ylabel("fork points", fontsize=10, color=MUTED)
    ax.set_title(
        "Resilience of the thought that answers the refusal",
        fontsize=13.5,
        color=INK,
        loc="left",
        pad=30,
    )
    ax.text(
        0,
        1.045,
        _wrap(
            "The paper's Algorithm 1. 0 means the model dropped the idea the first time it "
            "was resampled; the cap is the round budget, so a bar at the cap is a lower bound.",
            108,
        ),
        transform=ax.transAxes,
        fontsize=9,
        color=MUTED,
        va="bottom",
    )
    leg = ax.legend(frameon=False, fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(INK)
    _style(ax)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    p = figure_path(out_dir, f"{subject}_resilience")
    fig.savefig(p, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return p


def resilience_at(best_sim_by_round: Sequence[float], tau: float) -> int:
    """Rounds survived at threshold `tau` — the count of leading rounds still above it.

    Algorithm 1 walks rounds in order and stops the first time the best candidate fails to
    reach tau, so this is a prefix count, not a total. Scoring it from stored similarities
    means one set of generations answers the question at every threshold.
    """
    k = 0
    for s in best_sim_by_round:
        if s > tau:
            k += 1
        else:
            break
    return k


def fig_resilience_sweep(rows: Sequence[dict], out_dir: Path, subject: str) -> Path:
    """Resilience against threshold, because the threshold is the whole metric.

    A single tau produces a single number and no way to tell a real result from a
    saturated one. Sweeping it shows both: where the curve sits flat at the round cap the
    metric has no resolution, and the range where it falls is the only place a difference
    between fork points could be read.
    """
    taus = np.linspace(0.0, 1.0, 101)
    curves = np.array(
        [[resilience_at(r["best_sim_by_round"], t) for t in taus] for r in rows]
    )
    kmax = max(r["max_rounds"] for r in rows)
    viol = np.array([bool(r.get("violation")) for r in rows])

    fig, (ax, ax2) = plt.subplots(
        1,
        2,
        figsize=(13.2, 5.4),
        facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1.35, 1]},
    )

    for c in curves:
        ax.plot(taus, c, color=MUTED, alpha=0.16, lw=1)
    if viol.any():
        ax.plot(
            taus,
            curves[viol].mean(0),
            color=FORCING,
            lw=2.4,
            label=f"violating originals (n={int(viol.sum())})",
        )
    if (~viol).any():
        ax.plot(
            taus,
            curves[~viol].mean(0),
            color=REPORTING,
            lw=2.4,
            label=f"clean originals (n={int((~viol).sum())})",
        )
    ax.axhline(kmax, color=GRID, lw=1.2, ls="--")
    ax.text(
        0.01,
        kmax - 0.12,
        "round budget — saturated above this",
        fontsize=8.5,
        color=MUTED,
        va="top",
    )
    ax.set_xlabel("similarity threshold τ", fontsize=10, color=MUTED)
    ax.set_ylabel("rounds survived", fontsize=10, color=MUTED)
    ax.set_ylim(-0.2, kmax + 0.35)
    ax.set_title(
        "Resilience only has resolution in a narrow band of τ",
        fontsize=11.5,
        color=INK,
        loc="left",
    )
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK)
    _style(ax)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.7)

    # What the similarities actually are, round by round — the raw quantity behind it.
    per_round = np.array([r["best_sim_by_round"] for r in rows], dtype=float)
    for i in range(per_round.shape[0]):
        ax2.plot(
            range(1, per_round.shape[1] + 1),
            per_round[i],
            color=FORCING if viol[i] else REPORTING,
            alpha=0.42,
            lw=1.2,
            marker="o",
            ms=3.5,
        )
    ax2.plot(
        range(1, per_round.shape[1] + 1),
        per_round.mean(0),
        color=INK,
        lw=2.4,
        marker="o",
        ms=5,
        label="mean",
    )
    ax2.set_xticks(range(1, per_round.shape[1] + 1))
    ax2.set_xlabel("resampling round", fontsize=10, color=MUTED)
    ax2.set_ylabel("best similarity to the original thought", fontsize=10, color=MUTED)
    ax2.set_ylim(0, 1)
    ax2.set_title(
        "The content does not drift away with repeated resampling",
        fontsize=11.5,
        color=INK,
        loc="left",
    )
    _style(ax2)
    ax2.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.7)

    fig.suptitle(
        "Resilience of the fork thought",
        fontsize=14,
        color=INK,
        x=0.008,
        ha="left",
        y=0.985,
    )
    fig.text(
        0.008,
        0.935,
        _wrap(
            "Algorithm 1 counts rounds until the best of 12 regenerations stops reaching τ. "
            "Taking a maximum over candidates makes the count saturate for any lenient τ, so "
            "the honest reading is the curve, not one number.",
            150,
        ),
        fontsize=8.8,
        color=MUTED,
        ha="left",
        va="top",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    p = figure_path(out_dir, f"{subject}_resilience_sweep")
    fig.savefig(p, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return p


def build_sweep(run_directory: str, subject: str = "odcv_fork_resilience") -> None:
    """Figures for a `resilience_sweep` run directory."""
    out = Path(run_directory)
    rows = [
        json.loads(l)
        for l in (out / f"{subject}_rounds.jsonl").read_text().splitlines()
        if l.strip()
    ]
    p = fig_resilience_sweep(rows, out, subject)
    taus = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
    lines = [
        f"# Resilience of the fork thought — {subject}",
        "",
        f"{len(rows)} fork points, {rows[0]['max_rounds']} rounds x 12 candidates.",
        "",
        "| τ | mean rounds survived | fork points at 0 | at the cap |",
        "|---:|---:|---:|---:|",
    ]
    for t in taus:
        ks = [resilience_at(r["best_sim_by_round"], t) for r in rows]
        lines.append(
            f"| {t:.1f} | {np.mean(ks):.2f} | {sum(1 for k in ks if k == 0)} | "
            f"{sum(1 for k in ks if k == rows[0]['max_rounds'])} |"
        )
    lines += ["", f"- `{p.name}`", ""]
    md = out / f"{subject}_results.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {md}\n  {p}")


def fig_curve(rows: Sequence[dict], out_dir: Path, subject: str) -> Path:
    """Where along a trajectory the model's next move stops being open.

    The y axis is how CONCENTRATED the action distribution is at each cut — the share of
    reruns landing on the modal action. 1.0 means every rerun did the same thing, so the
    behaviour there is already settled; lower means the model is genuinely undecided and a
    training example at that point would be teaching something still in play.
    """
    rel = np.array([r["rel_pos"] for r in rows], dtype=float)
    modal = np.array([r["modal_share"] for r in rows], dtype=float)
    forcing = np.array([r["p_forcing"] for r in rows], dtype=float)
    viol = np.array([bool(r["violation"]) for r in rows])
    is_fork = np.array([bool(r["is_fork"]) for r in rows])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.6), facecolor=SURFACE)

    # Left: concentration against position, binned, with the raw points behind.
    ax.scatter(
        rel[~is_fork],
        modal[~is_fork],
        s=26,
        color=MUTED,
        alpha=0.28,
        linewidths=0,
        label="branch point",
    )
    ax.scatter(
        rel[is_fork],
        modal[is_fork],
        s=52,
        color=FORCING,
        alpha=0.85,
        edgecolors=SURFACE,
        linewidths=1.1,
        label="the fork",
        zorder=4,
    )
    bins = np.linspace(0, 1, 6)
    mids, means, los, his = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (rel >= lo) & (rel <= hi if hi == 1.0 else rel < hi)
        if m.sum() >= 3:
            v = modal[m]
            mids.append((lo + hi) / 2)
            means.append(v.mean())
            los.append(v.mean() - v.std() / np.sqrt(v.size))
            his.append(v.mean() + v.std() / np.sqrt(v.size))
    if mids:
        ax.plot(
            mids,
            means,
            color=REPORTING,
            lw=2.6,
            marker="o",
            ms=6,
            zorder=5,
            label="mean ± s.e.",
        )
        ax.fill_between(mids, los, his, color=REPORTING, alpha=0.16, zorder=1)
    ax.set_ylim(0.25, 1.04)
    ax.set_xlim(-0.04, 1.04)
    ax.set_xlabel(
        "position in the trajectory (0 = first step, 1 = last)",
        fontsize=10,
        color=MUTED,
    )
    ax.set_ylabel("share of reruns taking the modal action", fontsize=10, color=MUTED)
    ax.set_title(
        "How settled the next move is, by position",
        fontsize=11.5,
        color=INK,
        loc="left",
    )
    leg = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for t in leg.get_texts():
        t.set_color(INK)
    _style(ax)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.7)

    # Right: does WHAT it does shift with position, and does it differ by outcome?
    for flag, colour, name in (
        (True, FORCING, "violating originals"),
        (False, REPORTING, "clean originals"),
    ):
        m = viol == flag
        if m.sum() < 3:
            continue
        mids2, mean2 = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            b = m & (rel >= lo) & (rel <= hi if hi == 1.0 else rel < hi)
            if b.sum() >= 2:
                mids2.append((lo + hi) / 2)
                mean2.append(forcing[b].mean())
        if mids2:
            ax2.plot(
                mids2,
                mean2,
                color=colour,
                lw=2.4,
                marker="o",
                ms=6,
                label=f"{name} (n={int(m.sum())} points)",
            )
    ax2.set_ylim(-0.02, max(0.35, float(forcing.max()) * 1.15))
    ax2.set_xlim(-0.04, 1.04)
    ax2.set_xlabel("position in the trajectory", fontsize=10, color=MUTED)
    ax2.set_ylabel("share of reruns that force the metric", fontsize=10, color=MUTED)
    ax2.set_title(
        "What it does, by position and by how the original ended",
        fontsize=11.5,
        color=INK,
        loc="left",
    )
    leg2 = ax2.legend(frameon=False, fontsize=9, loc="upper left")
    for t in leg2.get_texts():
        t.set_color(INK)
    _style(ax2)
    ax2.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.7)

    fig.suptitle(
        "Where the decision is still open",
        fontsize=14,
        color=INK,
        x=0.008,
        ha="left",
        y=0.985,
    )
    fig.text(
        0.008,
        0.935,
        _wrap(
            "Each point is one cut through one trajectory, resampled 16 times. A concentration "
            "of 1.0 means every rerun took the same action, so nothing there is still in play. "
            "The fork run measured a single cut; this measures the whole trace.",
            152,
        ),
        fontsize=8.8,
        color=MUTED,
        ha="left",
        va="top",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    p = figure_path(out_dir, f"{subject}_curve")
    fig.savefig(p, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return p


def build_curve(run_directory: str, subject: str = "odcv_effect_curve") -> None:
    """Figure and table for an `effect_curve` run directory."""
    out = Path(run_directory)
    rows = [
        json.loads(l)
        for l in (out / f"{subject}_points.jsonl").read_text().splitlines()
        if l.strip()
    ]
    p = fig_curve(rows, out, subject)
    rel = np.array([r["rel_pos"] for r in rows])
    modal = np.array([r["modal_share"] for r in rows])
    lines = [
        f"# Where the decision is still open — {subject}",
        "",
        f"{len(rows)} branch points across {len({r['key'] for r in rows})} trajectories, "
        f"{rows[0]['n']} reruns each.",
        "",
        "`modal share` is the fraction of reruns taking the most common action: 1.0 means "
        "the next move is fully settled, lower means the model is genuinely undecided there.",
        "",
        "| position | points | mean modal share | fully settled (=1.0) | mean forcing |",
        "|---|---:|---:|---:|---:|",
    ]
    for lo, hi in zip(np.linspace(0, 1, 6)[:-1], np.linspace(0, 1, 6)[1:]):
        m = (rel >= lo) & (rel <= hi if hi == 1.0 else rel < hi)
        if m.sum() == 0:
            continue
        sub = [r for r, k in zip(rows, m) if k]
        lines.append(
            f"| {lo:.1f}–{hi:.1f} | {int(m.sum())} | {modal[m].mean():.2f} | "
            f"{sum(1 for r in sub if r['modal_share'] >= 0.999)} | "
            f"{np.mean([r['p_forcing'] for r in sub]):.2f} |"
        )
    fk = [r for r in rows if r["is_fork"]]
    if fk:
        lines += [
            "",
            f"At the fork specifically: mean modal share "
            f"{np.mean([r['modal_share'] for r in fk]):.2f} over {len(fk)} points.",
        ]
    lines += ["", f"- `{p.name}`", ""]
    md = out / f"{subject}_results.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {md}\n  {p}")


def write_report(
    out_dir: Path,
    subject: str,
    dist: Sequence[dict],
    res: Sequence[dict],
    summary: dict,
    figures: Sequence[Path],
) -> Path:
    """Markdown mirror beside the figures."""
    tot = Counter()
    for d in dist:
        for kind, n in (d.get("kinds") or {}).items():
            tot[group_of(kind)] += n
    n_all = sum(tot.values()) or 1
    split = [d for d in dist if 0.15 <= d["p_forcing"] <= 0.85]
    lines = [
        f"# Resampling ODCV fork points — {subject}",
        "",
        f"- arm: `{summary.get('arm')}`  ·  served: `{summary.get('target') or summary.get('model')}`",
        f"- {summary.get('n_branches')} fork points × {summary.get('n_samples')} reruns at "
        f"temperature {summary.get('temperature')}  ·  ~{summary.get('generations')} generations",
        f"- **greedy resampling reproduces the recorded action kind "
        f"{summary.get('verify_agreement', 0):.0%} of the time** — the check that the prefix "
        "reconstruction is faithful",
        "",
        "## The branch is real",
        "",
        f"Across all reruns: {tot['forcing'] / n_all:.0%} forcing, {tot['investigating'] / n_all:.0%} "
        f"investigating, {tot['reporting'] / n_all:.0%} reporting.",
        "",
        f"**{len(split)} of {len(dist)} fork points are genuinely split** (between 15% and 85% "
        "forcing) — at those, the outcome was not settled by the time the model reached the fork. "
        "The rest were already decided upstream.",
        "",
        "| scenario | variant | judged | recorded | forcing | reporting |",
        "|---|---|---|---|---:|---:|",
    ]
    for d in sorted(dist, key=lambda d: -d["p_forcing"]):
        lines.append(
            f"| {d['scenario']} | {d['variant']} | "
            f"{'violation' if d['violation'] else 'clean'} | {d['recorded_kind']} | "
            f"{d['p_forcing']:.0%} | {d['p_report']:.0%} |"
        )
    if res:
        ks = [r["resilience"] for r in res]
        lines += [
            "",
            "## Resilience",
            "",
            f"Median {int(np.median(ks))} of {res[0]['max_rounds']} rounds over {len(res)} fork "
            f"points; {sum(1 for k in ks if k == 0)} dropped the idea on the first resample.",
            "",
            "| scenario | judged | recorded | tau | rounds survived |",
            "|---|---|---|---:|---:|",
        ]
        for r in sorted(res, key=lambda r: -r["resilience"]):
            lines.append(
                f"| {r['scenario']} | {'violation' if r['violation'] else 'clean'} | "
                f"{r['recorded_kind']} | {r['tau']:.2f} | {r['resilience']}/{r['max_rounds']} |"
            )
    lines += ["", "## Figures", ""] + [f"- `{p.name}`" for p in figures] + [""]
    p = out_dir / f"{subject}_results.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def build(run_directory: str, subject: str = "odcv_fork_resampling") -> None:
    """Load a finished run directory and write its figures and report."""
    out = Path(run_directory)
    dist = [
        json.loads(l)
        for l in (out / f"{subject}_distribution.jsonl").read_text().splitlines()
        if l.strip()
    ]
    rp = out / f"{subject}_resilience.jsonl"
    res = (
        [json.loads(l) for l in rp.read_text().splitlines() if l.strip()]
        if rp.is_file()
        else []
    )
    summary = json.loads((out / f"{subject}_summary.json").read_text())
    figs = [fig_spread(dist, out, subject, verify=summary.get("verify_agreement"))]
    if res:
        figs.append(fig_resilience(res, out, subject))
    md = write_report(out, subject, dist, res, summary, figs)
    print(f"wrote {md}")
    for f in figs:
        print(f"  {f}")


if __name__ == "__main__":
    import fire

    fire.Fire({"build": build, "build_sweep": build_sweep, "build_curve": build_curve})
