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
                "label": f"{d['scenario'][:30]} · {d['variant'][:4]}",
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
    for tick, r in zip(ax.get_yticklabels(), rows):
        tick.set_color(FORCING if r["violation"] else INK)
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
        "The pale tick marks where the recorded rollout landed. Orange labels are rollouts the "
        "judges scored as violations."
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

    fire.Fire({"build": build})
