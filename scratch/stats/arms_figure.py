# ABOUTME: Shared data + figure for the two-arm ODCV comparison, so the scratch and src
# ABOUTME: implementations of the error bars can be plotted identically and diffed.
"""One loader, one figure, two callers.

`plot_arms_scratch.py` computes the intervals with `scratch/stats/crossed_ci.py`;
`plot_arms_src.py` computes them with `src/eval/stats.py`. Everything else -- the data, the
axes, the colours -- comes from here, so any difference between the two figures is a
difference between the two statistics implementations and nothing else.

Data: the six seed repos, INCENTIVIZED ONLY, first judged pass per cell, on the scenarios
all six models share. Seeds 42 and 69 have a single pass, so one pass per seed is the only
comparable choice even though numina seed 0 now has four.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
COLOR = {"numina control": BLUE, "5% difficult advice": ORANGE}


def figure(arms: dict[str, dict], diff: dict, per_seed: dict[str, list[float]], subtitle: str,
           out_dir: Path, stem: str, extra: str = "") -> Path:
    """Two panels: the arms with their intervals, and the paired difference.

    Args:
        arms: {arm: {"mean", "lo", "hi", "df", "method"}} in percent.
        diff: the same for (5% difficult advice - numina control).
        per_seed: {arm: [per-seed MR in percent]}, drawn as dots over each bar.
        subtitle: Which implementation produced the numbers.
        out_dir, stem: Where to write `<stem>_<timestamp>.png`.
        extra: Optional footnote.

    Returns:
        The PNG path.
    """
    names = list(arms)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.5, 5.4), facecolor="#fcfcfb",
                                   gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.28})
    top = max(max(a["hi"] for a in arms.values()), max(max(v) for v in per_seed.values())) * 1.28 + 6

    for x, arm in enumerate(names):
        a = arms[arm]
        ax0.bar(x, a["mean"], width=0.52, color=COLOR[arm], alpha=0.85, zorder=2)
        ax0.errorbar(x, a["mean"], yerr=[[a["mean"] - a["lo"]], [a["hi"] - a["mean"]]], fmt="none",
                     ecolor=INK, elinewidth=1.5, capsize=8, capthick=1.5, zorder=4)
        ax0.scatter([x] * len(per_seed[arm]), per_seed[arm], s=26, facecolor="white",
                    edgecolor=INK, linewidth=1.1, zorder=5)
        ax0.text(x, a["hi"] + 1.5, f"{a['mean']:.1f}%\n[{a['lo']:.1f}, {a['hi']:.1f}]",
                 ha="center", va="bottom", fontsize=9.5, color=INK)
    ax0.set_xticks(range(len(names)), names)
    ax0.set_xlim(-0.62, len(names) - 0.38)
    ax0.set_ylim(0, top)
    ax0.set_ylabel("misalignment rate (%)", fontsize=10, color=INK2)
    ax0.set_title("per arm — models AND scenarios sampled\n"
                  f"{arms[names[0]]['method']}   (open dots: the 3 seeds)",
                  fontsize=9.5, loc="left", color=INK)

    ax1.errorbar(0, diff["mean"], yerr=[[diff["mean"] - diff["lo"]], [diff["hi"] - diff["mean"]]],
                 fmt="o", color=INK, ecolor=INK, elinewidth=2, capsize=10, capthick=2,
                 markersize=9, zorder=3)
    ax1.axhline(0, color=INK, lw=1, zorder=2)
    ax1.text(0.16, diff["mean"], f"{diff['mean']:+.1f} pp\n[{diff['lo']:+.1f}, {diff['hi']:+.1f}]",
             ha="left", va="center", fontsize=10, color=INK)
    ax1.text(-0.42, 1.0, "no difference", fontsize=8.5, color=INK2, va="bottom")
    ax1.set_xticks([0], ["5% difficult advice\n− numina control"])
    ax1.set_xlim(-0.5, 0.9)
    ax1.set_ylim(min(diff["lo"] * 1.25, -5), max(8, diff["hi"] + 8))
    ax1.set_ylabel("difference in misalignment rate (pp)", fontsize=10, color=INK2)
    method = diff["method"]
    if len(method) > 46:                      # the five-term name overruns the panel
        cut = method.rfind(",", 0, 46)
        method = method[:cut] + "\n" + method[cut + 2:]
    ax1.set_title(f"the gap — paired on scenario\n{method}", fontsize=9, loc="left", color=INK)

    for ax in (ax0, ax1):
        ax.yaxis.grid(True, color=GRID, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9.5)
    fig.suptitle(subtitle, fontsize=11.5, color=INK, y=0.98)
    if extra:
        fig.text(0.01, 0.015, extra, fontsize=8.5, color=INK2)
    fig.subplots_adjust(top=0.82, bottom=0.17, left=0.075, right=0.99)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}_{stamp}.png"
    fig.savefig(png, dpi=160)
    return png


def dump(payload: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, default=float))
