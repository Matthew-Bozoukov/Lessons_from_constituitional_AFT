# ABOUTME: Per-shift team totals under the self-sacrificial instruction, control vs DA pair: seed dots
# ABOUTME: with mean and median marked, to show equal means hiding a milder DA median and a shared tail.

"""uv run python scratch/colosseum_hospital/ss_distribution.py [--open]"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils import figure_path

ROOT = Path("output/colosseum_hospital/merged")
OUT = Path("output/colosseum_hospital/analysis")
CELLS = {
    "control pair": "2026-09-04_colosseum_hospital_self_sacrificial_qwen36_table2_only_9284",
    "DA pair": "2026-09-04_colosseum_hospital_self_sacrificial_qwen36_difficult_advice_chunk_only_702",
}
COLOR = {"control pair": "#2E6FBF", "DA pair": "#C95B2F"}
INK, MUTED = "#1B2430", "#5B6875"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    lines = [
        "| pair | n | mean | median | shifts >= 6,000 | shifts < 3,000 | worst |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, (label, cell) in enumerate(CELLS.items()):
        tot = np.array(
            [
                sum(json.loads(f.read_text()).values())
                for f in (ROOT / cell).glob(
                    "rollouts/colosseum/*/runs/*/*/*/agent_rewards.json"
                )
            ]
        )
        x = i + rng.normal(0, 0.06, len(tot))
        ax.scatter(x, tot, s=22, alpha=0.5, color=COLOR[label], linewidths=0, zorder=2)
        ax.hlines(tot.mean(), i - 0.28, i + 0.28, color=INK, lw=2.2, zorder=3)
        ax.hlines(
            np.median(tot),
            i - 0.28,
            i + 0.28,
            color=INK,
            lw=2.2,
            ls=(0, (3, 2)),
            zorder=3,
        )
        ax.annotate(
            f"mean {tot.mean():,.0f}",
            (i + 0.3, tot.mean()),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK,
        )
        ax.annotate(
            f"median {np.median(tot):,.0f}",
            (i + 0.3, np.median(tot)),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK,
        )
        lines.append(
            f"| {label} | {len(tot)} | {tot.mean():,.0f} | {np.median(tot):,.0f} | {np.mean(tot >= 6000):.0%} | {np.mean(tot < 3000):.0%} | {tot.min():,.0f} |"
        )
    ax.axhline(8000, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.annotate(
        "8,000 = every patient treated, no penalties",
        (1.55, 8000),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=MUTED,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ["control pair\nin coalition", "DA pair\nin coalition"], fontsize=10
    )
    ax.set_xlim(-0.55, 1.6)
    ax.set_ylabel("team total (points), one dot per shift", fontsize=10)
    ax.set_title(
        "Self-sacrificial instruction: same mean, different middle",
        fontsize=11,
        loc="left",
        color=INK,
    )
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.plot([], [], color=INK, lw=2.2, label="mean")
    ax.plot([], [], color=INK, lw=2.2, ls=(0, (3, 2)), label="median")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    png = figure_path(
        OUT, "colosseum_hospital_self_sacrificial_team_total_distribution"
    )
    fig.savefig(png, dpi=160)
    (png.with_name(png.stem + "_results.md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nfigure {png}")
    if args.open:
        subprocess.run(["open", str(png)], check=False)


if __name__ == "__main__":
    main()
