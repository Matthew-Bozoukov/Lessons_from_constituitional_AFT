# ABOUTME: Appendix figure: Figure 2's Petri bar broken down by judge dimension. For each of the 19 misbehaviour
# ABOUTME: dimensions, each arm's mean (score - 1) per audit; the 19 values of an arm sum to its Figure 2 bar.
"""Reads the per-audit judge scores cached by fig_petri_misbehaviour.audits (Hub Inspect logs). A
three-adapter arm is the plain mean of its adapters. Two panels with their own x-scales: the four
overall judgements (concerning, needs attention, disappointing, concerning with no mitigating factors)
and the fifteen specific behaviours. Dimensions that are zero in every arm are named under the figure
instead of drawn. No intervals: one audit per seed per adapter.

Run: uv run python scratch/petri_dims/fig_petri_per_dimension.py
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from scratch.petri_dims.fig_petri_misbehaviour import (  # noqa: E402
    ARMS,
    BASE,
    BASE_COLOUR,
    MISBEHAVIOUR,
    ORG,
    audits,
)
from src.utils import save_figure_pdf  # noqa: E402

OVERALL = [
    "concerning",
    "needs_attention",
    "disappointing",
    "concerning_with_no_mitigating_factors",
]
DIM_DIR = Path(".venv/lib/python3.12/site-packages/inspect_petri/_judge/dimensions")


def display_name(dim: str) -> str:
    m = re.search(r'^display_name:\s*"(.+)"', (DIM_DIR / f"{dim}.md").read_text(), re.M)
    assert m, dim
    return m.group(1)


def per_dimension(repos: list[str]) -> dict[str, float]:
    return {
        d: sum(sum(a[d] - 1 for a in audits(r)) / len(audits(r)) for r in repos)
        / len(repos)
        for d in MISBEHAVIOUR
    }


def main() -> None:
    vals = {key: per_dimension(repos) for key, _, _, repos in ARMS}
    base = per_dimension([BASE])
    zero = [
        d
        for d in MISBEHAVIOUR
        if all(v[d] == 0 for v in vals.values()) and base[d] == 0
    ]
    specific = sorted(
        [d for d in MISBEHAVIOUR if d not in OVERALL and d not in zero],
        key=lambda d: -max(v[d] for v in vals.values()),
    )
    panels = [("Overall judgements", OVERALL), ("Specific behaviours", specific)]

    plt.rcParams.update({"font.size": 12, "pdf.fonttype": 42})
    heights = [len(OVERALL), len(specific)]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 0.62 * sum(heights) + 1.9),
        gridspec_kw={"height_ratios": heights, "hspace": 0.25},
    )
    n = len(ARMS)
    w = 0.8 / n
    for ax, (title, dims) in zip(axes, panels):
        for j, d in enumerate(dims):
            y = len(dims) - 1 - j
            for k, (key, _, colour, _) in enumerate(ARMS):
                ax.barh(
                    y + (n - 1) / 2 * w - k * w,
                    vals[key][d],
                    w,
                    color=colour,
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=3,
                )
            if base[d] > 0:  # a zero base mark would sit on the axis line
                ax.plot(
                    [base[d], base[d]],
                    [y - 0.42, y + 0.42],
                    color=BASE_COLOUR,
                    linestyle=(0, (2.5, 2.5)),
                    linewidth=1.6,
                    zorder=4,
                )
        ax.set_yticks(range(len(dims)))
        ax.set_yticklabels([display_name(d) for d in dims][::-1])
        ax.set_ylim(-0.6, len(dims) - 0.4)
        ax.set_title(title, loc="left", fontsize=13)
        ax.grid(axis="x", linestyle="--", alpha=0.25, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[-1].set_xlabel("Mean judge score above minimum, per audit (lower is better)")
    handles = [
        Line2D(
            [0],
            [0],
            color=BASE_COLOUR,
            linestyle=(0, (2.5, 2.5)),
            linewidth=1.6,
            label="Qwen3.6-27B",
        )
    ] + [Patch(facecolor=c, edgecolor="black", label=label) for _, label, c, _ in ARMS]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.subplots_adjust(
        left=0.36, right=0.97, top=0.96, bottom=1.55 / fig.get_figheight()
    )

    out = Path("output/petri")
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{datetime.now():%Y-%m-%d}_petri_per_dimension"
    fig.savefig(f"{stem}.png", dpi=150)
    artifacts = [f"{ORG}/{r}" for _, _, _, repos in ARMS for r in repos] + [
        f"{ORG}/{BASE}"
    ]
    save_figure_pdf(
        fig,
        f"{stem}.pdf",
        "Petri per-dimension breakdown",
        artifacts,
        "scratch/petri_dims/fig_petri_per_dimension.py",
    )
    md = [
        f"# {stem.name}",
        "",
        "| Dimension | "
        + " | ".join(label for _, label, _, _ in ARMS)
        + " | Qwen3.6-27B |",
        "|---" * (len(ARMS) + 2) + "|",
    ]
    for d in MISBEHAVIOUR:
        md.append(
            f"| {display_name(d)} | "
            + " | ".join(f"{vals[k][d]:.3f}" for k, _, _, _ in ARMS)
            + f" | {base[d]:.3f} |"
        )
    md.append(
        "| **Sum (Figure 2)** | "
        + " | ".join(f"{sum(vals[k].values()):.3f}" for k, _, _, _ in ARMS)
        + f" | {sum(base.values()):.3f} |"
    )
    md += ["", "Zero in every arm: " + ", ".join(display_name(d) for d in zero)]
    Path(f"{stem}_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("zero:", [display_name(d) for d in zero])
    print(stem)


if __name__ == "__main__":
    main()
