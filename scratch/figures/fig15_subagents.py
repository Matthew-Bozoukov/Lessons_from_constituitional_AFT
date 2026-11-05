# ABOUTME: Fig. 15 remake: Subagents Comply More (the delegated-harm eval) over control / Deliberative Alignment /
# ABOUTME: difficult advice / difficult advice + tool definitions, every bar read from the Hub at render time.
# Run: uv run python scratch/figures/fig15_subagents.py [--out_dir output/figures]
"""Four arms over three outcomes (harmful action, explicit refusal, concern raised with a human) on one shared
0-100 scale, in the Fig. 2 layout. A bar is the arm's rate over its valid episodes with a Wilson 95% interval
over those episodes: the interval the submitted figure drew (writeup/figures/fig15_subagents.pdf), kept so that
the 2026-11-05 relabel changes nothing but the legend wording. Declared in tests/test_figure_intervals.py::
WILSON_OVER_ITEMS; the per-item rule (_hub.item_se over scenarios x repeats) is still undecided for this eval.

Arms (seed 0, 15% of supervised tokens on the 2026-10-05 plain base):
  control   2026-10-12-dh-qwen36-0-plain
  delib     2026-10-12-dh-qwen36-0-delib-sonnet-15
  da        2026-10-12-dh-qwen36-0-da-15
  da_tools  2026-10-12-dh-qwen36-0-da-tools-15
"""

from __future__ import annotations

from datetime import datetime

import fire
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.naming import figure_path
from src.utils import save_figure_pdf
from scratch.figures import _hub

matplotlib.use("Agg")

ARMS = ["control", "delib", "da", "da_tools"]
ARM = {
    "control": "+ Control (Standard SFT)",
    "delib": "+ Deliberative Alignment 15% mix",
    "da": "+ Difficult Advice 15% mix",
    "da_tools": "+ Difficult Advice + tool definitions 15% mix",
}
COLOUR = {
    "control": "#5a6472",
    "delib": "#2196f3",
    "da": "#7e22ce",
    "da_tools": "#d0558f",
}
RUNS = {
    "control": "2026-10-12-dh-qwen36-0-plain",
    "delib": "2026-10-12-dh-qwen36-0-delib-sonnet-15",
    "da": "2026-10-12-dh-qwen36-0-da-15",
    "da_tools": "2026-10-12-dh-qwen36-0-da-tools-15",
}
PANELS = ["harmful_action", "explicit_refusal", "concern_any_human"]
TITLE = {
    "harmful_action": "Harmful action\n(lower is better)",
    "explicit_refusal": "Explicit refusal\n(higher is better)",
    "concern_any_human": "Concern raised with a human\n(higher is better)",
}

plt.rcParams.update(
    {
        "font.size": 19,
        "axes.labelsize": 19,
        "xtick.labelsize": 17,
        "ytick.labelsize": 18,
        "axes.titlesize": 20,
        "legend.fontsize": 18,
    }
)


def read(repo: str) -> dict[str, tuple[float, float, float, int]]:
    """Each outcome's rate (%) over the run's valid episodes, its Wilson 95% interval and the episode count."""
    arm = _hub.fetch(repo, "results/results.json")["arms"]["subagent"]
    n = int(arm["valid"])
    out = {}
    for metric in PANELS:
        rate = float(arm["rates"][metric])
        k = round(rate * n)
        assert abs(k / n - rate) < 1e-6, (
            f"{repo}: {metric} rate {rate} is not a count over {n} valid episodes"
        )
        out[metric] = (100 * rate, *_hub.wilson(k, n), n)
    return out


def main(out_dir: str = "output/figures") -> None:
    vals = {a: read(RUNS[a]) for a in ARMS}

    fig, axes = plt.subplots(1, 3, figsize=(20.0, 7.8), gridspec_kw={"wspace": 0.30})
    for ax, p in zip(axes, PANELS):
        for i, a in enumerate(ARMS):
            m, lo, hi, _ = vals[a][p]
            ax.bar(
                i,
                m,
                width=0.68,
                color=COLOUR[a],
                edgecolor="black",
                linewidth=0.8,
                zorder=3,
            )
            ax.errorbar(
                i,
                m,
                yerr=[[m - lo], [hi - m]],
                fmt="none",
                ecolor="#2b2523",
                elinewidth=1.3,
                capsize=5,
                capthick=1.3,
                zorder=4,
            )
            ax.text(
                i,
                hi + 1.4,
                f"{m:.1f}",
                ha="center",
                va="bottom",
                fontsize=16,
                fontweight="bold",
                zorder=5,
                clip_on=False,
            )
        ax.set_title(TITLE[p], pad=14)
        ax.set_ylabel("Episodes (%)")
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20))
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([])
        ax.set_xlim(-0.65, len(ARMS) - 0.35)
        ax.grid(True, axis="y", linestyle="--", alpha=0.2)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles = [
        Patch(facecolor=COLOUR[a], edgecolor="black", label=ARM[a]) for a in ARMS
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=18,
        columnspacing=1.6,
        handlelength=1.8,
        handleheight=1.0,
        labelspacing=0.5,
    )
    fig.subplots_adjust(top=0.76, bottom=0.08, wspace=0.30)

    png = figure_path(out_dir, "fig15-subagents", ext="png")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    artifacts = [f"{_hub.ORG}/{RUNS[a]}" for a in ARMS]
    save_figure_pdf(
        fig,
        png.with_suffix(".pdf"),
        "Fig. 15: Subagents Comply More over four arms",
        artifacts,
        "scratch/figures/fig15_subagents.py",
    )
    md = [
        f"# {png.stem}",
        "",
        f"Rendered {datetime.now().isoformat(timespec='seconds')} by "
        "scratch/figures/fig15_subagents.py; every bar read from the Hub at render time. Wilson 95% intervals",
        "over the run's valid episodes (12 scenarios x 9 repeats scheduled, invalid episodes dropped).",
        "",
        "| Panel | Arm | Value | 95% CI | Valid episodes | Run |",
        "|---|---|---|---|---|---|",
    ]
    for p in PANELS:
        for a in ARMS:
            m, lo, hi, n = vals[a][p]
            md.append(
                f"| {p} | {ARM[a]} | {m:.1f} | [{lo:.1f}, {hi:.1f}] | {n} | {RUNS[a]} |"
            )
    png.with_name(png.stem + "_results.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    print("\n".join(md))
    print(f">>> {png}")


if __name__ == "__main__":
    fire.Fire(main)
