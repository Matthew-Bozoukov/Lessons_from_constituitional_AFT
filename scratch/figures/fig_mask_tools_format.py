# ABOUTME: MASK+tools honesty for DA 15%, DA+tools (native tool block) and DA+tools in Qwen-Agent's tool format,
# ABOUTME: with the untuned base as a dashed line; every bar read from the Hub at render time.
# Run: uv run python scratch/figures/fig_mask_tools_format.py
"""One panel, three single-adapter arms (seed 0), bars styled as the tools-dose figures. Bars: mean over
MASK's rows with a 95% interval from each row's passes (`_hub.mask`). The base ran one pass, so it is a line."""
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scratch.figures import _hub
from src.utils import save_figure_pdf

OUT = Path("output/plots")
BASE_NAME = "Qwen3.6-27B"
ARMS = ["da", "dat", "fmt"]
ARM = {
    "da": "+ Difficult Advice 15% mix",
    "dat": "+ DA with tools",
    "fmt": "+ DA with tools, Qwen-Agent tool format",
}
COLOUR = {"da": "#7e22ce", "dat": "#e8762b", "fmt": "#e3a21a"}
BASE_COLOUR = "#c1121f"
RUNS = {"da": "2026-11-11-mask-tools-qwen36-0-da-15",
        "dat": "2026-11-11-mask-tools-qwen36-0-da-tools-15",
        "fmt": "2026-11-17-mask-tools-qwen36-0-da-tools-15-tools-format-qwen-agent"}
BASE_RUN = "2026-11-16-mask-tools-qwen36"

plt.rcParams.update({"font.size": 19, "axes.labelsize": 19, "xtick.labelsize": 17,
                     "ytick.labelsize": 18, "axes.titlesize": 20, "legend.fontsize": 17})


def main() -> None:
    vals = {a: _hub.mask(RUNS[a]) for a in ARMS}
    base = _hub.mask(BASE_RUN)[0]

    fig, ax = plt.subplots(figsize=(8.6, 8.0))
    for i, a in enumerate(ARMS):
        m, lo, hi = vals[a]
        ax.bar(i, m, width=0.68, color=COLOUR[a], edgecolor="black", linewidth=0.8, zorder=3)
        if lo is not None:
            ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor="#2b2523",
                        elinewidth=1.3, capsize=5, capthick=1.3, zorder=4)
        top_of = hi if hi is not None else m
        ax.text(i, top_of + 1.4, f"{m:.1f}", ha="center", va="bottom",
                fontsize=16, fontweight="bold", zorder=5, clip_on=False)
    ax.axhline(base, color=BASE_COLOUR, linestyle=(0, (2.5, 2.5)), linewidth=1.8, zorder=2)
    ax.set_ylim(0, 100); ax.set_yticks(range(0, 101, 20))
    ax.set_title("MASK+tools honesty\n(higher is better)", pad=14)
    ax.set_ylabel("Honesty score (%)")
    ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([])
    ax.set_xlim(-0.65, len(ARMS) - 0.35)
    ax.grid(True, axis="y", linestyle="--", alpha=0.2); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    handles = [Line2D([0], [0], color=BASE_COLOUR, linestyle=(0, (2.5, 2.5)), linewidth=1.8, label=BASE_NAME)]
    handles += [Patch(facecolor=COLOUR[a], edgecolor="black", label=ARM[a]) for a in ARMS]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=1,
               bbox_to_anchor=(0.5, 1.0), handlelength=1.8, handleheight=1.0, labelspacing=0.5)
    fig.subplots_adjust(top=0.70, bottom=0.06)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"{datetime.now():%Y-%m-%d}_mask_tools_tool_format"
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    artifacts = [f"{_hub.ORG}/{r}" for r in [*RUNS.values(), BASE_RUN]]
    save_figure_pdf(fig, Path(f"{stem}.pdf"), "MASK+tools: DA tool format", artifacts,
                    "scratch/figures/fig_mask_tools_format.py")
    md = [f"# {stem.name}", "", f"Rendered {datetime.now().isoformat(timespec='seconds')} by "
          "scratch/figures/fig_mask_tools_format.py; every bar read from the Hub at render time.", "",
          "| Arm | Honesty | 95% CI | Run |", "|---|---|---|---|"]
    md += [f"| {ARM[a]} | {vals[a][0]:.1f} | "
           + (f"[{vals[a][1]:.1f}, {vals[a][2]:.1f}]" if vals[a][1] is not None else "-")
           + f" | {RUNS[a]} |" for a in ARMS]
    md += [f"| {BASE_NAME} (base line, one pass) | {base:.1f} | | {BASE_RUN} |"]
    Path(f"{stem}_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"{stem}.png")


if __name__ == "__main__":
    main()
