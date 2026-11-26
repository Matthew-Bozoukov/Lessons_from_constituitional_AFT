# ABOUTME: MASK and ODCV over control, DA, DA+tools and the two base-tool-dose arms (half / double the
# ABOUTME: base's tool rows), in the trace-ablations figure style; every bar read from the Hub at render time.
# Run: uv run python scratch/figures/fig_base_tools_dose_two_panel.py
"""Five single-adapter arms (training draw A throughout, which is what the dose arms were built on),
two probes, each panel on its own scale with a "Chat eval" / "Single-agent eval" rule underneath.
Bars: mean over the benchmark's items with a 95% interval from each item's repeats (`_hub.item_se`).
The red dashed line is the untuned base model.
"""
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
ARMS = ["control", "da", "dat", "half", "double"]
ARM = {
    "control": "+ Control (standard SFT)",
    "da": "+ Difficult Advice 15% mix",
    "dat": "+ DA with tools",
    "half": "+ DA with tools, half base tool rows",
    "double": "+ DA with tools, double base tool rows",
}
# Report-wide arm colours: control grey, DA purple, DA+tools orange; the two dose arms are the light
# and dark ends of the same orange so they read as DA+tools variants.
COLOUR = {"control": "#5a6472", "da": "#7e22ce", "dat": "#e8762b", "half": "#f5b98a", "double": "#9a4a12"}
BASE_COLOUR = "#c1121f"

PANELS = ["mask", "odcv"]
RUNS = {
    "mask": {"control": "2026-11-03-mask-qwen36-0-plain",
             "da": "2026-11-03-mask-qwen36-0-da-15",
             "dat": "2026-11-09-mask-qwen36-0-da-tools-15",
             "half": "2026-11-16-mask-qwen36-0-da-tools-15-base-tools-half",
             "double": "2026-11-17-mask-qwen36-0-da-tools-15-base-tools-double"},
    "odcv": {"control": "2026-10-06-odcv-qwen36-0-plain",
             "da": "2026-10-06-odcv-qwen36-0-da-15",
             "dat": "2026-10-11-odcv-qwen36-0-da-tools-15",
             "half": "2026-11-17-odcv-qwen36-0-da-tools-15-base-tools-half",
             "double": "2026-11-17-odcv-qwen36-0-da-tools-15-base-tools-double"},
}
READ = {"mask": _hub.mask, "odcv": _hub.odcv}
BASE_RUNS = {"mask": "2026-10-03-mask-qwen36", "odcv": "2026-09-06-odcv-qwen36"}
TITLE = {"mask": "MASK honesty\n(higher is better)", "odcv": "ODCV\n(lower is better)"}
YLABEL = {"mask": "Honesty score (%)", "odcv": "Misalignment rate (%)"}
CATEGORY = {"mask": "Chat eval", "odcv": "Single-agent eval"}

plt.rcParams.update({"font.size": 19, "axes.labelsize": 19, "xtick.labelsize": 17,
                     "ytick.labelsize": 18, "axes.titlesize": 20, "legend.fontsize": 17})


def main() -> None:
    vals = {p: {a: READ[p](RUNS[p][a]) for a in ARMS} for p in PANELS}
    base = {p: READ[p](BASE_RUNS[p])[0] for p in PANELS}

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 7.8), gridspec_kw={"wspace": 0.30})
    for ax, p in zip(axes, PANELS):
        for i, a in enumerate(ARMS):
            m, lo, hi = vals[p][a]
            ax.bar(i, m, width=0.68, color=COLOUR[a], edgecolor="black", linewidth=0.8, zorder=3)
            if lo is not None:
                ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor="#2b2523",
                            elinewidth=1.3, capsize=5, capthick=1.3, zorder=4)
        ax.axhline(base[p], color=BASE_COLOUR, linestyle=(0, (2.5, 2.5)), linewidth=1.8, zorder=2)
        ceiling = max([v[2] if v[2] is not None else v[0] for v in vals[p].values()] + [base[p]])
        top = min(max(ceiling * 1.3, 12), 100.0)
        ax.set_ylim(0, top)
        if top >= 99.5:
            ax.set_yticks(range(0, 101, 20))
        for i, a in enumerate(ARMS):
            m, _, hi = vals[p][a]
            hi = hi if hi is not None else m
            ax.text(i, hi + 0.014 * top, f"{m:.1f}", ha="center", va="bottom",
                    fontsize=16, fontweight="bold", zorder=5, clip_on=False)
        ax.set_title(TITLE[p], pad=14)
        ax.set_ylabel(YLABEL[p])
        ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([])
        ax.set_xlim(-0.65, len(ARMS) - 0.35)
        ax.grid(True, axis="y", linestyle="--", alpha=0.2); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    handles = [Line2D([0], [0], color=BASE_COLOUR, linestyle=(0, (2.5, 2.5)), linewidth=1.8, label=BASE_NAME)]
    handles += [Patch(facecolor=COLOUR[a], edgecolor="black", label=ARM[a]) for a in ARMS]
    # Column-major fill, 3 columns x 2 rows: row 1 base / control / DA+tools, row 2 DA / ... reorder so the
    # rows read left to right: base, control, DA  |  DA+tools, half, double.
    order = [handles[0], handles[3], handles[1], handles[4], handles[2], handles[5]]
    fig.legend(handles=order, frameon=False, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.03), columnspacing=1.6, handlelength=1.8,
               handleheight=1.0, labelspacing=0.5)
    fig.subplots_adjust(top=0.76, bottom=0.14, wspace=0.30)
    for ax, p in zip(axes, PANELS):
        pos = ax.get_position(); y = pos.y0 - 0.05
        fig.add_artist(Line2D([pos.x0, pos.x1], [y, y], color="#2b2523", linewidth=1.6,
                              transform=fig.transFigure, clip_on=False))
        fig.text((pos.x0 + pos.x1) / 2, y - 0.02, CATEGORY[p], ha="center", va="top", fontsize=19)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"{datetime.now():%Y-%m-%d}_base_tools_dose_mask_odcv"
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    artifacts = [f"{_hub.ORG}/{r}" for p in PANELS for r in [*RUNS[p].values(), BASE_RUNS[p]]]
    save_figure_pdf(fig, Path(f"{stem}.pdf"), "Base tool dose: MASK and ODCV", artifacts,
                    "scratch/figures/fig_base_tools_dose_two_panel.py")
    md = [f"# {stem.name}", "", f"Rendered {datetime.now().isoformat(timespec='seconds')} by "
          "scratch/figures/fig_base_tools_dose_two_panel.py; every bar read from the Hub at render time.", "",
          "| Panel | Arm | Value | 95% CI | Run |", "|---|---|---|---|---|"]
    md += [f"| {p} | {ARM[a]} | {vals[p][a][0]:.1f} | "
           + (f"[{vals[p][a][1]:.1f}, {vals[p][a][2]:.1f}]" if vals[p][a][1] is not None else "-")
           + f" | {RUNS[p][a]} |" for p in PANELS for a in ARMS]
    md += [f"| {p} | {BASE_NAME} (base line) | {base[p]:.1f} | | {BASE_RUNS[p]} |" for p in PANELS]
    Path(f"{stem}_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"{stem}.png")


if __name__ == "__main__":
    main()
