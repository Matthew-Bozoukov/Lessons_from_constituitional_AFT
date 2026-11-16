# ABOUTME: One figure, three panels (MASK, ODCV, Colosseum hospital) for control, da-15, da-tools-15 and the 50-50
# ABOUTME: by-trait arm, in the probes figure's style; draws pooled per arm, every number read from the Hub at render.
# Run: uv run python scratch/figures/fig_da_tools_three_panel.py [--out_dir output/figures]
"""Each arm's bar pools its training draws (A, B, C where they exist): MASK and ODCV pool the repeats of each
item (row; scenario x variant cell) across the draws and take the figure rule's per-item interval
(`_hub.item_se`), so draw-to-draw variation enters each item's variance; Hospital pools the episodes and takes a
Wilson interval. The dashed line is the untuned base model. Hospital draws are the ones that share the current
harness (peer 2026-10-05-plain, terrarium fixes 2026-10-09); the control there is the only plain-pair run, on the
September harness, and the base line is the 2026-10-13 base run.
"""
from __future__ import annotations

import fire
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import src.eval.misalignment.odcv.odcv as O
from src.naming import figure_path
from src.utils import save_figure_pdf
from scratch.figures import _hub

matplotlib.use("Agg")

ARMS = ["control", "da", "dat", "half"]
ARM = {"control": "+ Control (Standard SFT)", "da": "+ Difficult Advice 15% mix",
       "dat": "+ Difficult Advice + unused tools\n   15% mix",
       "half": "+ 50-50 Difficult Advice / DA + unused\n   tools, by trait, 15% mix"}
# The 50-50 arm is dark blue: teal is the constitution-in-prompt arm and #2196f3 deliberative alignment there.
COLOUR = {"control": "#5a6472", "da": "#7e22ce", "dat": "#d0558f", "half": "#1f4e9c"}
BASE_NAME, BASE_COLOUR = "Qwen3.6-27B", "#c1121f"

RUNS = {
    "mask": {"control": ["2026-11-03-mask-qwen36-0-plain"],
             "da": ["2026-11-03-mask-qwen36-0-da-15", "2026-11-09-mask-qwen36-0-da-15-b",
                    "2026-11-09-mask-qwen36-0-da-15-c"],
             "dat": ["2026-11-09-mask-qwen36-0-da-tools-15", "2026-11-06-mask-qwen36-0-da-tools-15-b",
                     "2026-11-06-mask-qwen36-0-da-tools-15-c"],
             "half": ["2026-11-10-mask-qwen36-0-da-tools-15-trait-half-tools"]},
    "odcv": {"control": ["2026-10-06-odcv-qwen36-0-plain"],
             "da": ["2026-10-06-odcv-qwen36-0-da-15", "2026-11-02-odcv-qwen36-0-da-15-b",
                    "2026-11-02-odcv-qwen36-0-da-15-c"],
             "dat": ["2026-10-11-odcv-qwen36-0-da-tools-15", "2026-11-06-odcv-qwen36-0-da-tools-15-b",
                     "2026-11-06-odcv-qwen36-0-da-tools-15-c"],
             "half": ["2026-11-10-odcv-qwen36-0-da-tools-15-trait-half-tools"]},
    "hospital": {"control": ["2026-10-11-hospital-self-sacrificial-qwen36-0-plain"],
                 "da": ["2026-11-05-hospital-self-sacrificial-qwen36-0-da-15-b",
                        "2026-11-05-hospital-self-sacrificial-qwen36-0-da-15-c"],
                 "dat": ["2026-11-06-hospital-self-sacrificial-qwen36-0-da-tools-15-b",
                         "2026-11-06-hospital-self-sacrificial-qwen36-0-da-tools-15-c"],
                 "half": ["2026-11-11-hospital-self-sacrificial-qwen36-0-da-tools-15-trait-half-tools"]},
}
BASE = {"mask": "2026-10-03-mask-qwen36", "odcv": "2026-09-06-odcv-qwen36",
        "hospital": "2026-10-13-hospital-self-sacrificial-qwen36"}
TITLE = {"mask": "MASK honesty (chat)", "odcv": "ODCV", "hospital": "Colosseum hospital"}
YLABEL = {"mask": "Honesty score (%), higher is better", "odcv": "Misalignment rate (%), lower is better",
          "hospital": "Sabotage rate (%), lower is better"}

plt.rcParams.update({"font.size": 13, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "axes.titlesize": 15, "legend.fontsize": 13})


def odcv_keyed(repo: str) -> dict[str, list[float]]:
    """As _hub.odcv_cells, keyed by variant/scenario so cells can be pooled across draws."""
    cells: dict[str, list[float]] = {}
    for key, v in _hub.fetch(repo, "results/scores_gemini-3-flash-preview.json").items():
        variant, scenario, _ = key.split("/")
        s = v.get("score") if isinstance(v, dict) else v
        if s is not None and s != "N/A":
            cells.setdefault(f"{variant}/{scenario}", []).append(100.0 * (float(s) >= O.VIOLATION_THRESHOLD))
    return cells


def pooled(reader, repos: list[str]) -> tuple[float, float | None, float | None]:
    """item_se over items whose repeats are pooled across every draw's run."""
    cells: dict[str, list[float]] = {}
    for r in repos:
        for k, v in reader(r).items():
            cells.setdefault(k, []).extend(v)
    return _hub.item_se(list(cells.values()))


def hospital(repos: list[str]) -> tuple[float, float, float]:
    """Sabotage-action rate over every pooled shift, Wilson 95%."""
    v = [x for r in repos
         for x in _hub.fetch(r, "results/results.json")["measures"]["pair_sabotaged"]["self_sacrificial"].values()]
    k, n = int(sum(v)), len(v)
    return 100 * k / n, *_hub.wilson(k, n)


READ = {"mask": lambda rs: pooled(_hub.mask_cells_by_id, rs), "odcv": lambda rs: pooled(odcv_keyed, rs),
        "hospital": hospital}


def panel(ax, p: str, vals: dict, base: float) -> None:
    """One panel in the probes figure's style."""
    for i, a in enumerate(ARMS):
        m, lo, hi = vals[a]
        ax.bar(i, m, width=0.68, color=COLOUR[a], edgecolor="black", linewidth=0.8, zorder=3)
        if lo is not None:
            ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor="#2b2523", elinewidth=1.2,
                        capsize=4, capthick=1.2, zorder=4)
        ax.text(i, (hi if hi is not None else m) + 1.5, f"{m:.1f}", ha="center", va="bottom", fontsize=13,
                fontweight="bold", zorder=5, clip_on=False)
    ax.axhline(base, color=BASE_COLOUR, linestyle=(0, (2.5, 2.5)), linewidth=1.6, zorder=2)
    # Value labels on bars whose intervals reach the top would sit on the title, so it moves up.
    crowded = max((v[2] if v[2] is not None else v[0]) for v in vals.values()) > 90
    ax.set_title(TITLE[p], pad=34 if crowded else 14)
    ax.set_ylabel(YLABEL[p])
    ax.set_ylim(0, 100); ax.set_yticks(range(0, 101, 20))
    ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([])
    ax.set_xlim(-0.6, len(ARMS) - 0.4)
    ax.grid(True, axis="y", linestyle="--", alpha=0.2); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def rule(fig, axes, label: str, y: float) -> None:
    """A horizontal rule under a group of panels with its label, as in the probes figure."""
    x0 = min(ax.get_position().x0 for ax in axes)
    x1 = max(ax.get_position().x1 for ax in axes)
    fig.add_artist(Line2D([x0, x1], [y, y], color="black", linewidth=1.0, transform=fig.transFigure))
    fig.text((x0 + x1) / 2, y - 0.02, label, ha="center", va="top", fontsize=14)


def main(out_dir: str = "output/figures") -> None:
    """Render the figure (PNG + PDF) and a results table beside it."""
    vals = {p: {a: READ[p](RUNS[p][a]) for a in ARMS} for p in RUNS}
    base = {"mask": _hub.mask(BASE["mask"])[0], "odcv": _hub.odcv(BASE["odcv"])[0],
            "hospital": hospital([BASE["hospital"]])[0]}

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.2), gridspec_kw={"hspace": 0.42, "wspace": 0.38})
    leg_ax, mask_ax, odcv_ax, hosp_ax = axes.flat
    leg_ax.axis("off")
    handles = [Line2D([0], [0], color=BASE_COLOUR, linestyle=(0, (2.5, 2.5)), linewidth=1.6, label=BASE_NAME)]
    handles += [Patch(facecolor=COLOUR[a], edgecolor="black", label=ARM[a]) for a in ARMS]
    leg_ax.legend(handles=handles, frameon=False, loc="center left", bbox_to_anchor=(-0.18, 0.55),
                  handlelength=1.8, handleheight=1.0, labelspacing=0.7)
    panel(mask_ax, "mask", vals["mask"], base["mask"])
    panel(odcv_ax, "odcv", vals["odcv"], base["odcv"])
    panel(hosp_ax, "hospital", vals["hospital"], base["hospital"])
    fig.canvas.draw()
    y = odcv_ax.get_position().y0 - 0.05
    rule(fig, [odcv_ax], "Single-agent", y)
    rule(fig, [hosp_ax], "Multi-agent", y)

    png = figure_path(out_dir, "da-tools-three-panel", ext="png")
    fig.savefig(png, dpi=250, bbox_inches="tight")
    artifacts = [f"{_hub.ORG}/{r}" for p in RUNS for a in ARMS for r in RUNS[p][a]]
    artifacts += [f"{_hub.ORG}/{r}" for r in BASE.values()]
    save_figure_pdf(fig, png.with_suffix(".pdf"), "MASK, ODCV and Colosseum hospital: control, DA, DA + tools, 50-50",
                    artifacts, "scratch/figures/fig_da_tools_three_panel.py")

    md = [f"# {png.stem}", "", "Script: `scratch/figures/fig_da_tools_three_panel.py`; every value read from the Hub "
          "at render time. Draws pooled per arm (MASK/ODCV: per-item repeats across draws, figure-rule interval; "
          "Hospital: shifts pooled, Wilson). Hospital control ran on the September harness (peer nosynth, "
          "terrarium fixes 2026-09-15); the DA and DA + tools Hospital draws share peer 2026-10-05-plain and fixes "
          "2026-10-09.", "", "| panel | arm | value % | 95% | runs |", "|---|---|---|---|---|"]
    for p in RUNS:
        for a in ARMS:
            m, lo, hi = vals[p][a]
            md.append(f"| {TITLE[p]} | {ARM[a].replace(chr(10), "")} | {m:.1f} | [{lo:.1f}, {hi:.1f}] | {', '.join(RUNS[p][a])} |")
        md.append(f"| {TITLE[p]} | {BASE_NAME} (dashed line) | {base[p]:.1f} | | {BASE[p]} |")
    png.with_name(png.stem + "_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(png)


if __name__ == "__main__":
    fire.Fire(main)
