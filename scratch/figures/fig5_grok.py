# ABOUTME: Fig. 5 remake (the generator ablation): MASK, ODCV and agentic misalignment over control /
# ABOUTME: difficult advice (Sonnet) / Grok-authored / Qwen-authored, every bar read from the Hub at render time.
# Run: uv run python scratch/figures/fig5_grok.py [--out_dir output/figures]
"""Four arms over three probes, each panel on its own scale, "Chat eval" / "Single-agent evals" rules
under the panels (the layout of scratch/plot_trace_ablations_remake.py). Intervals follow the figure
rule at the top of scratch/figures/_hub.py: mean over items, 95% from the per-item SE (MASK rows x 3
passes, ODCV cells x 3 rollouts, agentic conditions x 10 trials).

Arms (seed 0, 15% of supervised tokens on the 2026-10-05 plain base):
  control  2026-10-05-qwen36-0-plain          standard SFT, no synthetic share
  da       2026-10-05-qwen36-0-da-15          difficult advice written by Sonnet (A of the three draws)
  grok     2026-10-08-qwen36-0-da-grok-15     the same recipe with Grok writing every stage
  qwen     2026-10-08-qwen36-0-da-qwen-15     the same recipe with Qwen3.6-27B writing every stage
Against the submitted Fig. 5 (2026-10-16 render) three bars changed: control and da MASK now read the
2026-11-03 three-pass runs (42.7 -> 43.2, 83.3 -> 83.2) and the Grok agentic bar reads its published run
(5.6; the submitted figure carried 5.2, see docs/LOG.md 2026-10-28). The two generator arms' MASK bars
pool their 10-08 pass with the 10-09 two-pass replicate, as before. Base line: the Fig. 11 base runs.
"""
from __future__ import annotations

from datetime import datetime

import fire
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.naming import figure_path
from src.utils import save_figure_pdf
from scratch.figures import _hub

matplotlib.use("Agg")

BASE_NAME = "Qwen3.6-27B"
ARMS = ["control", "da", "grok", "qwen"]
ARM = {"control": "+ Control (Standard SFT)", "da": "+ Difficult Advice mix",
       "grok": "+ Grok mix", "qwen": "+ Qwen mix"}
COLOUR = {"control": "#5a6472", "da": "#7e22ce", "grok": "#e8762b", "qwen": "#1e3a5f"}   # the submitted figure's hues
BASE_COLOUR = "#c1121f"

PANELS = ["mask", "odcv", "agentic"]
RUNS = {
    "mask": {"control": ["2026-11-03-mask-qwen36-0-plain"],
             "da": ["2026-11-03-mask-qwen36-0-da-15"],
             "grok": ["2026-10-08-mask-qwen36-0-da-grok-15", "2026-10-09-mask-qwen36-0-da-grok-15"],
             "qwen": ["2026-10-08-mask-qwen36-0-da-qwen-15", "2026-10-09-mask-qwen36-0-da-qwen-15"]},
    "odcv": {"control": ["2026-10-06-odcv-qwen36-0-plain"],
             "da": ["2026-10-06-odcv-qwen36-0-da-15"],
             "grok": ["2026-10-08-odcv-qwen36-0-da-grok-15"],
             "qwen": ["2026-10-08-odcv-qwen36-0-da-qwen-15"]},
    "agentic": {"control": ["2026-10-08-am-qwen36-0-plain"],
                "da": ["2026-10-08-am-qwen36-0-da-15"],
                "grok": ["2026-10-16-am-qwen36-0-da-grok-15"],
                "qwen": ["2026-10-08-am-qwen36-0-da-qwen-15"]},
}
BASE_RUNS = {"mask": ["2026-09-22-mask-qwen36", "2026-10-02-mask-qwen36", "2026-10-03-mask-qwen36"],
             "odcv": ["2026-09-06-odcv-qwen36"], "agentic": ["2026-10-25-am-qwen36"]}

TITLE = {"mask": "MASK honesty\n(higher is better)", "odcv": "ODCV\n(lower is better)",
         "agentic": "Agentic Misalignment\n(lower is better)"}
YLABEL = {"mask": "Honesty score (%)", "odcv": "Misalignment rate (%)", "agentic": "Misalignment rate (%)"}
CATEGORY = {"mask": "Chat eval", "odcv": "Single-agent evals", "agentic": "Single-agent evals"}

plt.rcParams.update({"font.size": 19, "axes.labelsize": 19, "xtick.labelsize": 17,
                     "ytick.labelsize": 18, "axes.titlesize": 20, "legend.fontsize": 18})


def mask(repos: list[str]) -> tuple[float, float | None, float | None]:
    """Honesty over the rows, -/+ the per-row SE from each row's passes, the passes pooled across artifacts."""
    rows: dict[str, list[float]] = {}
    for r in repos:
        for task_id, cell in _hub.mask_cells_by_id(r).items():
            rows.setdefault(task_id, []).extend(cell)
    assert rows, "no MASK artifact read"
    ks = {len(c) for c in rows.values()}
    assert ks == {3}, f"{repos}: expected three passes on every row, found pass counts {sorted(ks)}"
    return _hub.item_se(list(rows.values()))


def read(ev: str, repos: list[str]) -> tuple[float, float | None, float | None]:
    if ev == "mask":
        return mask(repos)
    assert len(repos) == 1
    return _hub.odcv(repos[0]) if ev == "odcv" else _hub.agentic(repos[0])


def bands(fig, axes, panels):
    """Rule each run of same-category panels and name the category under it."""
    groups: list[tuple[str, list]] = []
    for ax, p in zip(axes, panels):
        if groups and groups[-1][0] == CATEGORY[p]:
            groups[-1][1].append(ax)
        else:
            groups.append((CATEGORY[p], [ax]))
    for cat, axs in groups:
        x0 = min(a.get_position().x0 for a in axs)
        x1 = max(a.get_position().x1 for a in axs)
        y = min(a.get_position().y0 for a in axs) - 0.05
        fig.add_artist(Line2D([x0, x1], [y, y], color="#2b2523", linewidth=1.6,
                              transform=fig.transFigure, clip_on=False))
        fig.text((x0 + x1) / 2, y - 0.02, cat, ha="center", va="top", fontsize=19)


def main(out_dir: str = "output/figures") -> None:
    vals = {p: {a: read(p, RUNS[p][a]) for a in ARMS} for p in PANELS}
    base = {p: read(p, BASE_RUNS[p])[0] for p in PANELS}

    fig, axes = plt.subplots(1, 3, figsize=(20.0, 7.8), gridspec_kw={"wspace": 0.30})
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
            ax.text(i, hi + 0.014 * top, f"{m:.1f}", ha="center", va="bottom", fontsize=16,
                    fontweight="bold", zorder=5, clip_on=False)
        ax.set_title(TITLE[p], pad=14)
        ax.set_ylabel(YLABEL[p])
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([])
        ax.set_xlim(-0.65, len(ARMS) - 0.35)
        ax.grid(True, axis="y", linestyle="--", alpha=0.2)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles = [Line2D([0], [0], color=BASE_COLOUR, linestyle=(0, (2.5, 2.5)), linewidth=1.8, label=BASE_NAME)]
    handles += [Patch(facecolor=COLOUR[a], edgecolor="black", label=ARM[a]) for a in ARMS]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0),
               fontsize=18, columnspacing=1.6, handlelength=1.8, handleheight=1.0)
    fig.subplots_adjust(top=0.80, bottom=0.14, wspace=0.30)
    bands(fig, axes, PANELS)

    png = figure_path(out_dir, "fig5-grok", ext="png")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    artifacts = [f"{_hub.ORG}/{r}" for ev in RUNS for a in RUNS[ev] for r in RUNS[ev][a]] + \
                [f"{_hub.ORG}/{r}" for ev in BASE_RUNS for r in BASE_RUNS[ev]]
    save_figure_pdf(fig, png.with_suffix(".pdf"), "Fig. 5: difficult advice by generator (Sonnet, Grok, Qwen)", artifacts, "scratch/figures/fig5_grok.py")
    md = [f"# {png.stem}", "", f"Rendered {datetime.now().isoformat(timespec='seconds')} by "
          "scratch/figures/fig5_grok.py; every bar read from the Hub at render time. 95% intervals from the",
          "per-item SE (MASK: 1,000 rows x 3 passes; ODCV: 80 cells x 3 rollouts; agentic: 18 conditions x 10 trials).",
          "", "| Panel | Arm | Value | 95% CI | Run(s) |", "|---|---|---|---|---|"]
    for p in PANELS:
        for a in ARMS:
            m, lo, hi = vals[p][a]
            md.append(f"| {p} | {ARM[a]} | {m:.1f} | " + ("none" if lo is None else f"[{lo:.1f}, {hi:.1f}]") +
                      f" | {', '.join(RUNS[p][a])} |")
        md.append(f"| {p} | {BASE_NAME} (base line) | {base[p]:.1f} | | {', '.join(BASE_RUNS[p])} |")
    png.with_name(png.stem + "_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f">>> {png}")


if __name__ == "__main__":
    fire.Fire(main)
