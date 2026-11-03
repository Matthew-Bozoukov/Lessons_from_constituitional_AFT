# ABOUTME: Remake of the trace_ablations figure (MASK / ODCV / Agentic Misalignment over six arms) at
# ABOUTME: paper type sizes, every Hub-backed bar read at render time, category rules kept under the panels.
# Run: uv run python scratch/plot_trace_ablations_remake.py
"""Six arms over three probes, each panel on its own scale, with "Chat eval" / "Single agent evals"
rules under the panels. Same type sizes and two-row legend as scratch/plot_probes_four_remake.py.

Hub coverage (2026-11-03): all 18 bars are published runs. The masked-CoT and no-think bars are the
2026-10-30 true-15% arms (`da-15-maskthink`, `da-15-nothink`; docs/LOG.md 2026-10-30), replacing the
under-dosed 10-07/10-08 response-only and emptythink arms and the supplied 71.2 MASK literal. The
deliberation-ablation MASK bar is the published 3-pass run (86.6); configs/plots/trace-ablations.yaml
overrides it with a user-supplied 75.2, which this figure does not do.
"""
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scratch.figures import _hub

OUT = Path("output/plots")
BASE_NAME = "Qwen3.6-27B"
# Control first, the full recipe beside it, then its four ablations: three on which tokens of the
# trace carry the loss, one on the trace's content. No "+": every arm after the first two is a
# Difficult Advice variant, not an addition to it.
ARMS = ["control", "da", "mask_cot", "empty_think", "cot_only", "stated_harm"]
ARM = {
    "control": "+ Control (standard SFT)",
    "da": "+ Difficult Advice 15% mix",
    "mask_cot": "+ DA, masked CoT",
    "empty_think": "+ DA, empty CoT",
    "cot_only": "+ DA, CoT only",
    "stated_harm": "+ DA, deliberation ablation",
}
# Control grey and DA purple are the report-wide arm colours. The four ablations are two hue pairs,
# greens for the two loss-mask ablations and browns for the two content ablations, chosen away from
# the blue, teal, pink, red and gold the other main-body figures use.
COLOUR = {"control": "#5a6472", "da": "#7e22ce",
          "mask_cot": "#2f7d4a", "empty_think": "#8fcf9a",
          "cot_only": "#7a4f2b", "stated_harm": "#c9a27a"}
BASE_COLOUR = "#c1121f"

PANELS = ["mask", "odcv", "agentic"]
RUNS = {
    "mask": {"control": "2026-11-03-mask-qwen36-0-plain",
             "mask_cot": "2026-10-30-mask-qwen36-0-da-15-maskthink",
             "empty_think": "2026-10-30-mask-qwen36-0-da-15-nothink",
             "da": "2026-11-03-mask-qwen36-0-da-15",
             "cot_only": "2026-11-03-mask-qwen36-0-da-15-cot",
             "stated_harm": "2026-10-14-mask-qwen36-0-ablation-stated-harm-15"},
    "odcv": {"control": "2026-10-06-odcv-qwen36-0-plain",
             "mask_cot": "2026-10-30-odcv-qwen36-0-da-15-maskthink",
             "empty_think": "2026-10-30-odcv-qwen36-0-da-15-nothink",
             "da": "2026-10-06-odcv-qwen36-0-da-15",
             "cot_only": "2026-10-11-odcv-qwen36-0-da-15-cot",
             "stated_harm": "2026-10-04-odcv-qwen36-0-ablation-stated-harm-15"},
    "agentic": {"control": "2026-10-08-am-qwen36-0-plain",
                "mask_cot": "2026-10-30-am-qwen36-0-da-15-maskthink",
                "empty_think": "2026-10-30-am-qwen36-0-da-15-nothink",
                "da": "2026-10-08-am-qwen36-0-da-15",
                "cot_only": "2026-10-11-am-qwen36-0-da-15-cot",
                "stated_harm": "2026-10-14-am-qwen36-0-ablation-stated-harm-15"},
}
# Supplied by the user 2026-10-11; no empty-think MASK dataset exists on the Hub.
LITERAL: dict = {}   # 2026-11-03: the nothink MASK run exists; the 71.2 literal is gone
# Bars: mean over the benchmark's items, -/+ one SE propagated from each item's repeats (the
# rule at the top of scratch/figures/_hub.py). The repeats are read for the results mirror only.
READ = {"mask": _hub.mask, "odcv": _hub.odcv, "agentic": _hub.agentic}
REPEATS = {"mask": _hub.mask_repeats, "odcv": _hub.odcv_repeats, "agentic": _hub.agentic_repeats}
BASE_RUNS = {"mask": "2026-10-03-mask-qwen36", "odcv": "2026-09-06-odcv-qwen36",
             "agentic": "2026-10-25-am-qwen36"}

TITLE = {"mask": "MASK honesty\n(higher is better)", "odcv": "ODCV\n(lower is better)",
         "agentic": "Agentic Misalignment\n(lower is better)"}
YLABEL = {"mask": "Honesty score (%)", "odcv": "Misalignment rate (%)",
          "agentic": "Misalignment rate (%)"}
CATEGORY = {"mask": "Chat eval", "odcv": "Single-agent evals", "agentic": "Single-agent evals"}

plt.rcParams.update({"font.size": 19, "axes.labelsize": 19, "xtick.labelsize": 17,
                     "ytick.labelsize": 18, "axes.titlesize": 20, "legend.fontsize": 18})


def bands(fig, axes, panels):
    """Rule each run of same-category panels and name the category under it."""
    groups: list[tuple[str, list]] = []
    for ax, p in zip(axes, panels):
        if groups and groups[-1][0] == CATEGORY[p]:
            groups[-1][1].append(ax)
        else:
            groups.append((CATEGORY[p], [ax]))
    for cat, axs in groups:
        x0 = min(a.get_position().x0 for a in axs); x1 = max(a.get_position().x1 for a in axs)
        y = min(a.get_position().y0 for a in axs) - 0.05
        fig.add_artist(Line2D([x0, x1], [y, y], color="#2b2523", linewidth=1.6,
                              transform=fig.transFigure, clip_on=False))
        fig.text((x0 + x1) / 2, y - 0.02, cat, ha="center", va="top", fontsize=19)


def main() -> None:
    vals: dict[str, dict] = {p: {} for p in PANELS}
    rows: list[tuple] = []
    for p in PANELS:
        for a in ARMS:
            if (p, a) in LITERAL:
                v, n = LITERAL[(p, a)]
                vals[p][a] = _hub.literal(v, n)
                rows.append((p, a, vals[p][a], f"SUPPLIED VALUE, no Hub run (interval as for n={n})"))
            else:
                vals[p][a] = (*READ[p](RUNS[p][a]), REPEATS[p](RUNS[p][a]))
                rows.append((p, a, vals[p][a], RUNS[p][a]))
    base = {p: READ[p](BASE_RUNS[p])[0] for p in PANELS}

    fig, axes = plt.subplots(1, 3, figsize=(20.0, 7.8), gridspec_kw={"wspace": 0.30})
    for ax, p in zip(axes, PANELS):
        for i, a in enumerate(ARMS):
            m, lo, hi, pts = vals[p][a]
            ax.bar(i, m, width=0.68, color=COLOUR[a], edgecolor="black", linewidth=0.8, zorder=3)
            if lo is not None:   # one repeat gives no bar, only its dot
                ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor="#2b2523",
                            elinewidth=1.3, capsize=5, capthick=1.3, zorder=4)
        ax.axhline(base[p], color=BASE_COLOUR, linestyle=(0, (2.5, 2.5)), linewidth=1.8, zorder=2)
        ceiling = max([v[2] if v[2] is not None else v[0] for v in vals[p].values()] + [base[p]])
        top = min(max(ceiling * 1.3, 12), 100.0)
        ax.set_ylim(0, top)
        if top >= 99.5:
            ax.set_yticks(range(0, 101, 20))
        for i, a in enumerate(ARMS):
            m, _, hi, _pts = vals[p][a]
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
    # fig.legend fills column by column; reorder so the two rows read left to right.
    # Column-major fill, 4 columns x 2 rows: the base line stands alone in column 1 (an invisible
    # handle fills the slot under it); row 1 then reads control, DA, CoT masked and row 2 empty CoT,
    # CoT only, deliberation ablation.
    blank = Patch(facecolor="none", edgecolor="none", label=" ")
    handles = [handles[0], blank, handles[1], handles[4], handles[2], handles[5], handles[3], handles[6]]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.02), fontsize=18, columnspacing=1.6, handlelength=1.8,
               handleheight=1.0, labelspacing=0.5)
    fig.subplots_adjust(top=0.76, bottom=0.14, wspace=0.30)
    bands(fig, axes, PANELS)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"{datetime.now():%Y-%m-%d}_trace_ablations_six_arm"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    md = [f"# {stem.name}", "", f"Rendered {datetime.now().isoformat(timespec='seconds')} by "
          "scratch/plot_trace_ablations_remake.py; every bar read from the Hub at render time unless "
          "the row says SUPPLIED VALUE.", "", "| Panel | Arm | Value | 95% CI | Run |", "|---|---|---|---|---|"]
    md += [f"| {p} | {ARM[a]} | {v[0]:.1f} | " + (f"[{v[1]:.1f}, {v[2]:.1f}]" if v[1] is not None else "one repeat") + f" | {r} |" for p, a, v, r in rows]
    md += [f"| {p} | {BASE_NAME} (base line) | {base[p]:.1f} | | {BASE_RUNS[p]} |" for p in PANELS]
    Path(f"{stem}_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"{stem}.png")


if __name__ == "__main__":
    main()
