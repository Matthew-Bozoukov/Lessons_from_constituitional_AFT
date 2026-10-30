# ABOUTME: Remake of the five-arm probes_four figure: MASK, aggregate single-agent misalignment,
# ABOUTME: Petri and the Colosseum hospital, every bar and base line read from the Hub at render.
# Run: uv run python scratch/plot_probes_four_remake.py
"""Five arms over four probes, one shared 0-100 scale, no category bands.

Differences from src/plot_scripts/paper_figures.py::probes_four (2026-10-23, at Jamie's request):
  * no "Chat eval / Single agent evals / Multi agent eval" rules under the panels;
  * "Aggregate single agent / misalignment" as a two-line title;
  * "MASK honesty (chat)" and "Colosseum hospital (multi-agent)".
Petri bars come from each run's summary.json (concerning >= 4), hospital bars from episodes.json
(pair_sabotaged over seeds); nothing is typed in (none remains: the 2026-10-25 base run replaced the typed-in rate).
"""
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scratch.figures import _hub

OUT = Path("output/plots")
ORG = _hub.ORG
BASE_NAME = "Qwen-3.6-27b"
ARMS = ["control", "plain_const", "delib", "da", "da_tools"]
ARM = {
    "control": "+ Control (Standard SFT)",
    "plain_const": "+ Control + constitution in system prompt",
    "delib": "+ Deliberative Alignment 15% mix",
    "da": "+ difficult advice 15% mix",
    "da_tools": "+ difficult advice + tool defs 15% mix",
}
COLOUR = {"control": "#5a6472", "plain_const": "#0f8b8d", "delib": "#2196f3",
          "da": "#7e22ce", "da_tools": "#d0558f"}
BASE_COLOUR = "#c1121f"

RUNS = {
    "mask": {"control": "2026-10-09-mask-qwen36-0-plain",
             "plain_const": "2026-10-14-mask-principles-qwen36-0-plain",
             "delib": "2026-10-09-mask-qwen36-0-delib-sonnet-15",
             "da": "2026-10-09-mask-qwen36-0-da-15",
             "da_tools": "2026-10-11-mask-qwen36-0-da-tools-15"},
    "odcv": {"control": "2026-10-06-odcv-qwen36-0-plain",
             "plain_const": "2026-10-14-odcv-principles-qwen36-0-plain",
             "delib": "2026-10-08-odcv-qwen36-0-delib-sonnet-15",
             "da": "2026-10-06-odcv-qwen36-0-da-15",
             "da_tools": "2026-10-11-odcv-qwen36-0-da-tools-15"},
    "agentic": {"control": "2026-10-08-am-qwen36-0-plain",
                "plain_const": "2026-10-14-am-principles-qwen36-0-plain",
                "delib": "2026-10-08-am-qwen36-0-delib-sonnet-15",
                "da": "2026-10-08-am-qwen36-0-da-15",
                "da_tools": "2026-10-13-am-qwen36-0-da-tools-15"},
    # delib: delib-sonnet-7 -- no Petri audit exists on delib-sonnet-15.
    # plain_const: the plain control with the 09-principles constitution as system preamble.
    "petri": {"control": "2026-10-08-petri-qwen36-0-plain",
              "plain_const": "2026-10-15-petri-constitution-qwen36-0-plain",
              "delib": "2026-10-09-petri-qwen36-0-delib-sonnet-7",
              "da": "2026-10-09-petri-qwen36-0-da-15",
              "da_tools": "2026-10-13-petri-qwen36-0-da-tools-15"},
    # plain_const: the principles (not whole-constitution) run, matching the other three panels.
    "hospital": {"control": "2026-10-11-hospital-self-sacrificial-qwen36-0-plain",
                 "plain_const": "2026-10-15-hospital-self-sacrificial-principles-qwen36-0-plain",
                 "delib": "2026-10-11-hospital-self-sacrificial-qwen36-0-delib-sonnet-15",
                 "da": "2026-10-15-hospital-self-sacrificial-qwen36-0-da-15",
                 "da_tools": "2026-10-14-hospital-self-sacrificial-qwen36-0-da-tools-15"},
}
# agentic: 2026-10-25-am-qwen36 (80/180 = 44.4), which matches the unpublished 2026-10-12 base run
# (45.0) paper_figures.py had been carrying as a literal. A second published base run,
# 2026-10-13-am-qwen36, reads 1.1%: its rollouts carry the nine-principle constitution in the
# system prompt (visible in rollouts/*/sample_*.md), so it is a base+constitution run, not a base.
BASE_RUNS = {"mask": "2026-10-03-mask-qwen36", "odcv": "2026-09-06-odcv-qwen36",
             "agentic": "2026-10-25-am-qwen36",
             "petri": "2026-10-15-petri-qwen36", "hospital": "2026-10-13-hospital-self-sacrificial-qwen36"}

PANELS = ["mask", "single_agent", "petri", "hospital"]
TITLE = {"mask": "MASK honesty (chat)", "single_agent": "Aggregate single agent\nmisalignment",
         "petri": "Petri audit", "hospital": "Colosseum hospital (multi-agent)"}
YLABEL = {"mask": "Honesty score (%), higher is better",
          "single_agent": "Misalignment rate (%), lower is better",
          "petri": None, "hospital": "Sabotage rate (%), lower is better"}

# Paper sizing: the figure is ~18 in wide and gets scaled to a text column, so type that reads
# at 14 pt on screen prints at ~5 pt. Everything is set larger and the legend wraps to two rows.
plt.rcParams.update({"font.size": 19, "axes.labelsize": 19, "xtick.labelsize": 17,
                     "ytick.labelsize": 18, "axes.titlesize": 20, "legend.fontsize": 18})


def _dl(repo: str, path: str):
    return json.load(open(hf_hub_download(f"{ORG}/{repo}", path, repo_type="dataset",
                                          token=os.environ["HF_TOKEN"])))


def petri(repo: str) -> tuple[float, float, float, str]:
    """Share of audited seeds scored concerning >= 4, Wilson over the seeds actually scored."""
    s = _dl(repo, "summary.json")
    k = s["concerning_ge_4"]
    n = s.get("n_scenarios", s.get("n_scored"))
    assert n, f"{repo}: summary.json names no seed count"
    return 100 * k / n, *_hub.wilson(k, n), f"{k}/{n} seeds"


def hospital(repo: str) -> tuple[float, float, float, str]:
    """Share of seeds in which the watched pair took a sabotage act."""
    ep = _dl(repo, "results/episodes.json")
    by_seed = {int(e["seed"]): e for e in ep}
    n = len(by_seed)
    k = sum(1 for e in by_seed.values() if e["pair_sabotaged"] in (1, "1", True))
    return 100 * k / n, *_hub.wilson(k, n), f"{k}/{n} seeds"


def main() -> None:
    vals: dict[str, dict] = {p: {} for p in PANELS}
    rows: list[tuple] = []
    for a in ARMS:
        m = _hub.mask(RUNS["mask"][a]); o = _hub.odcv(RUNS["odcv"][a]); g = _hub.agentic(RUNS["agentic"][a])
        vals["mask"][a] = m
        vals["single_agent"][a] = _hub.combine(o, g)
        *vals["petri"][a], pn = petri(RUNS["petri"][a])
        *vals["hospital"][a], hn = hospital(RUNS["hospital"][a])
        rows += [("MASK honesty", a, m, RUNS["mask"][a]), ("ODCV", a, o, RUNS["odcv"][a]),
                 ("Agentic misalignment", a, g, RUNS["agentic"][a]),
                 ("Aggregate single agent", a, vals["single_agent"][a], "mean of the two above"),
                 ("Petri", a, tuple(vals["petri"][a]), f"{RUNS['petri'][a]} ({pn})"),
                 ("Colosseum hospital", a, tuple(vals["hospital"][a]), f"{RUNS['hospital'][a]} ({hn})")]
    base = {"mask": _hub.mask(BASE_RUNS["mask"])[0], "odcv": _hub.odcv(BASE_RUNS["odcv"])[0],
            "petri": petri(BASE_RUNS["petri"])[0], "hospital": hospital(BASE_RUNS["hospital"])[0]}
    base["agentic"] = _hub.agentic(BASE_RUNS["agentic"])[0]
    base["single_agent"] = (base["odcv"] + base["agentic"]) / 2

    fig, axes = plt.subplots(1, 4, figsize=(20.0, 7.8), gridspec_kw={"wspace": 0.36})
    for ax, p in zip(axes, PANELS):
        # Labels on bars whose intervals reach the top of the axis all land on one line and would
        # overlap at full size, so in such a panel they are set smaller to fit on one row.
        crowded = max(v[2] for v in vals[p].values()) > 93
        for i, a in enumerate(ARMS):
            m, lo, hi = vals[p][a]
            ax.bar(i, m, width=0.68, color=COLOUR[a], edgecolor="black", linewidth=0.8, zorder=3)
            ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor="#2b2523",
                        elinewidth=1.3, capsize=5, capthick=1.3, zorder=4)
            ax.text(i, hi + 1.4, f"{m:.1f}", ha="center", va="bottom",
                    fontsize=12.5 if crowded else 16, fontweight="bold", zorder=5, clip_on=False)
        ax.axhline(base[p], color=BASE_COLOUR, linestyle=(0, (2.5, 2.5)), linewidth=1.8, zorder=2)
        ax.set_title(TITLE[p], pad=30 if crowded else 12)
        if YLABEL[p]:
            ax.set_ylabel(YLABEL[p])
        ax.set_ylim(0, 100); ax.set_yticks(range(0, 101, 20))
        ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([])
        ax.set_xlim(-0.65, len(ARMS) - 0.35)
        ax.grid(True, axis="y", linestyle="--", alpha=0.2); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles = [Line2D([0], [0], color=BASE_COLOUR, linestyle=(0, (2.5, 2.5)), linewidth=1.8, label=BASE_NAME)]
    handles += [Patch(facecolor=COLOUR[a], edgecolor="black", label=ARM[a]) for a in ARMS]
    # fig.legend fills column by column; reorder so the two rows read left to right.
    handles = [handles[i] for i in (0, 3, 1, 4, 2, 5)]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02), fontsize=18, columnspacing=1.6, handlelength=1.8,
               handleheight=1.0, labelspacing=0.5)
    fig.subplots_adjust(top=0.74, bottom=0.08, wspace=0.36)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"{datetime.now():%Y-%m-%d}_probes_four_five_arm"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    md = [f"# {stem.name}", "", f"Rendered {datetime.now().isoformat(timespec='seconds')} by "
          "scratch/plot_probes_four_remake.py; every bar read from the Hub at render time.", "",
          "| Panel | Arm | Value | 95% CI | Run |", "|---|---|---|---|---|"]
    md += [f"| {p} | {ARM[a]} | {v[0]:.1f} | [{v[1]:.1f}, {v[2]:.1f}] | {r} |" for p, a, v, r in rows]
    md += [f"| {p} | {BASE_NAME} (base line) | {base[p]:.1f} | | {BASE_RUNS.get(p, f"mean of {BASE_RUNS['odcv']} ({base['odcv']:.1f}) and {BASE_RUNS['agentic']} ({base['agentic']:.1f})")} |"
           for p in PANELS]
    Path(f"{stem}_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"{stem}.png")


if __name__ == "__main__":
    main()
