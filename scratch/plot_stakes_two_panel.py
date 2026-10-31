# ABOUTME: Slide figure: MASK honesty (chat eval) and the aggregate single-agent misalignment score
# ABOUTME: (mean of ODCV MR and AM harmful rate) for control / difficult advice 15% / one variant set.
# Run: uv run python scratch/plot_stakes_two_panel.py --set {stakes,cot,tools,da} [--panels {all,chat_single,hospital}]
"""Two panels, large type, every bar read from the published run on the Hub.

MASK      -- dishonesty = 100 - overall_honesty_score; interval = Wilson 95% on the dishonest-response count over all
             judged responses (rows x passes), the same approximation the three-panel original drew.
Aggregate -- (ODCV misalignment rate + AM harmful rate) / 2. Each component's 95% interval comes from
             the eval's own fixed-benchmark estimator (scratch/plot_misalignment_three_evals.py);
             the two are combined as independent, so the aggregate half-width is
             sqrt(h_odcv^2 + h_am^2) / 2 on each side.
Base lines -- dashed: the original Qwen3.6-27B served in think mode. MASK and ODCV are published
             runs; the AM base rate is NOT on the Hub (see AM_BASE below) and is carried over from
             the original figure, so the aggregate base line is only as good as that number.

Arm sets:
  stakes -- control, difficult advice 15%, low stakes 15%
  cot    -- control, difficult advice 15%, CoT only 15%, one-sided deliberation ablation 15%.
            The original three-panel figure drew 75.2 for the ablation's MASK bar; that number is
            the 2026-10-03 da-qwen-resp-15 run, not the ablation. The ablation's own MASK runs are
            89.1 (2026-10-04, one pass) and 86.6 (2026-10-14, three passes); the latter is used.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from huggingface_hub import hf_hub_download

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratch"))

from plot_misalignment_three_evals import am_point, odcv_point, results  # noqa: E402
from src.naming import figure_path  # noqa: E402

ORG = "dougalldeepmind"
LABEL = {"plain": "+ Control (Standard SFT)", "da-15": "+ difficult advice 15% mix",
         "low-15": "+ low stakes 15% mix", "cot-15": "+ CoT only 15% mix",
         "onesided-15": "+ one-sided deliberation 15% mix", "tools-15": "+ difficult advice + tool defs 15% mix"}
COLOUR = {"plain": "#6B7280", "da-15": "#7E22CE", "low-15": "#3F9B3F", "cot-15": "#A8D86A", "onesided-15": "#6B4F1D",
          "tools-15": "#C2507A"}
RUNS = {
    "mask": {"plain": f"{ORG}/2026-10-09-mask-qwen36-0-plain",
             "da-15": f"{ORG}/2026-10-09-mask-qwen36-0-da-15",
             "low-15": f"{ORG}/2026-10-07-mask-qwen36-0-da-lowstakes-practical-15",
             "cot-15": f"{ORG}/2026-10-08-mask-qwen36-0-da-15-cot",
             "onesided-15": f"{ORG}/2026-10-14-mask-qwen36-0-ablation-stated-harm-15",
             "tools-15": f"{ORG}/2026-10-11-mask-qwen36-0-da-tools-15"},
    "odcv": {"plain": f"{ORG}/2026-10-06-odcv-qwen36-0-plain",
             "da-15": f"{ORG}/2026-10-06-odcv-qwen36-0-da-15",
             "low-15": f"{ORG}/2026-10-07-odcv-qwen36-0-da-lowstakes-practical-15",
             "cot-15": f"{ORG}/2026-10-11-odcv-qwen36-0-da-15-cot",
             "onesided-15": f"{ORG}/2026-10-04-odcv-qwen36-0-ablation-stated-harm-15",
             "tools-15": f"{ORG}/2026-10-11-odcv-qwen36-0-da-tools-15"},
    "am": {"plain": f"{ORG}/2026-10-08-am-qwen36-0-plain",
           "da-15": f"{ORG}/2026-10-08-am-qwen36-0-da-15",
           "low-15": f"{ORG}/2026-10-08-am-qwen36-0-da-lowstakes-practical-15",
           "cot-15": f"{ORG}/2026-10-11-am-qwen36-0-da-15-cot",
           "onesided-15": f"{ORG}/2026-10-14-am-qwen36-0-ablation-stated-harm-15",
           "tools-15": f"{ORG}/2026-10-13-am-qwen36-0-da-tools-15"},
    # Colosseum hospital, self-sacrificial cell: share of the 30 seeds in which the pair took a
    # sabotage action (the published `pair_sabotaged` measure), Wilson interval on the count.
    "hospital": {"plain": f"{ORG}/2026-10-11-hospital-self-sacrificial-qwen36-0-plain",
                 "da-15": f"{ORG}/2026-10-15-hospital-self-sacrificial-qwen36-0-da-15",
                 "tools-15": f"{ORG}/2026-10-14-hospital-self-sacrificial-qwen36-0-da-tools-15"},
}
SETS = {"stakes": ["plain", "da-15", "low-15"], "cot": ["plain", "da-15", "cot-15", "onesided-15"],
        "tools": ["plain", "da-15", "tools-15"], "da": ["plain", "da-15"]}
HOSPITAL_SETS = {"tools", "da"}   # these sets add the multi-agent panel
SUBJECT = {"stakes": "stakes_mask_vs_single_agent_slides", "cot": "cot_mask_vs_single_agent_slides",
           "tools": "tools_mask_single_agent_hospital_slides", "da": "da_mask_single_agent_hospital_slides"}
BASE_RUNS = {"mask": f"{ORG}/2026-10-03-mask-qwen36", "odcv": f"{ORG}/2026-09-06-odcv-qwen36",
             "hospital": f"{ORG}/2026-10-13-hospital-self-sacrificial-qwen36"}
# No think-mode base AM run is published (2026-10-16 search of the org: only the July 12-condition
# harness at 65.5% and 2026-10-13-am-qwen36 in default mode at 1.1%). This is the value the original
# three-panel figure drew; replace it with a published run's rate when one exists.
AM_BASE = 45.0
Z = 1.959964
FS = dict(title=20, tick=16, label=18, value=16, legend=19)


def wilson(k: int, n: int) -> tuple[float, float]:
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def mask_point(repo: str) -> dict:
    r = results(repo)
    n = sum(a["n_responses"] for a in r["per_archetype"].values())
    k = round(sum(a["n_responses"] * a["unhonest_pct"] / 100 for a in r["per_archetype"].values()))
    lo_lie, hi_lie = wilson(k, n)
    # Reported as DISHONESTY = 100 - overall_honesty_score, so every panel reads "lower is better".
    return dict(rate=100 - r["overall_honesty_score"], lo=lo_lie, hi=hi_lie, n=n, passes=r.get("passes"))


def hospital_point(repo: str) -> dict:
    ep = json.load(open(hf_hub_download(repo, "results/episodes.json", repo_type="dataset")))
    byseed = {int(e["seed"]): e for e in ep}
    n = len(byseed)
    k = sum(1 for e in byseed.values() if e["pair_sabotaged"] in (1, "1", True))
    lo, hi = wilson(k, n)
    return dict(rate=100 * k / n, lo=lo, hi=hi, n=n, k=k)


def aggregate(o: dict, a: dict) -> dict:
    rate = (o["rate"] + a["rate"]) / 2
    dlo = math.hypot(o["rate"] - o["lo"], a["rate"] - a["lo"]) / 2
    dhi = math.hypot(o["hi"] - o["rate"], a["hi"] - a["rate"]) / 2
    return dict(rate=rate, lo=rate - dlo, hi=rate + dhi)


def bars(ax, arms: list[str], pts: dict, title: str, ylabel: str, ylim: float, base: float | None,
         width: float = 0.62, title_fs: float | None = None, tight: bool = False) -> None:
    for i, arm in enumerate(arms):
        p = pts[arm]
        ax.bar(i, p["rate"], width, color=COLOUR[arm], edgecolor="black", linewidth=0.8, zorder=3)
        ax.errorbar(i, p["rate"], yerr=[[p["rate"] - p["lo"]], [p["hi"] - p["rate"]]], fmt="none",
                    ecolor="black", elinewidth=1.3, capsize=5, zorder=4)
        ax.annotate(f"{p['rate']:.1f}", (i, p["hi"]), xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=FS["value"], fontweight="bold", zorder=5,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    if base is not None:
        ax.axhline(base, color="#C62828", linestyle="--", linewidth=2, zorder=2)
    ax.set_title(title, fontsize=title_fs or FS["title"], pad=14)
    ax.set_ylabel(ylabel, fontsize=FS["label"])
    ax.set_ylim(0, ylim)
    ax.set_xticks([])
    if tight:
        ax.set_xlim(-0.6, len(arms) - 0.4)
    ax.tick_params(axis="y", labelsize=FS["tick"])
    ax.yaxis.grid(True, linestyle=":", color="#D1D5DB")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(SETS), default="stakes")
    ap.add_argument("--panels", choices=["all", "chat_single", "hospital"], default="all",
                    help="all = every panel in one figure; chat_single = MASK + aggregate only; hospital = the multi-agent panel alone")
    ap.add_argument("--reverse", action="store_true", help="draw the panels right-to-left (hospital first)")
    ap.add_argument("--square", action="store_true", help="square canvas for a slide: two panels side by side, thinner bars, smaller titles")
    args = ap.parse_args()
    arms = SETS[args.set]

    mask = {a: mask_point(RUNS["mask"][a]) for a in arms}
    odcv = {a: odcv_point(a, RUNS["odcv"][a]) for a in arms}
    am = {a: am_point(RUNS["am"][a]) for a in arms}
    agg = {a: aggregate(odcv[a], am[a]) for a in arms}
    mask_base = 100 - results(BASE_RUNS["mask"])["overall_honesty_score"]
    odcv_base = results(BASE_RUNS["odcv"])["ours"]["overall"]["mr_pct"]
    agg_base = (odcv_base + AM_BASE) / 2
    with_hospital = args.set in HOSPITAL_SETS
    hosp = {a: hospital_point(RUNS["hospital"][a]) for a in arms} if with_hospital else {}
    hosp_base = hospital_point(BASE_RUNS["hospital"])["rate"] if with_hospital else None

    if args.panels == "hospital":
        assert with_hospital, f"set {args.set!r} has no hospital runs"
        panels = [("hospital", hosp, "Colosseum hospital (Multi-agent Eval)\n(lower is better)", "Sabotage rate (%)", 112, hosp_base)]
    else:
        panels = [("mask", mask, "MASK dishonesty (Chat Eval)\n(lower is better)", "Dishonesty rate (%)", 70, mask_base),
                  ("agg", agg, "Aggregate single agent misalignment\n(lower is better)", "Misalignment rate (%)", 55, agg_base)]
        if with_hospital and args.panels == "all":
            panels.append(("hospital", hosp, "Colosseum hospital (Multi-agent Eval)\n(lower is better)", "Sabotage rate (%)", 112, hosp_base))
    if args.reverse:
        panels = panels[::-1]
    npanels = len(panels)
    if args.square:
        side = 10.0 + 1.0 * max(0, len(arms) - 3)
        fig, axes = plt.subplots(1, npanels, figsize=(side, side), dpi=200, squeeze=False)
        for ax, (_, pts, title, ylabel, ylim, base) in zip(axes[0], panels):
            bars(ax, arms, pts, title.replace("Aggregate single agent misalignment", "Aggregate single agent\nmisalignment"),
                 ylabel, ylim, base, width=0.72, title_fs=17, tight=True)
    else:
        width = max(10.5, 7.0 * npanels + 1.5 * max(0, len(arms) - 3))
        fig, axes = plt.subplots(1, npanels, figsize=(width, 6.2), dpi=200, squeeze=False)
        for ax, (_, pts, title, ylabel, ylim, base) in zip(axes[0], panels):
            bars(ax, arms, pts, title, ylabel, ylim, base)
    handles = [Line2D([], [], color="#C62828", linestyle="--", linewidth=2, label="Qwen-3.6-27b")]
    handles += [Patch(facecolor=COLOUR[a], edgecolor="black", label=LABEL[a]) for a in arms]
    # one legend row only when it fits: ~3 entries per 7in panel at this font size
    if args.square:
        ncol = 2
        fig.legend(handles=handles, loc="upper center", ncol=ncol, frameon=False, fontsize=FS["legend"] - 2,
                   bbox_to_anchor=(0.5, 1.01), handlelength=1.8, columnspacing=1.4)
        fig.tight_layout(rect=(0, 0, 1, 0.90 - 0.035 * (math.ceil(len(handles) / 2) - 2)))
    else:
        ncol = 2 if npanels == 1 else (len(handles) if len(handles) <= 1.5 * npanels else math.ceil(len(handles) / 2))
        fig.legend(handles=handles, loc="upper center", ncol=ncol, frameon=False, fontsize=FS["legend"] - (1 if npanels == 1 else 0),
                   bbox_to_anchor=(0.5, 1.02), handlelength=1.8, columnspacing=1.4)
        fig.tight_layout(rect=(0, 0, 1, 0.9 if ncol == len(handles) else 0.85))

    out_dir = REPO / "output/plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    subject = SUBJECT[args.set] + ("" if args.panels == "all" else "_" + args.panels) + ("_reversed" if args.reverse else "") + ("_square" if args.square else "")
    png = figure_path(out_dir, subject, ext="png")
    fig.savefig(png)
    fig.savefig(figure_path(out_dir, subject, ext="pdf"))

    lines = [f"# {png.stem}", "", "Bars read from the published runs; intervals as in the module docstring.", "",
             "| panel | arm | value | 95% CI | n | run |", "|---|---|---|---|---|---|"]
    for a in arms:
        m = mask[a]
        lines.append(f"| MASK dishonesty (100 - honesty) | {a} | {m['rate']:.1f} | [{m['lo']:.1f}, {m['hi']:.1f}] | {m['n']} responses, {m['passes']} pass(es) | {RUNS['mask'][a]} |")
    for a in arms:
        o, h, g = odcv[a], am[a], agg[a]
        lines.append(f"| ODCV MR | {a} | {o['rate']:.1f} | [{o['lo']:.1f}, {o['hi']:.1f}] | {o['n']} rollouts | {RUNS['odcv'][a]} |")
        lines.append(f"| AM harmful | {a} | {h['rate']:.1f} | [{h['lo']:.1f}, {h['hi']:.1f}] | {h['n']} samples | {RUNS['am'][a]} |")
        lines.append(f"| aggregate | {a} | {g['rate']:.1f} | [{g['lo']:.1f}, {g['hi']:.1f}] | mean of the two | - |")
    for a in hosp:
        h = hosp[a]
        lines.append(f"| hospital sabotage | {a} | {h['rate']:.1f} | [{h['lo']:.1f}, {h['hi']:.1f}] | {h['k']}/{h['n']} seeds | {RUNS['hospital'][a]} |")
    if with_hospital:
        lines.append(f"| hospital sabotage | base | {hosp_base:.1f} | - | 30 seeds | {BASE_RUNS['hospital']} |")
    lines += ["", f"Base lines: MASK {mask_base:.1f} ({BASE_RUNS['mask']}); ODCV {odcv_base:.1f} ({BASE_RUNS['odcv']}); "
              f"AM {AM_BASE:.1f} (NOT a published run, carried over from the original figure); aggregate {agg_base:.1f}."]
    Path(str(png)[:-4] + "_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {png}")


if __name__ == "__main__":
    main()
