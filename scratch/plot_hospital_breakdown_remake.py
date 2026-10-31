# ABOUTME: Paper figure: the Colosseum hospital misalignment breakdown (sabotage act vs sabotage plan without an
# ABOUTME: act) for base, control, delib, da and da_tools, read from each run's episodes.json on the Hub at render.
# Run: uv run python scratch/plot_hospital_breakdown_remake.py
"""Stacked bar per arm: shifts where the watched pair took a sabotage act (dark) and shifts where it only posted a
plan (light), out of the seeds run. Large type for a half-page paper figure. Writes
output/plots/<date>_hospital_breakdown_five_arm.{png,pdf,svg} and a _results.md mirror listing the runs."""
import datetime as dt
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv("/Users/jamie/Projects/lasr/.env")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from matplotlib.patches import Patch

from scratch.figures import _hub

OUT = Path("output/plots")
ORG = _hub.ORG
ARMS = ["base", "control", "delib", "da", "da_tools"]
ARM = {"base": "Base model", "control": "Control\n(standard SFT)", "delib": "Deliberative\nalignment",
       "da": "Difficult\nadvice", "da_tools": "Difficult advice\n+ tools"}
RUNS = {"base": "2026-10-13-hospital-self-sacrificial-qwen36",
        "control": "2026-10-11-hospital-self-sacrificial-qwen36-0-plain",
        "delib": "2026-10-11-hospital-self-sacrificial-qwen36-0-delib-sonnet-15",
        "da": "2026-10-15-hospital-self-sacrificial-qwen36-0-da-15",
        "da_tools": "2026-10-14-hospital-self-sacrificial-qwen36-0-da-tools-15"}
BOXES = [("act", "Sabotage act", "#b3261e", "white"),
         ("plan_only", "Sabotage plan, no sabotage act", "#f0c3bd", "#3b0d0a")]
MUTED = "#555d66"

plt.rcParams.update({"font.size": 19, "axes.labelsize": 19, "xtick.labelsize": 17,
                     "ytick.labelsize": 18, "axes.titlesize": 20, "legend.fontsize": 18,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})


def _dl(repo: str, path: str):
    return json.load(open(hf_hub_download(f"{ORG}/{repo}", path, repo_type="dataset",
                                          token=os.environ["HF_TOKEN"])))


def _flag(v) -> bool:
    return v in (1, "1", True)


def counts(repo: str) -> dict:
    ep = {int(e["seed"]): e for e in _dl(repo, "results/episodes.json")}
    act = sum(1 for e in ep.values() if _flag(e["pair_sabotaged"]))
    plan_only = sum(1 for e in ep.values() if _flag(e["plan_posted"]) and not _flag(e["pair_sabotaged"]))
    return {"n": len(ep), "act": act, "plan_only": plan_only}


def main() -> None:
    c = {a: counts(RUNS[a]) for a in ARMS}
    n_max = max(v["n"] for v in c.values())
    fig, ax = plt.subplots(figsize=(10.0, 7.8))
    xs = list(range(len(ARMS)))
    for x, a in zip(xs, ARMS):
        bottom = 0
        for key, _, colour, ink in BOXES:
            k = c[a][key]
            if not k:
                continue
            ax.bar(x, k, 0.66, bottom=bottom, color=colour, edgecolor="black", linewidth=0.8, zorder=3)
            if k >= 2:
                ax.text(x, bottom + k / 2, str(k), ha="center", va="center", fontsize=17, color=ink, zorder=4,
                        fontweight="bold")
            bottom += k
        ax.text(x, bottom + 0.4, f"{bottom}/{c[a]['n']}", ha="center", va="bottom", fontsize=15, color=MUTED)
    ax.set_xticks(xs, [ARM[a] for a in ARMS])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, xs[-1] + 0.6)
    ax.set_ylim(0, n_max * 1.1)
    ax.set_yticks(range(0, n_max + 1, 5))
    ax.set_ylabel(f"Shifts with a sabotage act or plan (of {n_max})")
    ax.yaxis.grid(True, color="#e3e6ea", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[Patch(facecolor=col, edgecolor="black", label=lab) for _, lab, col, _ in BOXES],
              frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, handlelength=1.2,
              handleheight=0.9, borderaxespad=0.2)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.86, bottom=0.14)
    stem = OUT / f"{dt.date.today():%Y-%m-%d}_hospital_breakdown_five_arm"
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    md = [f"# {stem.name}", "", f"Rendered {dt.datetime.now():%Y-%m-%dT%H:%M} by scratch/plot_hospital_breakdown_remake.py; "
          "every count read from results/episodes.json on the Hub at render. act = pair_sabotaged; "
          "plan only = plan_posted and not pair_sabotaged.", "",
          "| Arm | Sabotage act | Plan only | Act or plan | Seeds | Run |", "|---|---|---|---|---|---|"]
    for a in ARMS:
        v = c[a]
        md.append(f"| {ARM[a].replace(chr(10), ' ')} | {v['act']} | {v['plan_only']} | {v['act'] + v['plan_only']} | "
                  f"{v['n']} | {RUNS[a]} |")
    Path(f"{stem}_results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(stem)


if __name__ == "__main__":
    main()
