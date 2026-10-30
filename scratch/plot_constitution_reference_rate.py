# ABOUTME: How often each arm's reasoning explicitly references the constitution -- in its training corpus and
# ABOUTME: at eval time on ODCV and MASK -- read off the published LLM-labelled trace analysis on the Hub.
"""    uv run python scratch/plot_constitution_reference_rate.py

Reads `dougalldeepmind/2026-09-17-train-vs-eval-trait-ref` (Gemini 3 Flash labels over every training row of
the four constitutional-SFT arms and ~250 eval traces per arm per eval, sampled uniformly over ALL MASK
traces -- pressured response and belief-elicitation runs alike -- and over all ODCV assistant turns) and plots
two rates per arm and setting: NAMES = the trace references the constitution as a document (scheme
`reference`, label `substantive`); CITES = the trace invokes a specific principle of it (scheme `traits`, label
`trait_specific`). Bars carry Wilson 95% intervals from the label counts. Writes the figure and a results.md
beside it under output/figures/. nosynth is not in the artifact (it never saw the constitution).
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from math import sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trait_ref_arms import ARM_SETS  # noqa: E402
from src.naming import figure_path

load_dotenv()
ap = argparse.ArgumentParser()
ap.add_argument("--arm-set", default="7", choices=list(ARM_SETS))
ap.add_argument("--artifact", default="", help="the published trait-ref artifact (overrides the arm set's)")
ARGS = ap.parse_args()
ARM_SET = ARM_SETS[ARGS.arm_set]
DATE = "2026-09-18" if ARGS.arm_set == "7" else ARM_SET["date"]
OUT = Path("output/figures")
REPO = ARGS.artifact or ARM_SET["artifact"]
assert REPO, f"arm set {ARGS.arm_set!r} has no published artifact yet; pass --artifact"
csv.field_size_limit(1 << 30)
# label -> colour: family colours per CLAUDE.md, variants as tints of their family
ARMS = {arm: spec["colour"] for arm, spec in ARM_SET["arms"].items()}
SETTINGS = (("training", "Training corpus"), ("odcv", "ODCV"), ("mask", "MASK"))
# BOTH metrics are CITATION detectors: they ask whether the constitution is verbally recoverable
# from the trace, not whether the reasoning engages it. A difficult-advice row weighs the principle
# it was dealt in its own words with nothing quotable, and scores "none" here by design -- so these
# bars are a lower bound on engagement, and the labels say "states"/"names", never "invokes".
METRICS = (("cites", "traits", "trait_specific", "states a specific principle (number, title or paraphrase)"),
           ("names", "reference", "substantive", "names the constitution as a document"))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """(rate, lo, hi) in percent."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


def fetch() -> tuple[dict, dict]:
    labels = Path(hf_hub_download(REPO, "labels.csv", repo_type="dataset"))
    summary = json.loads(Path(hf_hub_download(REPO, "summary.json", repo_type="dataset")).read_text())
    n, k = Counter(), Counter()
    with labels.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for key, scheme, label, _ in METRICS:
                if r["scheme"] == scheme:
                    n[(key, r["eval"], r["arm"])] += 1
                    k[(key, r["eval"], r["arm"])] += r["label"] == label
    rows = {(key, ev, arm): wilson(k[(key, ev, arm)], n[(key, ev, arm)]) + (k[(key, ev, arm)], n[(key, ev, arm)])
            for key, *_ in METRICS for ev, _ in SETTINGS for arm in ARMS}
    return rows, summary


def main() -> None:
    rows, summary = fetch()
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(SETTINGS), figsize=(12.5, 4.4), dpi=160, sharey=True)
    x = np.arange(len(ARMS))
    w = 0.38
    for ax, (ev, title) in zip(axes, SETTINGS):
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", color="#e6e6e3", linewidth=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for i, (arm, colour) in enumerate(ARMS.items()):
            for j, (key, *_) in enumerate(METRICS):
                p, lo, hi, _, _ = rows[(key, ev, arm)]
                xx = x[i] + (j - 0.5) * w
                style = dict(color=colour, edgecolor=colour, linewidth=1.2) if key == "cites" else \
                    dict(facecolor="white", edgecolor=colour, linewidth=1.6, hatch="////")
                ax.bar(xx, p, w, zorder=3, **style)
                ax.errorbar(xx, p, yerr=[[max(p - lo, 0)], [max(hi - p, 0)]], fmt="none", ecolor="#2b2b2b", elinewidth=1, capsize=2.5, zorder=4)
                ax.annotate(f"{p:.0f}", (xx, max(hi, p)), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                            fontsize=8, color="#2b2b2b")
        ax.set_xticks(x, list(ARMS))
        ax.set_title(title, fontsize=10, loc="left")
        ax.tick_params(colors="#555", labelsize=9)
    axes[0].set_ylabel("reasoning traces, %")
    axes[0].set_ylim(0, 112)
    axes[0].set_yticks(range(0, 101, 20))
    # A figure-level legend, not one parked inside the MASK axes: the labels are longer than a panel
    # is wide, so an in-axes legend overflowed the frame.
    fig.legend(handles=[Patch(color="#6b6b6b"), Patch(facecolor="white", edgecolor="#6b6b6b", hatch="////")],
               labels=[m[3] for m in METRICS], frameon=False, fontsize=8.5, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=1.4, columnspacing=1.8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    arm_suffix = "" if ARGS.arm_set == "7" else "_" + ARGS.arm_set   # the lint reads the literal; keep it one f-string
    png = figure_path(OUT, f"constitution_reference_rate{arm_suffix}")
    fig.savefig(png)

    md = [f"# Explicit constitution reference in reasoning, training vs eval ({DATE})", "",
          f"Source: `{REPO}` (labels by google/gemini-3-flash-preview, temperature 0). "
          "NAMES = scheme `reference`, label `substantive`; CITES = scheme `traits`, label `trait_specific`. "
          "BOTH are CITATION detectors -- they ask whether the constitution is verbally recoverable from the trace, "
          "not whether the reasoning engages it, so they are a LOWER BOUND on engagement. On the difficult-advice "
          "corpora, where the dealt principle is implicit by design, a detection is nearly always the dealt trait "
          "(da-15 85/92, da-qwen-15 430/436) but the rate is low and uneven by trait (t3/t4/t5 ~1% for da-15), which "
          "measures how quotable a principle is, not how much it drove the reasoning. "
          "Eval traces are a uniform sample over all MASK traces (pressured + belief-elicitation) and all ODCV assistant turns; "
          "training corpora are labelled in full. Rates in % with Wilson 95% intervals; one seed per arm.", "",
          "| arm | setting | states a specific principle | names the document |", "|---|---|---|---|"]
    for arm in ARMS:
        for ev, _ in SETTINGS:
            cells = []
            for key, *_ in METRICS:
                p, lo, hi, kk, nn = rows[(key, ev, arm)]
                cells.append(f"{p:.1f} [{lo:.1f}, {hi:.1f}] ({kk}/{nn})")
            md.append(f"| {arm} | {ev} | {cells[0]} | {cells[1]} |")
    md += ["", "Adapters: " + ", ".join(f"`{v}`" for v in summary["adapters"].values()),
           "Corpora: " + ", ".join(f"`{v}`" for v in summary["corpora"].values())]
    (OUT / png.name.replace(".png", "_results.md")).write_text("\n".join(md) + "\n")
    print(png)
    print("\n".join(md[4:]))


if __name__ == "__main__":
    main()
