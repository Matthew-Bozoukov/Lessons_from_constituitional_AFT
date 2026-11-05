# ABOUTME: Qwen3.6-27B base / Control / DA15 on MMLU and SWE-bench Lite in the paper's Fig. 2 style
# ABOUTME: (base as a red dashed line), every number read from a pinned Hub run at render time.
# Run: uv run python scratch/plot_qwen_capabilities.py
"""Remake of the old appendix capability figure (scratch/capability_preservation/plot.py), styled
like scratch/plot_probes_four_remake.py. Each arm was run once on each benchmark, so under the
error-bar rule in scratch/figures/_hub.py there are no repeats and no bars."""
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
ARMS = ["control", "da"]
ARM = {"control": "+ Control (Standard SFT)", "da": "+ difficult advice 15% mix"}
COLOUR = {"control": "#5a6472", "da": "#7e22ce"}   # as in plot_probes_four_remake.py
BASE_COLOUR = "#c1121f"

# Pins: the revisions the old figure read (scratch/capability_preservation/figures.yaml).
MMLU = {"base": ("2026-10-14-mmlu-qwen36", "6f41ca3384b14c101d81f996065529021dc78eac"),
        "control": ("2026-10-14-mmlu-qwen36-0-plain", "a4556e5a8b2a24de05892133165fbef75035a88e"),
        "da": ("2026-10-14-mmlu-qwen36-0-da-15", "da022054c494987bc1c75515851c49b55357eb12")}
SWE = {"base": ("2026-10-07-swebench-qwen36-lite-v5", "9e7447f072aaacfbb4919a8c6f62ac881c04004a"),
       "control": ("2026-10-07-swebench-qwen36-0-plain-lite-v5", "1148e0db51d4bbb7ac25d22be593a6bb77e63581"),
       "da": ("2026-10-07-swebench-qwen36-0-da-15-lite-v5", "bde7963c515c5f2e7be59fe50d6883cf48650e31")}

PANELS = ["mmlu", "swe"]
TITLE = {"mmlu": "MMLU", "swe": "SWE-bench Lite"}
YLABEL = {"mmlu": "Accuracy (%), higher is better", "swe": "Tasks resolved (%), higher is better"}

plt.rcParams.update({"font.size": 19, "axes.labelsize": 17, "xtick.labelsize": 17,
                     "ytick.labelsize": 18, "axes.titlesize": 20, "legend.fontsize": 18,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})


def _dl(repo: str, rev: str) -> dict:
    return json.load(open(hf_hub_download(f"{ORG}/{repo}", "results/results.json", repo_type="dataset",
                                          revision=rev, token=os.environ["HF_TOKEN"]), encoding="utf-8"))


def read_all() -> tuple[dict, dict, list[tuple]]:
    """Every bar value, every base line, and a provenance row per number."""
    vals: dict[str, dict] = {p: {} for p in PANELS}
    base: dict[str, float] = {}
    rows: list[tuple] = []
    for a in ["base", *ARMS]:
        m = _dl(*MMLU[a])
        s = _dl(*SWE[a])
        assert s["status"] == "complete" and s["n_graded"] == s["n_total"], f"{SWE[a][0]} is not fully graded"
        point = {"mmlu": (100 * m["n_correct"] / m["n"], f"{MMLU[a][0]} ({m['n_correct']}/{m['n']}, one pass)"),
                 "swe": (100 * s["n_resolved"] / s["n_total"], f"{SWE[a][0]} ({s['n_resolved']}/{s['n_total']}, one attempt)")}
        for p, (v, src) in point.items():
            if a == "base":
                base[p] = v
            else:
                vals[p][a] = v
            rows.append((p, a, v, src))
    return vals, base, rows


def main() -> None:
    vals, base, rows = read_all()
    fig, axes = plt.subplots(1, len(PANELS), figsize=(10.5, 7.8), gridspec_kw={"wspace": 0.45})
    for ax, p in zip(axes, PANELS):
        for i, a in enumerate(ARMS):
            m = vals[p][a]
            ax.bar(i, m, width=0.62, color=COLOUR[a], edgecolor="black", linewidth=0.8, zorder=3)
            ax.text(i, m + 1.4, f"{m:.1f}", ha="center", va="bottom",
                    fontsize=16, fontweight="bold", zorder=5, clip_on=False)
        ax.axhline(base[p], color=BASE_COLOUR, linestyle=(0, (2.5, 2.5)), linewidth=1.8, zorder=2)
        ax.set_title(TITLE[p], pad=12)
        ax.set_ylabel(YLABEL[p])
        ax.set_ylim(0, 100); ax.set_yticks(range(0, 101, 20))
        ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([])
        ax.set_xlim(-0.6, len(ARMS) - 0.4)
        ax.grid(True, axis="y", linestyle="--", alpha=0.2); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles = [Line2D([0], [0], color=BASE_COLOUR, linestyle=(0, (2.5, 2.5)), linewidth=1.8, label=BASE_NAME)]
    handles += [Patch(facecolor=COLOUR[a], edgecolor="black", label=ARM[a]) for a in ARMS]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.0), fontsize=16, columnspacing=1.2, handlelength=1.6,
               handleheight=1.0)
    fig.subplots_adjust(top=0.82, bottom=0.06)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"{datetime.now():%Y-%m-%d}_qwen_capabilities"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    md = [f"# {stem.name}", "", f"Rendered {datetime.now().isoformat(timespec='seconds')} by "
          "scratch/plot_qwen_capabilities.py; every number read from a pinned Hub revision at render time.", "",
          "| Panel | Arm | Value | Source |", "|---|---|---|---|"]
    md += [f"| {p} | {a} | {v:.1f} | {src} |" for p, a, v, src in rows]
    md += ["", "No error bars: each arm was run once on each benchmark (rule: scratch/figures/_hub.py)."]
    Path(f"{stem}_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"{stem}.png")


if __name__ == "__main__":
    main()
