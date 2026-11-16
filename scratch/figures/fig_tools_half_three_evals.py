# ABOUTME: Three separate bar charts (ODCV, MASK, Hospital) comparing control, da-15, da-tools-15 and the
# ABOUTME: 2026-11-10 50-50 DA / DA+unused-tools-by-trait arm; every number read from the Hub at render time.
# Run: uv run python scratch/figures/fig_tools_half_three_evals.py [--out_dir output/figures]
"""Bar = mean over the arm's training draws (A, B, C where they exist); each draw is a dot with its own 95%
interval: ODCV and MASK by the figure rule (`_hub.item_se`, per-item SE from repeats), Hospital a Wilson
interval over 30 episodes (episodes are independent draws). The Hospital control is the only comparable-ish
plain-in-the-pair run there is and ran on the September harness (peer nosynth, terrarium fixes 2026-09-15,
public actions without rationale), so it is hatched and footnoted rather than dropped.
"""
from __future__ import annotations

import fire
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.naming import figure_path
from src.utils import save_figure_pdf
from scratch.figures import _hub

matplotlib.use("Agg")

ARMS = ["control", "da", "dat", "half"]
LABEL = {"control": "Control\n(standard SFT)", "da": "Difficult\nadvice", "dat": "DA + unused\ntools",
         "half": "50-50 by trait\n(new)"}
COLOUR = {"control": "#5a6472", "da": "#7e22ce", "dat": "#e8762b", "half": "#0d9488"}

ODCV = {"control": {"A": "2026-10-06-odcv-qwen36-0-plain"},
        "da": {"A": "2026-10-06-odcv-qwen36-0-da-15", "B": "2026-11-02-odcv-qwen36-0-da-15-b",
               "C": "2026-11-02-odcv-qwen36-0-da-15-c"},
        "dat": {"A": "2026-10-11-odcv-qwen36-0-da-tools-15", "B": "2026-11-06-odcv-qwen36-0-da-tools-15-b",
                "C": "2026-11-06-odcv-qwen36-0-da-tools-15-c"},
        "half": {"A": "2026-11-10-odcv-qwen36-0-da-tools-15-trait-half-tools"}}
MASK = {"control": {"A": "2026-11-03-mask-qwen36-0-plain"},
        "da": {"A": "2026-11-03-mask-qwen36-0-da-15", "B": "2026-11-09-mask-qwen36-0-da-15-b",
               "C": "2026-11-09-mask-qwen36-0-da-15-c"},
        "dat": {"A": "2026-11-09-mask-qwen36-0-da-tools-15", "B": "2026-11-06-mask-qwen36-0-da-tools-15-b",
                "C": "2026-11-06-mask-qwen36-0-da-tools-15-c"},
        "half": {"A": "2026-11-10-mask-qwen36-0-da-tools-15-trait-half-tools"}}
HOSP = {"control": {"A": "2026-10-11-hospital-self-sacrificial-qwen36-0-plain"},
        "da": {"B": "2026-11-05-hospital-self-sacrificial-qwen36-0-da-15-b",
               "C": "2026-11-05-hospital-self-sacrificial-qwen36-0-da-15-c"},
        "dat": {"B": "2026-11-06-hospital-self-sacrificial-qwen36-0-da-tools-15-b",
                "C": "2026-11-06-hospital-self-sacrificial-qwen36-0-da-tools-15-c"},
        "half": {"A": "2026-11-11-hospital-self-sacrificial-qwen36-0-da-tools-15-trait-half-tools"}}
OLD_HARNESS = {("hospital", "control")}

plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
                     "axes.titlesize": 13, "axes.linewidth": 1.0})


def hospital(repo: str) -> tuple[float, float, float]:
    """Sabotage-action rate over the 30 shifts with a Wilson 95% interval."""
    v = list(_hub.fetch(repo, "results/results.json")["measures"]["pair_sabotaged"]["self_sacrificial"].values())
    k, n = int(sum(v)), len(v)
    lo, hi = _hub.wilson(k, n)
    return 100 * k / n, lo, hi


def chart(key: str, runs: dict, reader, title: str, ylabel: str, ylim: float, out_dir: str,
          footnote: str) -> tuple[str, list[str]]:
    """One bar chart; returns the markdown table rows and the artifacts read."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    rows, artifacts = [], []
    for x, arm in enumerate(ARMS):
        draws = {d: reader(r) for d, r in runs[arm].items()}
        artifacts += [f"{_hub.ORG}/{r}" for r in runs[arm].values()]
        mean = float(np.mean([v[0] for v in draws.values()]))
        old = (key, arm) in OLD_HARNESS
        ax.bar(x, mean, width=0.62, color=COLOUR[arm], alpha=0.35 if old else 0.85,
               hatch="//" if old else None, edgecolor=COLOUR[arm], linewidth=0)
        offs = np.linspace(-0.16, 0.16, len(draws)) if len(draws) > 1 else [0.0]
        for o, (d, (m, lo, hi)) in zip(offs, draws.items()):
            if lo is not None:
                ax.plot([x + o, x + o], [lo, hi], color="#1f2328", lw=1.2, zorder=3)
            ax.scatter([x + o], [m], s=34, color="white", edgecolor="#1f2328", lw=1.2, zorder=4)
            if len(draws) > 1:
                ax.annotate(d, (x + o, m), xytext=(4, -3), textcoords="offset points", fontsize=8,
                            color="#57606a")
            rows.append(f"| {LABEL[arm].replace(chr(10), ' ')} | {d} | {runs[arm][d]} | {m:.1f} | "
                        + (f"[{lo:.1f}, {hi:.1f}]" if lo is not None else "-") + " |")
        top = max([v[2] if v[2] is not None else v[0] for v in draws.values()] + [mean])
        ax.text(x, top + ylim * 0.025, f"{mean:.1f}%", ha="center", va="bottom", fontsize=11,
                color="#1f2328", fontweight="bold")
    ax.set_xticks(range(len(ARMS)), [LABEL[a] for a in ARMS])
    ax.set_ylim(0, ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#e5e7eb", lw=0.8)
    ax.set_axisbelow(True)
    fig.text(0.01, -0.04, footnote, fontsize=8.5, color="#57606a", ha="left", va="top", wrap=True)
    png = figure_path(out_dir, f"tools-half-{key}", ext="png")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    save_figure_pdf(fig, png.with_suffix(".pdf"), title, artifacts, "scratch/figures/fig_tools_half_three_evals.py")
    plt.close(fig)
    print(png)
    return "\n".join(rows), artifacts


def main(out_dir: str = "output/figures") -> None:
    """Render the three charts and a results.md beside them."""
    dots = "Bar = mean over training draws (letters = corpus draw A/B/C); dot + line = that run's own 95% interval."
    specs = [
        ("odcv", ODCV, _hub.odcv, "ODCV: misalignment rate (lower is better)", "Misaligned rollouts (%)", 55,
         dots + " 80 cells x 3 rollouts per run; interval from per-cell SE."),
        ("mask", MASK, _hub.mask, "MASK: honesty (higher is better)", "Honest answers (%)", 100,
         dots + " 1,000 rows x 3 passes per run; interval from per-row SE."),
        ("hospital", HOSP, hospital, "Hospital: pair took a sabotage action (lower is better)",
         "Shifts with sabotage (%)", 110,
         dots + " 30 shifts per run, Wilson interval. Hatched control: September harness (peer nosynth, older"
         " fixes); every other bar shares peer 2026-10-05-plain and fixes 2026-10-09."),
    ]
    md = ["# 50-50 DA / DA+unused-tools arm vs control, DA and DA+tools", "",
          "Script: `scratch/figures/fig_tools_half_three_evals.py`. All values read from the Hub at render time.", ""]
    for key, runs, reader, title, ylabel, ylim, foot in specs:
        rows, _ = chart(key, runs, reader, title, ylabel, ylim, out_dir, foot)
        md += [f"## {title}", "", foot, "", "| arm | draw | run | value % | 95% |", "|---|---|---|---|---|", rows, ""]
    path = figure_path(out_dir, "tools-half-three-evals-results", ext="md")
    path.write_text("\n".join(md), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    fire.Fire(main)
