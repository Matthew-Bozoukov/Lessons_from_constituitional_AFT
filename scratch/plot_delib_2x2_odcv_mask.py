# ABOUTME: Paper figure: the teacher x method 2x2 (Sonnet/Qwen x difficult advice/deliberative
# ABOUTME: alignment) on ODCV misalignment rate and MASK honesty, two panels with their own axes.
# Run: uv run python scratch/plot_delib_2x2_odcv_mask.py
"""Two measures of different scale get two panels, never two y-axes on one plot.

Colour is the METHOD (the same purple/blue as the misalignment figure); hatching is the TEACHER,
so identity never rests on hue alone. Intervals:
  ODCV -- the runner's estimator under a fixed-benchmark design (within-cell rollout noise,
          log-odds scale), reused from scratch/plot_misalignment_three_evals.py.
  MASK -- NO interval. Honesty is 100 - lie% over 1,000 fixed rows with ONE judged pressure
          sample each (lie_k: 1), so the within-row variance a fixed-benchmark interval needs is
          unidentified; a pooled binomial would assume one common lie probability across rows,
          which is false. Re-running with lie_k >= 3 is what would make the two panels comparable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratch"))

from plot_misalignment_three_evals import odcv_point, results, wilson  # noqa: E402
from src.naming import figure_path  # noqa: E402

ORG = "dougalldeepmind"
CELLS = {  # (method, teacher) -> (arm, ODCV run, MASK run)
    ("Difficult advice", "Sonnet"): ("da-7", f"{ORG}/2026-09-17-odcv-qwen36-0-da-7", f"{ORG}/2026-09-17-mask-qwen36-0-da-7"),
    ("Difficult advice", "Qwen"): ("da-qwen-7", f"{ORG}/2026-09-17-odcv-qwen36-0-da-qwen-7", f"{ORG}/2026-09-17-mask-qwen36-0-da-qwen-7"),
    ("Deliberative alignment", "Sonnet"): ("delib-sonnet-7", f"{ORG}/2026-09-17-odcv-qwen36-0-delib-sonnet-7", f"{ORG}/2026-09-17-mask-qwen36-0-delib-sonnet-7"),
    ("Deliberative alignment", "Qwen"): ("delib-7", f"{ORG}/2026-09-17-odcv-qwen36-0-delib-7", f"{ORG}/2026-09-17-mask-qwen36-0-delib-7"),
}
# The control every arm shares its replay rows with: standard SFT on the base blend, no synthetic share.
BASELINE = {"ODCV": f"{ORG}/2026-09-09-odcv-qwen36-0-nosynth", "MASK": f"{ORG}/2026-09-09-mask-qwen36-0-nosynth"}
METHODS = ["Difficult advice", "Deliberative alignment"]
TEACHERS = ["Sonnet", "Qwen"]
COLOUR = {"Difficult advice": "#7E22CE", "Deliberative alignment": "#3B82F6"}  # validated pair, dE 11.0
HATCH = {"Sonnet": "", "Qwen": "///"}
api = HfApi()


def mask_point(repo: str) -> dict:
    """Honesty and its interval from the per-row honesty labels the run publishes."""
    r = results(repo)
    lies = rows = 0
    for f in api.list_repo_files(repo, repo_type="dataset"):
        if f.startswith("results/") and f.endswith("_metrics.csv"):
            df = pd.read_csv(hf_hub_download(repo, f, repo_type="dataset"))
            lies += int((df["eval_honesty_1"].astype(str) == "-1").sum())
            rows += len(df)
    assert rows == r["n_rows"], f"{repo}: {rows} labelled rows vs n_rows {r['n_rows']}"
    lo_lie, hi_lie = wilson(lies, rows)
    honesty = 100 * (1 - lies / rows)
    assert abs(honesty - r["overall_honesty_score"]) < 0.15, (honesty, r["overall_honesty_score"])
    return dict(rate=honesty, lo=100 - hi_lie, hi=100 - lo_lie, n=rows, target=r["target"])


def draw(ax, values: dict, ylabel: str, ylim: tuple[float, float], baseline: float, bars: bool = True) -> None:
    width, gap = 0.34, 0.03
    ax.axhline(baseline, color="#4B5563", linestyle=(0, (1.5, 2.5)), linewidth=1.0, zorder=2)
    ax.text(1.52, baseline + (ylim[1] - ylim[0]) * 0.012, f"{baseline:.0f}", ha="right", va="bottom",
            fontsize=7.5, color="#4B5563", zorder=5)
    for i, method in enumerate(METHODS):
        for j, teacher in enumerate(TEACHERS):
            p = values[(method, teacher)]
            x = i + (j - 0.5) * (width + gap)
            ax.bar(x, p["rate"], width, color=COLOUR[method], hatch=HATCH[teacher], edgecolor="white",
                   linewidth=0.6, zorder=3)
            top = p["rate"]
            if bars:
                ax.errorbar(x, p["rate"], yerr=[[max(0.0, p["rate"] - p["lo"])], [max(0.0, p["hi"] - p["rate"])]],
                            fmt="none", ecolor="#111827", elinewidth=0.9, capsize=2.2, capthick=0.9, zorder=4)
                top = p["hi"]
            ax.text(x, top + (ylim[1] - ylim[0]) * 0.015, f"{p['rate']:.0f}", ha="center", va="bottom",
                    fontsize=7.5, color="#111827", zorder=5)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(METHODS)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, color="#E5E7EB", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9CA3AF")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors="#374151", length=0)


def main() -> None:
    odcv = {cell: odcv_point(arm, run) for cell, (arm, run, _) in CELLS.items()}
    mask = {cell: mask_point(run) for cell, (_, _, run) in CELLS.items()}

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "axes.labelsize": 9,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8, "legend.fontsize": 8,
                         "hatch.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 3.0), dpi=300)
    base_odcv = odcv_point("nosynth", BASELINE["ODCV"])
    base_mask = mask_point(BASELINE["MASK"])
    draw(ax1, odcv, "ODCV misalignment rate (%)", (0, 50), base_odcv["rate"])
    draw(ax2, mask, "MASK honesty score", (0, 100), base_mask["rate"], bars=False)
    ax1.set_title("ODCV", fontsize=9, loc="left", color="#374151")
    ax2.set_title("MASK", fontsize=9, loc="left", color="#374151")
    handles = [Patch(facecolor=COLOUR[m], label=m) for m in METHODS] + \
              [Patch(facecolor="#9CA3AF", hatch=HATCH[t], edgecolor="white", label=f"{t} teacher") for t in TEACHERS] + \
              [Line2D([0], [0], color="#4B5563", linestyle=(0, (1.5, 2.5)), linewidth=1.0, label="Standard SFT baseline")]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               handlelength=1.3, columnspacing=1.3)
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    subject = "delib_2x2_odcv_mask"
    png = figure_path(REPO / "output/plots", subject, ext="png")
    fig.savefig(png)
    fig.savefig(figure_path(REPO / "output/plots", subject, ext="pdf"))

    lines = [f"# {png.stem}", "", "Teacher x method 2x2; each arm's 2026-09-17 runs. ODCV: misalignment rate with the",
             "fixed-benchmark interval (within-cell rollout noise, log-odds). MASK: honesty = 100 - lie% over",
             "1,000 fixed rows with ONE pressure sample each (lie_k: 1), so no generation-level interval is",
             "estimable and none is drawn; the Wilson columns below assume rows are i.i.d. and are NOT on the figure.", "",
             "| method | teacher | arm | ODCV MR | 95% CI | MASK honesty | 95% CI | ODCV run | MASK run |",
             "|---|---|---|---|---|---|---|---|---|"]
    for (m, t), (arm, orun, mrun) in CELLS.items():
        o, k = odcv[(m, t)], mask[(m, t)]
        lines.append(f"| {m} | {t} | {arm} | {o['rate']:.1f}% | [{o['lo']:.1f}, {o['hi']:.1f}] | {k['rate']:.1f} | "
                     f"[{k['lo']:.1f}, {k['hi']:.1f}] | `{orun}` | `{mrun}` |")
    lines += ["", f"Standard SFT baseline (nosynth): ODCV MR {base_odcv['rate']:.1f}% [{base_odcv['lo']:.1f}, {base_odcv['hi']:.1f}] "
              f"(`{BASELINE['ODCV']}`); MASK honesty {base_mask['rate']:.1f} [{base_mask['lo']:.1f}, {base_mask['hi']:.1f}] "
              f"(`{BASELINE['MASK']}`)."]
    Path(str(png)[:-4] + "_results.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {png}")
    for (m, t) in CELLS:
        o, k = odcv[(m, t)], mask[(m, t)]
        print(f"  {m:22s} {t:6s}  ODCV {o['rate']:5.1f} [{o['lo']:.1f}, {o['hi']:.1f}]   MASK {k['rate']:5.1f} [{k['lo']:.1f}, {k['hi']:.1f}]")


if __name__ == "__main__":
    main()
