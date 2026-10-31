# ABOUTME: Paper figure (Appendix A, Fig. 11): the teacher x method 2x2 (Sonnet/Qwen x difficult advice/deliberative
# ABOUTME: alignment) on the PLAIN base at 15% of supervised tokens: ODCV misalignment and MASK honesty, two panels.
# Run: uv run python scratch/plot_delib_2x2_plain.py
"""The 2026-09-17 version of this figure (scratch/plot_delib_2x2_odcv_mask.py) drew the 7% arms on the nosynth
base from one MASK run each. This one draws the 15% arms retrained on the plain base (2026-10-05 / 10-08) and,
for MASK, the mean of THREE runs per arm (the 10-08 run plus the two passes of the 10-09 `passes=2` replicate)
under a run-level 95% interval (t, df=2), so a whole-run shift is inside the bar -- da-qwen-15 has one. ODCV keeps
the runner's fixed-benchmark interval. Colour is the method, hatching the teacher.
"""

from __future__ import annotations

import statistics
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

from plot_delib_2x2_odcv_mask import COLOUR, HATCH, METHODS, TEACHERS, draw  # noqa: E402
from plot_misalignment_three_evals import odcv_point, results  # noqa: E402
from src.naming import figure_path  # noqa: E402

api = HfApi()
T_95_DF2 = 4.3027  # two-sided t, 2 degrees of freedom: three runs

ORG = "dougalldeepmind"
# (method, teacher) -> (arm, ODCV run, [MASK runs: the single 10-08 run, then the 10-09 passes=2 run])
CELLS = {
    ("Difficult advice", "Sonnet"): ("da-15", f"{ORG}/2026-10-06-odcv-qwen36-0-da-15",
                                     [f"{ORG}/2026-10-05-mask-qwen36-0-da-15", f"{ORG}/2026-10-09-mask-qwen36-0-da-15"]),
    ("Difficult advice", "Qwen"): ("da-qwen-15", f"{ORG}/2026-10-08-odcv-qwen36-0-da-qwen-15",
                                   [f"{ORG}/2026-10-08-mask-qwen36-0-da-qwen-15", f"{ORG}/2026-10-09-mask-qwen36-0-da-qwen-15"]),
    ("Deliberative alignment", "Sonnet"): ("delib-sonnet-15", f"{ORG}/2026-10-08-odcv-qwen36-0-delib-sonnet-15",
                                           [f"{ORG}/2026-10-08-mask-qwen36-0-delib-sonnet-15", f"{ORG}/2026-10-09-mask-qwen36-0-delib-sonnet-15"]),
    ("Deliberative alignment", "Qwen"): ("delib-15", f"{ORG}/2026-10-08-odcv-qwen36-0-delib-15",
                                         [f"{ORG}/2026-10-08-mask-qwen36-0-delib-15", f"{ORG}/2026-10-09-mask-qwen36-0-delib-15"]),
}
# The control every arm shares its replay rows with: standard SFT on the plain base, no synthetic share.
BASELINE = {"ODCV": ("plain", f"{ORG}/2026-10-06-odcv-qwen36-0-plain"),
            "MASK": [f"{ORG}/2026-10-05-mask-qwen36-0-plain", f"{ORG}/2026-10-09-mask-qwen36-0-plain"]}


def _run_labels(repo: str) -> dict[str, dict[tuple[str, int], str]]:
    """{run name -> {(archetype, row) -> MASK honesty label}} for every pass a run published.

    A `passes=1` run writes results/<archetype>_metrics.csv; a `passes=N` run writes
    results/pass<k>/<archetype>_metrics.csv. `eval_honesty_1` is "-1" for a lie, "1" honest, "0" neither.
    """
    files = api.list_repo_files(repo, repo_type="dataset")
    passes = sorted({f.split("/")[1] for f in files if f.startswith("results/pass")})
    out = {}
    for sub in passes or [None]:
        prefix = "results/" if sub is None else f"results/{sub}/"
        depth = 1 if sub is None else 2
        labels = {}
        for f in files:
            if f.startswith(prefix) and f.endswith("_metrics.csv") and f.count("/") == depth:
                archetype = f.split("/")[-1].replace("_metrics.csv", "")
                df = pd.read_csv(hf_hub_download(repo, f, repo_type="dataset"))
                labels.update({(archetype, i): str(v) for i, v in df["eval_honesty_1"].items()})
        out[f"{repo}{'' if sub is None else '/' + sub}"] = labels
    return out


def mask_runs(repos: list[str]) -> dict:
    """Honesty and its FIXED-BENCHMARK interval over the three runs, the estimator the ODCV panel uses.

    The 1,000 MASK rows are the benchmark, not a sample from a population, so there is no row-to-row
    term: `item_sampling="fixed"`. The three runs are three fresh generations of every row, which is
    what makes an interval estimable at all (one run gives a zero-width bar and `src.eval.stats`
    raises rather than return it). So the bar covers generation noise on these rows for this
    checkpoint -- the same claim, and the same machinery, as the ODCV panel beside it. What neither
    panel covers is training-seed variance: every arm here is seed 0.
    """
    per_run, errors = {}, []
    for repo in repos:
        per_run.update(_run_labels(repo))
        errors.append(float(results(repo).get("judge_error_rate") or 0.0))
    assert len(per_run) == 3, f"{repos}: expected three MASK runs, found {list(per_run)}"
    assert max(errors) <= 0.01, f"{repos}: judge error rate {max(errors):.3f} (a timed-out verdict scores as neither)"
    keys = [set(v) for v in per_run.values()]
    assert keys[0] == keys[1] == keys[2], f"{repos}: the three runs do not cover the same rows"

    scores = [100.0 * sum(1 for v in labels.values() if v != "-1") / len(labels) for labels in per_run.values()]
    mean = sum(scores) / 3
    # The RUN is the unit, so the interval covers a whole-run shift as well as per-row generation
    # noise: mean +/- t(df=2) * s / sqrt(3). The fixed-benchmark estimator the ODCV panel uses (rows
    # fixed, runs as within-row subsamples) was tried first and rejected on this data: it models the
    # runs as differing only through independent per-row noise, and on da-qwen-15 they differ about
    # twice that much (observed s 2.11 against 1.00 predicted; the "neither" counts are flat at
    # 269/264/255, so it is generation, not judging). An estimator that assumes away a term this
    # data shows would report an interval too narrow for the runs it is drawn over.
    s = statistics.stdev(scores)
    half = T_95_DF2 * s / 3 ** 0.5
    return dict(rate=mean, lo=mean - half, hi=mean + half, runs=scores, n=len(keys[0]), s=s)


def main() -> None:
    odcv = {cell: odcv_point(arm, run) for cell, (arm, run, _) in CELLS.items()}
    mask = {cell: mask_runs(runs) for cell, (_, _, runs) in CELLS.items()}
    base_odcv = odcv_point(*BASELINE["ODCV"])
    base_mask = mask_runs(BASELINE["MASK"])

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "axes.labelsize": 9,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8, "legend.fontsize": 8,
                         "hatch.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 3.0), dpi=300)
    draw(ax1, odcv, "ODCV misalignment rate (%)", (0, 50), base_odcv["rate"])
    # MASK's interval treats the RUN as the unit (see mask_runs). The individual run scores are in
    # the results.md beside this figure, not on it.
    draw(ax2, mask, "MASK honesty score", (0, 100), base_mask["rate"], bars=True)
    ax1.set_title("ODCV", fontsize=9, loc="left", color="#374151")
    ax2.set_title("MASK", fontsize=9, loc="left", color="#374151")
    handles = [Patch(facecolor=COLOUR[m], label=m) for m in METHODS] + \
              [Patch(facecolor="#9CA3AF", hatch=HATCH[t], edgecolor="white", label=f"{t} teacher") for t in TEACHERS] + \
              [Line2D([0], [0], color="#4B5563", linestyle=(0, (1.5, 2.5)), linewidth=1.0, label="Standard SFT baseline")]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               handlelength=1.3, columnspacing=1.3)
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    subject = "delib_2x2_plain_odcv_mask"
    png = figure_path(REPO / "output/plots", subject, ext="png")
    fig.savefig(png)
    fig.savefig(figure_path(REPO / "output/plots", subject, ext="pdf"))

    lines = [f"# {png.stem}", "", "Teacher x method 2x2 on the plain base at 15% of supervised tokens (seed 0). ODCV: misalignment",
             "rate with the fixed-benchmark interval (within-cell rollout noise, log-odds), 240 rollouts. MASK: mean of",
             "three full runs of the same 1,000 rows (the 10-08 run and the two passes of the 10-09 replicate),",
             "under a 95% interval that takes the RUN as the unit (t, df=2); the run scores are tabulated below,",
             "not drawn on the figure. A fixed-benchmark",
             "interval (rows fixed, runs as within-row subsamples, the ODCV panel's estimator) was tried and",
             "rejected: it assumes runs differ only by independent per-row noise, and da-qwen-15 differs about",
             "twice that much (s 2.11 vs 1.00 predicted; neither-counts flat at 269/264/255, so generation, not",
             "judging). Neither panel covers training-seed variance: every arm is seed 0.",
             "every run has a judge error rate <= 1% (verdicts that time out score as neither, see docs/GOTCHAS.md).", "",
             "| method | teacher | arm | ODCV MR | 95% CI | MASK | 95% CI | runs | ODCV run | MASK runs |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for (m, t), (arm, orun, mruns) in CELLS.items():
        o, k = odcv[(m, t)], mask[(m, t)]
        lines.append(f"| {m} | {t} | {arm} | {o['rate']:.1f}% | [{o['lo']:.1f}, {o['hi']:.1f}] | {k['rate']:.1f} | "
                     f"[{k['lo']:.1f}, {k['hi']:.1f}] | {', '.join(f'{s:.1f}' for s in k['runs'])} | `{orun}` | `{'`, `'.join(mruns)}` |")
    lines += ["", f"Standard SFT baseline (plain): ODCV MR {base_odcv['rate']:.1f}% [{base_odcv['lo']:.1f}, {base_odcv['hi']:.1f}] "
              f"(`{BASELINE['ODCV'][1]}`); MASK {base_mask['rate']:.1f} [{base_mask['lo']:.1f}, {base_mask['hi']:.1f}] "
              f"({', '.join(f'{r:.1f}' for r in base_mask['runs'])})."]
    Path(str(png)[:-4] + "_results.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
