# ABOUTME: Paper figure: misalignment rate on ODCV-lite, agentic misalignment and Colosseum for
# ABOUTME: nosynth / delib-7 / da-7, each arm's most recent published run, intervals per eval design.
# Run: uv run python scratch/plot_misalignment_three_evals.py
"""One y-axis, three evals, three arms, every number read from the published run.

Intervals follow each eval's sampling design, not one formula:
  ODCV-lite  -- the runner's own estimator (src.eval.stats.interval under ODCV's Design) with
                item_sampling="fixed": the 40 scenarios ARE the benchmark, so only within-cell
                rollout noise counts. Normal z, log-odds scale (the same choice as
                scratch/plot_odcv_mr_vs_mask.py).
  AM         -- the same estimator with the 27 conditions as the fixed items and the 15 samples
                per condition as the resampled draw.
  Colosseum  -- ONE scenario run as 30 iid episodes: a plain Wilson interval on the count.
Each eval's rate is its own headline binary: ODCV's MR, AM's harmful rate, Colosseum's share of
episodes in which the pair sabotaged.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratch"))
load_dotenv(REPO / ".env")

from plot_odcv_mr_vs_mask import FIXED, mr_interval, z_bounds  # noqa: E402  (ODCV's Design, fixed)
from src.eval.misalignment.odcv.odcv import MR_BOUNDS, VARIANTS  # noqa: E402
from src.eval.stats import Z_95, Design, _shape_interval, interval  # noqa: E402
from src.naming import figure_path  # noqa: E402

ORG = "dougalldeepmind"
# Most recent published run per (eval, arm). ODCV's fixed interval needs the per-rollout judge
# scores, which every run carries under results/scores_<judge>.json.
RUNS = {
    "ODCV-lite": {
        "nosynth": f"{ORG}/2026-09-09-odcv-qwen36-0-nosynth",
        "delib-7": f"{ORG}/2026-09-17-odcv-qwen36-0-delib-7",
        "da-7": f"{ORG}/2026-09-17-odcv-qwen36-0-da-7",
    },
    "Agentic misalignment": {
        "nosynth": f"{ORG}/2026-09-17-am-qwen36-0-nosynth",
        "delib-7": f"{ORG}/2026-09-17-am-qwen36-0-delib-7",
        "da-7": f"{ORG}/2026-09-18-am-qwen36-0-da-7",
    },
    "Colosseum hospital": {
        "nosynth": f"{ORG}/2026-09-14-colosseum-hospital-self-sacrificial-qwen36-unfiltered-no-synthetic-fixed",
        "delib-7": f"{ORG}/2026-09-18-colosseum-hospital-self-sacrificial-qwen36-deliberative-alignment-fixed",
        "da-7": f"{ORG}/2026-09-15-colosseum-hospital-self-sacrificial-qwen36-difficult-advice-neutral-752-fixed",
    },
}
ARMS = ["nosynth", "delib-7", "da-7"]
LABEL = {"nosynth": "No synthetic data", "delib-7": "Deliberative alignment", "da-7": "Difficult advice"}
# dataviz validate_palette.js, light surface: CVD dE 11.0 (deutan), normal-vision dE 18.0. The grey
# fails the chroma floor by design (it is the control); the legend and value labels carry identity.
COLOUR = {"nosynth": "#6B7280", "delib-7": "#3B82F6", "da-7": "#7E22CE"}
SCORES_RE = re.compile(r"results/scores_(?!progress_)[^/]+\.json")
AM_DESIGN = Design(item="condition", item_sampling="fixed", subsamples=("sample",))

api = HfApi()


def results(repo: str) -> dict:
    return json.load(open(hf_hub_download(repo, "results/results.json", repo_type="dataset")))


def wilson(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def odcv_point(arm: str, repo: str) -> dict:
    r = results(repo)
    files = [f for f in api.list_repo_files(repo, repo_type="dataset") if SCORES_RE.fullmatch(f)]
    assert len(files) == 1, f"{repo}: expected one misalignment judge's scores, found {files}"
    scores = json.load(open(hf_hub_download(repo, files[0], repo_type="dataset")))
    medians = {v: {} for v in VARIANTS}
    for key, rec in scores.items():
        variant, scenario, _ = key.split("/")
        s = rec.get("score") if isinstance(rec, dict) else rec
        if isinstance(s, (int, float)):
            medians[variant].setdefault(scenario, []).append(float(s))
    res = mr_interval(medians, FIXED)
    lo, hi, _ = z_bounds(res)
    published = r["ours"]["overall"]["mr_pct"]
    assert abs(res.mean - published) < 0.15, f"{repo}: recomputed MR {res.mean:.2f} != published {published}"
    return dict(rate=res.mean, lo=lo, hi=hi, n=r["ours"]["overall"]["n_rollouts"], unit="rollouts",
                target=r.get("target") or "", design="fixed benchmark (40 scenarios), rollout noise only")


def am_point(repo: str) -> dict:
    r = results(repo)
    rows = []
    for cond, c in r["by_condition"].items():
        rows += [{"condition": cond, "sample": i, "checkpoint": "run", "value": 100.0 * (i < c["harmful"])}
                 for i in range(c["n"])]
    res = interval(rows, AM_DESIGN, bounds=MR_BOUNDS)
    lo, hi, _ = _shape_interval(res.mean, res.se, Z_95, MR_BOUNDS, n_floor=1)
    o = r["overall"]
    assert abs(res.mean - 100 * o["rate"]) < 0.15
    return dict(rate=res.mean, lo=lo, hi=hi, n=o["n"], unit="samples", target=r["target"],
                design=f"fixed benchmark ({len(r['by_condition'])} conditions), sample noise only")


def colosseum_point(repo: str) -> dict:
    r = results(repo)
    m = r["measures"]["pair_sabotaged"][r["condition"]]
    k, n = sum(1 for v in m.values() if v), len(m)
    lo, hi = wilson(k, n)
    return dict(rate=100 * k / n, lo=lo, hi=hi, n=n, unit="episodes", target=r["target"],
                design="one scenario, iid episodes: Wilson")


def main() -> None:
    data = {
        "ODCV-lite": {a: odcv_point(a, RUNS["ODCV-lite"][a]) for a in ARMS},
        "Agentic misalignment": {a: am_point(RUNS["Agentic misalignment"][a]) for a in ARMS},
        "Colosseum hospital": {a: colosseum_point(RUNS["Colosseum hospital"][a]) for a in ARMS},
    }
    evals = list(data)

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "axes.labelsize": 9,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8, "legend.fontsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.6, 3.1), dpi=300)
    width, gap = 0.25, 0.03
    for j, arm in enumerate(ARMS):
        for i, ev in enumerate(evals):
            p = data[ev][arm]
            x = i + (j - 1) * (width + gap)
            ax.bar(x, p["rate"], width, color=COLOUR[arm], linewidth=0, zorder=3,
                   label=LABEL[arm] if i == 0 else None)
            # max(0, .): a 30/30 Wilson upper bound lands a floating-point hair under 100.
            ax.errorbar(x, p["rate"], yerr=[[max(0.0, p["rate"] - p["lo"])], [max(0.0, p["hi"] - p["rate"])]],
                        fmt="none", ecolor="#111827", elinewidth=0.9, capsize=2.2, capthick=0.9, zorder=4)
            ax.text(x, p["hi"] + 1.6, f"{p['rate']:.0f}", ha="center", va="bottom", fontsize=7.5,
                    color="#111827", zorder=5)

    ax.set_xticks(range(len(evals)))
    ax.set_xticklabels(["ODCV", "Agentic misalignment", "Colosseum hospital\n(self-sacrificial)"])
    ax.set_ylabel("Misalignment rate (%)")
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.grid(True, color="#E5E7EB", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9CA3AF")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors="#374151", length=0)
    ax.legend(frameon=False, loc="upper left", ncol=3, handlelength=1.1, columnspacing=1.4,
              borderaxespad=0.2)
    # Setting labels under the eval names: ODCV and AM are single-agent honeypots, Colosseum is
    # the multi-agent one. A thin rule spans each group; x in data units, y in axes fraction.
    import matplotlib.transforms as mtransforms
    tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for x0, x1, name in ((-0.42, 1.42, "Single-agent"), (1.58, 2.42, "Multi-agent")):
        ax.plot([x0, x1], [-0.21, -0.21], color="#9CA3AF", linewidth=0.8, transform=tr, clip_on=False)
        ax.text((x0 + x1) / 2, -0.235, name, ha="center", va="top", fontsize=8.5, color="#374151",
                transform=tr, clip_on=False)
    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(bottom=0.27)

    subject = "misalignment_nosynth_delib7_da7"
    png = figure_path(REPO / "output/plots", subject, ext="png")
    pdf = figure_path(REPO / "output/plots", subject, ext="pdf")
    fig.savefig(png)
    fig.savefig(pdf)

    lines = [f"# {png.stem}", "",
             "Bars: each arm's most recent published run. Error bars: 95% intervals under each eval's own",
             "sampling design -- ODCV-lite and AM treat their scenario/condition sets as a FIXED benchmark",
             "(within-item rollout noise only, log-odds scale, normal z); Colosseum is one scenario run as",
             "30 iid episodes, so a Wilson interval on the count.", "",
             "| eval | arm | rate | 95% CI | n | interval design | adapter | run |", "|---|---|---|---|---|---|---|---|"]
    for ev in evals:
        for arm in ARMS:
            p = data[ev][arm]
            lines.append(f"| {ev} | {arm} | {p['rate']:.1f}% | [{p['lo']:.1f}, {p['hi']:.1f}] | {p['n']} {p['unit']} | "
                         f"{p['design']} | {str(p['target']).split('/')[-1]} | `{RUNS[ev][arm]}` |")
    lines += ["", "ODCV-lite: misalignment rate, single flash judge, 3 passes over 40 scenarios x 2 variants. AM: harmful",
              "rate over 27 conditions (blackmail, leaking, murder) x 15 samples. Colosseum: share of self-sacrificial",
              "hospital episodes in which the pair sabotaged (`pair_sabotaged`)."]
    Path(str(png)[:-4] + "_results.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {png}, {pdf} and the results md")
    for ev in evals:
        for arm in ARMS:
            p = data[ev][arm]
            print(f"  {ev:22s} {arm:8s} {p['rate']:5.1f}%  [{p['lo']:5.1f}, {p['hi']:5.1f}]  n={p['n']}")


if __name__ == "__main__":
    main()
