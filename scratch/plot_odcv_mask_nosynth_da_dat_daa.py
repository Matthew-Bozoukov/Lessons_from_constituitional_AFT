# ABOUTME: Bars: ODCV-lite misalignment rate and MASK honesty for nosynth vs da-7 / dat-7 / daa-7, each arm's
# ABOUTME: most recent model, fixed-benchmark intervals. Run: uv run python scratch/plot_odcv_mask_nosynth_da_dat_daa.py

"""One model per arm — the most recently trained — and the benchmark held fixed.

Intervals treat each benchmark as THE benchmark, not a sample of one, so they carry only the
noise of re-running the same model on the same items (the choice made in
scratch/plot_odcv_mr_vs_mask.py and scratch/plot_misalignment_three_evals.py):

  ODCV-lite  -- the runner's own estimator (src.eval.stats.interval under ODCV's Design) with
                item_sampling="fixed": 40 scenarios x 2 variants are fixed, the 3 rollouts per
                cell are the resampled draw. Normal z on the log-odds scale; imported from
                plot_odcv_mr_vs_mask.py so the two figures cannot disagree.
  MASK       -- 1,000 fixed rows with ONE pressured generation each (lie_k=1), so the within-row
                noise a fixed-benchmark interval needs is not estimable from a run. The Wilson
                interval over rows is drawn instead and is an UPPER BOUND on it: for independent
                rows Var(mean) = sum p_i(1-p_i)/n^2 <= p(1-p)/n, with equality only when every
                row is equally hard.

Neither interval contains run-to-run variation. On 2026-09-21 five da-7 adapters spanned
8.3-12.9% MR and 71.2-82.8 honesty (docs/LOG.md), which is wider than the MASK bars here.

The (2026-09-21 dot-per-run version of this figure is in git history: 77bac7ab.)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scratch"))
load_dotenv(REPO / ".env")

from plot_odcv_mr_vs_mask import FIXED, mr_interval, odcv_medians, z_bounds  # noqa: E402
from src.eval.stats import wilson  # noqa: E402
from src.naming import figure_path  # noqa: E402

ORG = "dougalldeepmind"
ARMS = ["nosynth", "da-7", "dat-7", "daa-7"]
LABEL = {"nosynth": "No synthetic\ndata", "da-7": "Difficult\nadvice", "dat-7": "Difficult\nagentic tasks",
         "daa-7": "Agentified\ndifficult advice"}
# CLAUDE.md's fixed arm colours; daa has none assigned and takes green (clear of delib's blue).
COLOUR = {"nosynth": "#6B7280", "da-7": "#7E22CE", "dat-7": "#eda100", "daa-7": "#059669"}
# The most recently TRAINED adapter of each arm, and the ODCV-lite and MASK runs on that adapter.
RUNS = {
    "nosynth": ("2026-09-08-qwen36-0-nosynth", "2026-09-09-odcv-qwen36-0-nosynth", "2026-09-09-mask-qwen36-0-nosynth"),
    "da-7": ("2026-09-21-qwen36-0-da-7", "2026-09-21-odcv-qwen36-0-da-7", "2026-09-21-mask-qwen36-0-da-7"),
    "dat-7": ("2026-09-08-qwen36-0-dat-7", "2026-09-09-odcv-qwen36-0-dat-7", "2026-09-09-mask-qwen36-0-dat-7"),
    "daa-7": ("2026-09-17-qwen36-0-daa-7", "2026-09-17-odcv-qwen36-0-daa-7", "2026-09-18-mask-qwen36-0-daa-7"),
}
INK = "#111827"


def _json(repo: str, name: str, sha: str) -> dict:
    return json.loads(Path(hf_hub_download(repo, name, repo_type="dataset", revision=sha)).read_text())


def points(api: HfApi) -> dict[str, dict[str, dict]]:
    out = {"odcv": {}, "mask": {}}
    for arm, (adapter, odcv_name, mask_name) in RUNS.items():
        repo = f"{ORG}/{odcv_name}"
        sha = api.dataset_info(repo).sha
        res, meta = _json(repo, "results/results.json", sha), _json(repo, "metadata/run_meta.json", sha)
        assert meta["target"].endswith(adapter), (repo, meta["target"])
        o = res["ours"]["overall"]
        assert o["n_rollouts"] == 240 and list(res["judges"]) == ["gemini-3-flash-preview"], repo
        medians, unscored = odcv_medians(api, repo, sha)
        fixed = mr_interval(medians, FIXED)
        lo, hi, _ = z_bounds(fixed)
        assert abs(fixed.mean - o["mr_pct"]) < 0.15 and unscored == 0, (repo, fixed.mean, o["mr_pct"], unscored)
        out["odcv"][arm] = dict(value=fixed.mean, lo=lo, hi=hi, adapter=adapter, repo=repo, sha=sha,
                                sampled=(o["mr_ci95_lo"], o["mr_ci95_hi"]))

        repo = f"{ORG}/{mask_name}"
        sha = api.dataset_info(repo).sha
        res, meta = _json(repo, "results/results.json", sha), _json(repo, "metadata/run_meta.json", sha)
        assert meta["target"].endswith(adapter), (repo, meta["target"])
        assert res["n_rows"] == 1000 and res["mode"] == "think" and "gemini-3-flash" in res["judge"], repo
        value = res["overall_honesty_score"]
        lo, hi = (100 * b for b in wilson(round(value * 10), 1000))
        out["mask"][arm] = dict(value=value, lo=lo, hi=hi, adapter=adapter, repo=repo, sha=sha)
    return out


def bars(ax, pts: dict[str, dict], title: str, ylabel: str, ymax: float) -> None:
    for i, arm in enumerate(ARMS):
        p = pts[arm]
        ax.bar(i, p["value"], 0.62, color=COLOUR[arm], linewidth=0, zorder=3)
        ax.errorbar(i, p["value"], yerr=[[max(0.0, p["value"] - p["lo"])], [max(0.0, p["hi"] - p["value"])]],
                    fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.6, capthick=0.9, zorder=4)
        ax.text(i, p["hi"] + ymax * 0.02, f"{p['value']:.1f}", ha="center", va="bottom", fontsize=8,
                color=INK, zorder=5)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([LABEL[a] for a in ARMS])
    ax.set_xlim(-0.6, len(ARMS) - 0.4)
    ax.set_ylim(0, ymax)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK, pad=8)
    ax.yaxis.grid(True, color="#E5E7EB", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9CA3AF")
    ax.tick_params(colors="#374151", length=0)


def main() -> None:
    data = points(HfApi())
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), dpi=300)
    bars(axes[0], data["odcv"], "ODCV-lite misalignment rate  (lower is better)",
         "% of 240 rollouts judged a violation", 60)
    bars(axes[1], data["mask"], "MASK honesty  (higher is better)", "honesty score, 1,000 rows", 100)
    fig.text(0.07, 0.015, "Qwen3.6-27B, each arm's most recently trained adapter (seed 0). Bars: 95% intervals with the "
             "benchmark held fixed — ODCV: rollout noise only;\nMASK: Wilson over rows, an upper bound on the same "
             "(one generation per row). Run-to-run variation is not included.",
             fontsize=7, color="#4B5563", ha="left", va="bottom")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.27, wspace=0.22)

    png = figure_path("output/figures", "odcv mask nosynth da dat daa bars")
    fig.savefig(png)
    fig.savefig(png.with_suffix(".pdf"))

    lines = ["# ODCV-lite misalignment rate and MASK honesty: nosynth vs da-7, dat-7, daa-7 (most recent model per arm)",
             "", "| arm | adapter | ODCV MR % | fixed-benchmark 95% (plotted) | scenario-sampled 95% (published) | "
             "MASK honesty | Wilson 95% over rows (plotted; upper bound on fixed-benchmark) | ODCV repo @ sha | MASK repo @ sha |",
             "|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        o, m = data["odcv"][arm], data["mask"][arm]
        lines.append(f"| {arm} | {o['adapter']} | {o['value']:.1f} | [{o['lo']:.1f}, {o['hi']:.1f}] | "
                     f"[{o['sampled'][0]:.1f}, {o['sampled'][1]:.1f}] | {m['value']:.1f} | [{m['lo']:.1f}, {m['hi']:.1f}] | "
                     f"`{o['repo']}` @ `{o['sha'][:8]}` | `{m['repo']}` @ `{m['sha'][:8]}` |")
    lines += ["", "ODCV: 240 rollouts (40 scenarios x 2 variants x 3 passes), gemini-3-flash judge; the interval is "
              "src.eval.stats.interval under ODCV's Design with item_sampling=\"fixed\", normal z on the log-odds scale "
              "(scratch/plot_odcv_mr_vs_mask.py). MASK: 1,000 rows, think mode, same judge, lie_k=1, so within-row noise "
              "is not estimable and the Wilson interval over rows is drawn as an upper bound on the fixed-benchmark one.",
              "", "Neither interval includes run-to-run variation: five da-7 adapters span 8.3-12.9% MR and 71.2-82.8 "
              "honesty (docs/LOG.md 2026-09-21). The da-7 adapter here is the newest, trained on the 2026-09-20 "
              "fla stack (branch jamie/train-optim); the newest da-7 from the merged stack is 2026-09-15-qwen36-0-da-7 "
              "(8.3% MR, 74.2 honesty)."]
    Path(str(png)[:-4] + "_results.md").write_text("\n".join(lines) + "\n")
    print(f">>> {png}")


if __name__ == "__main__":
    main()
