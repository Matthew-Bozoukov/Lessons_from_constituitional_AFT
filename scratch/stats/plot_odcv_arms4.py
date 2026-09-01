# ABOUTME: Matthew's four-arm ODCV incentivized-MR bar chart, rebuilt with src/eval/stats.py
# ABOUTME: intervals (scenarios AND seeds sampled) instead of +/-1.96 SEM over the three seeds.
# Run: uv run python scratch/stats/plot_odcv_arms4.py

"""Four arms x three training seeds, incentivized MR, with the derived error bars.

Data: 12 ODCV eval repos in the contract layout. Everything needed is in `results/scores_*.json`
-- keys are `<variant>/<scenario>` or `<variant>/<scenario>/rollout_NNN` -- so the rollouts
themselves are never downloaded. A cell is a violation when the MEDIAN judge severity is >= 3.

ONE rollout per (seed, scenario) cell: `rollout_000` where a repo has several passes, the bare
key where it has one. That is R=1, which the interval supports -- rollout noise then sits inside
every spread and is measured with it, it just cannot be reported separately. (`base but R=1` in
scratch/stats/simulate_coverage.py covers 93.1/93.8%, so R=1 is not what breaks these bars.)

Two intervals per arm, so the change is visible:

  ours      seeds AND scenarios sampled: SE^2 = T_A + T_B - T_C, t_nu (Satterthwaite), built on
            the log-odds scale (`bounds=(0, 100)`) so it cannot leave [0, 100]. Generalises to
            "a checkpoint from this pipeline, on a scenario drawn like these".
  Matthew's +/-1.96 * SEM over the three seed means. Scenarios FIXED, so it generalises only to
            these 40-odd stories, and 1.96 is a z where 3 seeds warrant t_2 = 4.30.
"""

from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

import fire
import matplotlib
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.eval.misalignment.odcv.odcv import (  # noqa: E402
    DESIGN_ONE_VARIANT, MR_BOUNDS, VIOLATION_THRESHOLD, median_score,
)
from src.utils import figure_path
from src.eval.stats import difference, interval  # noqa: E402
from src.utils import git_sha, write_run_meta  # noqa: E402

VARIANT = "incentivized"
SEEDS = (0, 42, 69)
ARMS = {
    "0% diff advice": {
        0:  "matboz/2026-08-19-odcv-numina-control-716-seed0",
        42: "matboz/2026-08-26-odcv-numina-control-716-seed42",
        69: "matboz/2026-08-26-odcv-numina-control-716-seed69"},
    "7% sonnet diff advice": {
        0:  "matboz/2026-08-24-odcv-synthdoc-716-seed0-5pass",
        42: "matboz/2026-08-26-odcv-synthdoc-716-seed42",
        69: "matboz/2026-08-26-odcv-synthdoc-716-seed69"},
    "7% grok diff advice": {
        0:  "LASR-Callum/2026-08-24-odcv-grok-responder-703-paired-eval",
        42: "matboz/2026-08-27-odcv-grokresp703-paired-seed42",
        69: "matboz/2026-08-27-odcv-grokresp703-paired-seed69"},
    "7% verbose diff advice": {
        0:  "matboz/2026-08-27-odcv-qwen3-6-27b-lora-t2-9284-da716-verbose-r64-dynbatch",
        42: "matboz/2026-08-27-odcv-qwen3-6-27b-lora-t2-9284-verbosecot716-r64-seed42",
        69: "matboz/2026-08-27-odcv-qwen3-6-27b-lora-t2-9284-verbosecot716-r64-seed69"},
}
# validated categorical slots 1-4 (node scripts/validate_palette.js, all checks PASS)
COLORS = ("#2a78d6", "#eb6834", "#2e9e73", "#8b5cf6")
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def severities(repo: str) -> dict[str, float]:
    """{scenario: median judge severity} for ONE rollout of the incentivized variant."""
    root = Path(snapshot_download(repo, repo_type="dataset", allow_patterns=["results/*"]))
    judges = [json.loads(f.read_text()) for f in sorted(root.glob("results/scores_*.json"))]
    assert judges, f"no judge score files in {repo}"
    scenarios = sorted({k.split("/")[1] for jd in judges for k in jd if k.startswith(f"{VARIANT}/")})
    out = {}
    for scen in scenarios:
        # rollout_000 is "rollout_0"; a single-pass repo has no suffix at all.
        scores = []
        for jd in judges:
            hit = next((jd[k] for k in (f"{VARIANT}/{scen}/rollout_000", f"{VARIANT}/{scen}")
                        if k in jd), None)
            if hit and hit.get("score") not in (None, "N/A"):
                scores.append(float(hit["score"]))
        if len(scores) == len(judges):      # every judge scored it, or the cell is dropped
            out[scen] = median_score(scores)
    return out


def main(out: str = "output/odcv_arms4") -> None:
    load_dotenv()
    raw = {arm: {seed: severities(r) for seed, r in seeds.items()} for arm, seeds in ARMS.items()}
    shared = sorted(set.intersection(*(set(s) for a in raw.values() for s in a.values())))
    assert shared, "no scenario is scored in all 12 repos"

    rows, results, per_seed = {}, {}, {}
    for arm, seeds in raw.items():
        rows[arm] = [{"checkpoint": f"seed{s}", "scenario": scen, "pass": 0,
                      "value": 100.0 * float(seeds[s][scen] >= VIOLATION_THRESHOLD)}
                     for s in SEEDS for scen in shared]
        results[arm] = interval(rows[arm], DESIGN_ONE_VARIANT, bounds=MR_BOUNDS)
        per_seed[arm] = [100.0 * sum(seeds[s][sc] >= VIOLATION_THRESHOLD for sc in shared) / len(shared)
                         for s in SEEDS]

    # Matthew's bars: SEM over the three seed means, x 1.96 (scenarios treated as fixed).
    sem = {a: statistics.stdev(v) / np.sqrt(len(v)) for a, v in per_seed.items()}

    # Same arms on each arm's OWN 3-seed scenario intersection rather than the 12-repo one.
    # Costs the pairing across arms, but keeps every scenario an arm actually has -- this is the
    # set Matthew's numbers come from, so it is how to tell a real difference from a set difference.
    own = {}
    for arm, seeds in raw.items():
        sc = sorted(set.intersection(*(set(v) for v in seeds.values())))
        r = [{"checkpoint": f"seed{s}", "scenario": c, "pass": 0,
              "value": 100.0 * float(seeds[s][c] >= VIOLATION_THRESHOLD)} for s in SEEDS for c in sc]
        own[arm] = (interval(r, DESIGN_ONE_VARIANT, bounds=MR_BOUNDS), len(sc))

    # Each treatment against the control, paired on scenario. Checkpoints are NOT paired: seed0
    # of one arm and seed0 of another are different models that share only a seed number.
    ctrl = list(ARMS)[0]
    diffs = {a: difference(rows[a], rows[ctrl], DESIGN_ONE_VARIANT)
             for a in list(ARMS)[1:]}

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = REPO / out / stamp
    dest.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.5, 6.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    xs = np.arange(len(ARMS))
    for i, (arm, r) in enumerate(results.items()):
        ax.bar(i, r.mean, .62, color=COLORS[i], zorder=2, edgecolor=SURFACE, linewidth=2)
        ax.plot([i, i], [r.lo, r.hi], color=INK, lw=2.4, zorder=5, solid_capstyle="butt")
        for y in (r.lo, r.hi):
            ax.plot([i - .085, i + .085], [y, y], color=INK, lw=2.4, zorder=5)
        m = np.mean(per_seed[arm])
        ax.plot([i + .30, i + .30], [m - 1.96 * sem[arm], m + 1.96 * sem[arm]],
                color=INK2, lw=1.4, ls=(0, (2, 1.6)), zorder=4,
                label="±1.96 SEM over seeds (Matthew's)" if i == 0 else None)
        if sem[arm] == 0:   # three seeds landed on the same value: the SEM bar has NO width
            ax.annotate("all 3 seeds identical\n→ SEM bar has zero width", xy=(i + .30, m),
                        xytext=(i + .34, m + 7.5), fontsize=8, color=INK2, ha="left",
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=.9))
        ax.scatter([i - .30] * 3, per_seed[arm], s=26, color=INK2, zorder=6, alpha=.75,
                   label="per-seed MR" if i == 0 else None)
        ax.text(i, r.hi + 1.4, f"{r.mean:.1f}%", ha="center", va="bottom", fontsize=14,
                fontweight="bold", color=INK)
        ax.text(i, -3.2, f"[{r.lo:.1f}, {r.hi:.1f}]  df {r.df:.0f}", ha="center", va="top",
                fontsize=8.5, color=INK2)

    ax.set_xticks(xs)
    ax.set_xticklabels(list(ARMS), fontsize=11, color=INK)
    ax.set_ylabel("Incentivized misalignment rate (%)", fontsize=11.5, color=INK)
    ax.set_ylim(0, max(r.hi for r in results.values()) * 1.22)
    ax.set_title("ODCV incentivized MR by arm, k=3 training seeds", fontsize=15,
                 fontweight="bold", color=INK, loc="left", pad=26)
    ax.text(0, 1.035, f"error bars: 95% CI, seeds AND scenarios sampled "
                      f"(T_A + T_B − T_C, Satterthwaite t, log-odds scale) — "
                      f"{len(shared)} shared scenarios, 1 rollout per cell",
            transform=ax.transAxes, fontsize=9.5, color=INK2)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    ax.yaxis.grid(True, color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=10)
    fig.tight_layout()
    png = figure_path(dest, f"odcv arms4 {stamp}")
    fig.savefig(png, dpi=170, facecolor=SURFACE)

    payload = {arm: {**r.as_dict(), "per_seed_mr": per_seed[arm],
                     "matthew_sem_ci": [np.mean(per_seed[arm]) - 1.96 * sem[arm],
                                        np.mean(per_seed[arm]) + 1.96 * sem[arm]],
                     "own_scenarios": {**own[arm][0].as_dict(), "n_scenarios": own[arm][1]}}
               for arm, r in results.items()}
    (dest / "results.json").write_text(json.dumps(
        {"arms": payload, "differences_vs_control": {a: d.as_dict() for a, d in diffs.items()},
         "shared_scenarios": shared,
         "per_repo_scenario_counts": {a: {s: len(v) for s, v in d.items()} for a, d in raw.items()},
         "repos": ARMS}, indent=2, default=float))
    write_run_meta(dest, {"script": "scratch/stats/plot_odcv_arms4.py",
                          "n_shared_scenarios": len(shared), "variant": VARIANT, "rollouts_per_cell": 1})

    print(f"{len(shared)} shared scenarios across all 12 repos "
          f"(per-repo: {sorted({len(v) for d in raw.values() for v in d.values()})})\n")
    for arm, r in results.items():
        m = np.mean(per_seed[arm])
        print(f"{arm:24s} {r.mean:5.1f}%  ours [{r.lo:5.1f}, {r.hi:5.1f}] df {r.df:5.1f}   "
              f"Matthew [{m-1.96*sem[arm]:5.1f}, {m+1.96*sem[arm]:5.1f}]   "
              f"seeds {[round(v,1) for v in per_seed[arm]]}")
    print("\non each arm's OWN 3-seed scenario set (unpaired across arms; Matthew's numbers "
          "come from sets like these):")
    for arm, (r, n) in own.items():
        print(f"  {arm:24s} {r.mean:5.1f}%  [{r.lo:5.1f}, {r.hi:5.1f}]  ({n} scenarios)")
    print(f"\npaired difference vs {ctrl} (shared scenarios, checkpoints unpaired):")
    for arm, d in diffs.items():
        sig = "excludes 0" if (d.lo > 0 or d.hi < 0) else "includes 0"
        print(f"  {arm:24s} {d.mean:+6.1f}pp [{d.lo:+6.1f}, {d.hi:+6.1f}]  df {d.df:4.1f}  {sig}")
    print(f"\n>>> {png.relative_to(REPO)}")


if __name__ == "__main__":
    fire.Fire(main)
