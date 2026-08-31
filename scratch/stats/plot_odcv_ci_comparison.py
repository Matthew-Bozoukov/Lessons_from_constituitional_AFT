# ABOUTME: Numina control vs 5% difficult-advice on ODCV (incentivized, first rollout per cell,
# ABOUTME: three seeds each): six panels of error bars under different sampling assumptions.
# Run: uv run python scratch/stats/plot_odcv_ci_comparison.py [--out output/odcv_ci_comparison]

"""Six error-bar plots for the same two arms, differing only in what is treated as random.

Data: the six seed repos in LASR-Callum, incentivized variant only, ONE rollout per
(model, scenario) cell — the lowest-numbered pass that every judge scored — on the scenarios
all six models share. A cell is a violation when the median judge severity is >= 3.

Top row ("rollouts treated as deterministic": the cell value IS the model's behaviour):
  (a) models AND scenarios sampled      SE^2 = T_A + T_B - T_C, +/-1.96   <- THE DERIVED CALCULATION
  (b) scenarios fixed, models sampled   SE^2 = T_A                (Matthew's seed SEM)
  (c) each model fixed, scenarios sampled, six bars   SE^2 = s^2 / J

Bottom row ("rollouts are Bernoulli draws"): the OUTER bars are numerically the same — a
sample spread already contains whatever rollout luck is in the cells, so nothing is added —
and the INNER bars show the opposite extreme, the interval if ALL cell-to-cell spread were
rollout luck (every cell i.i.d.; Wilson). The gap between inner and outer is the least the
model/scenario structure can be contributing. See scratch/stats/crossed_ci.py.
"""

from __future__ import annotations

import json
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
from matplotlib.patches import Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scratch.stats.crossed_ci import (  # noqa: E402
    both_random, both_random_diff, model_fixed, naive_binomial, scenarios_fixed,
    scenarios_fixed_diff, wilson,
)
from src.eval.misalignment.odcv.odcv import VIOLATION_THRESHOLD, median_score  # noqa: E402
from src.utils import git_sha, write_run_meta  # noqa: E402

VARIANT = "incentivized"
SEEDS = (0, 42, 69)
ARMS = {
    "numina control": {
        0: "LASR-Callum/2026-08-19-odcv-numina-control-716-seed0",
        42: "LASR-Callum/2026-08-26-odcv-numina-control-716-seed42",
        69: "LASR-Callum/2026-08-26-odcv-numina-control-716-seed69"},
    "5% difficult advice": {
        0: "LASR-Callum/2026-08-24-odcv-synthdoc-716-seed0-rollout002",
        42: "LASR-Callum/2026-08-26-odcv-synthdoc-716-seed42",
        69: "LASR-Callum/2026-08-26-odcv-synthdoc-716-seed69"},
}
COLOR = {"numina control": "#2a78d6", "5% difficult advice": "#eb6834"}  # validated categorical slots 1, 2
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def first_rollout_scores(root: Path) -> dict[str, tuple[float, str]]:
    """{scenario: (median judge severity, pass)} for the first pass every judge scored."""
    judges = {f.name: json.loads(f.read_text()) for f in sorted(root.glob("results/scores_*.json"))}
    assert judges, f"no judge score files under {root}"
    out = {}
    for scen in sorted((root / "rollouts" / VARIANT).iterdir()):
        passes = sorted((p for p in scen.iterdir() if p.is_dir() and p.name.startswith("pass")),
                        key=lambda p: int(p.name[4:]))
        for p in passes:
            n = int(p.name[4:])
            keys = (f"{VARIANT}/{scen.name}/rollout_{n - 1:03d}", f"{VARIANT}/{scen.name}")
            scores = []
            for jd in judges.values():
                hit = next((jd[k] for k in keys if k in jd), None)
                if not hit or hit.get("score") in (None, "N/A"):
                    scores = None
                    break
                scores.append(float(hit["score"]))
            if scores:
                out[scen.name] = (median_score(scores), p.name)
                break
    return out


def load_tables() -> tuple[dict[str, np.ndarray], list[str], dict]:
    per_model = {}
    for arm, seeds in ARMS.items():
        for seed, repo in seeds.items():
            root = Path(snapshot_download(repo, repo_type="dataset"))
            per_model[(arm, seed)] = first_rollout_scores(root)
    shared = sorted(set.intersection(*(set(v) for v in per_model.values())))
    assert len(shared) >= 2, "too few shared scenarios"
    tables, passes = {}, {}
    for arm, seeds in ARMS.items():
        tables[arm] = np.array([[float(per_model[(arm, s)][sc][0] >= VIOLATION_THRESHOLD)
                                 for sc in shared] for s in seeds])
        passes[arm] = {s: sorted({per_model[(arm, s)][sc][1] for sc in shared}) for s in seeds}
    meta = {"shared_scenarios": shared, "cells_per_model": {f"{a}/seed{s}": len(v) for (a, s), v in per_model.items()},
            "passes_used": {a: {str(s): p for s, p in v.items()} for a, v in passes.items()}}
    return tables, shared, meta


def _bar(ax, x, iv, color, label=None, inner=None, width=0.5):
    ax.bar(x, 100 * iv.mean, width=width, color=color, alpha=0.85, label=label, zorder=2)
    ax.errorbar(x, 100 * iv.mean, yerr=[[100 * (iv.mean - iv.lo)], [100 * (iv.hi - iv.mean)]],
                fmt="none", ecolor=INK, elinewidth=1.4, capsize=6, capthick=1.4, zorder=4)
    if inner is not None:
        lo, hi = inner
        ax.errorbar(x, 100 * iv.mean, yerr=[[100 * (iv.mean - lo)], [100 * (hi - iv.mean)]],
                    fmt="none", ecolor=INK, elinewidth=5, alpha=0.35, capsize=0, zorder=3)
    ax.text(x, 100 * iv.hi + 1.2, f"{100 * iv.mean:.1f}%", ha="center", va="bottom", fontsize=9,
            color=INK2)


def _style(ax, title, ylim):
    ax.set_title(title, fontsize=9.5, loc="left", color=INK)
    ax.set_ylim(*ylim)
    ax.set_ylabel("misalignment rate (%)", fontsize=9, color=INK2)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


def main(out: str = "output/odcv_ci_comparison") -> None:
    load_dotenv()
    tables, shared, meta = load_tables()
    arms = list(ARMS)
    A, B = tables[arms[1]], tables[arms[0]]  # difficult advice, numina
    n, J = A.shape

    res = {arm: {
        "both_random": both_random(t).as_dict(),
        "scenarios_fixed": scenarios_fixed(t).as_dict(),
        "model_fixed": {str(s): model_fixed(t[i]).as_dict() for i, s in enumerate(SEEDS)},
        "naive_binomial": naive_binomial(t).as_dict(),
        "per_model_wilson": {str(s): wilson(float(t[i].sum()), J) for i, s in enumerate(SEEDS)},
        "per_model_rate": {str(s): float(t[i].mean()) for i, s in enumerate(SEEDS)},
    } for arm, t in tables.items()}
    diff = {"both_random": both_random_diff(A, B).as_dict(),
            "scenarios_fixed": scenarios_fixed_diff(A, B).as_dict()}

    ymax = 100 * max(max(r["both_random"]["hi"], r["scenarios_fixed"]["hi"],
                         max(m["hi"] for m in r["model_fixed"].values())) for r in res.values())
    ylim = (0, min(100, ymax * 1.25 + 4))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.2), facecolor="#fcfcfb")
    fig.subplots_adjust(hspace=0.62, wspace=0.32, top=0.86, bottom=0.11, left=0.06, right=0.99)
    for r, stochastic in enumerate((False, True)):
        # (a) both random
        ax = axes[r, 0]
        for x, arm in enumerate(arms):
            iv = both_random(tables[arm])
            inner = (res[arm]["naive_binomial"]["lo"], res[arm]["naive_binomial"]["hi"]) if stochastic else None
            _bar(ax, x, iv, COLOR[arm], label=arm, inner=inner)
        d = diff["both_random"]
        _style(ax, "(a) models AND scenarios sampled\n"
                   r"SE$^2=\hat T_A+\hat T_B-\hat T_C$, $\pm1.96$"
                   + ("   ← THE DERIVED CALCULATION" if not stochastic else "")
                   + f"\nDA − numina = {100 * d['mean']:+.1f} pp  [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]",
               ylim)
        ax.set_xticks(range(len(arms)), arms)
        # (b) scenarios fixed
        ax = axes[r, 1]
        for x, arm in enumerate(arms):
            iv = scenarios_fixed(tables[arm])
            inner = (res[arm]["naive_binomial"]["lo"], res[arm]["naive_binomial"]["hi"]) if stochastic else None
            _bar(ax, x, iv, COLOR[arm], inner=inner)
        d = diff["scenarios_fixed"]
        _style(ax, "(b) scenarios = fixed benchmark, models sampled\n"
                   r"SE$^2=\hat T_A$, $t_{n-1}$  (Matthew's seed SEM)"
                   + f"\nDA − numina = {100 * d['mean']:+.1f} pp  [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]  (Welch df {d['df']:.1f})",
               ylim)
        ax.set_xticks(range(len(arms)), arms)
        # (c) each model fixed
        ax = axes[r, 2]
        xs, labels = [], []
        for a_i, arm in enumerate(arms):
            for s_i, seed in enumerate(SEEDS):
                x = a_i * (len(SEEDS) + 0.6) + s_i
                iv = model_fixed(tables[arm][s_i])
                inner = res[arm]["per_model_wilson"][str(seed)] if stochastic else None
                _bar(ax, x, iv, COLOR[arm], inner=inner, width=0.7)
                xs.append(x)
                labels.append(f"seed {seed}")
        _style(ax, "(c) each model fixed, scenarios sampled\n"
                   r"per model: SE$^2=s^2/J$, $t_{J-1}$" + "\n ", ylim)
        ax.set_xticks(xs, labels, fontsize=8)
        for a_i, arm in enumerate(arms):
            ax.text(a_i * (len(SEEDS) + 0.6) + 1, ylim[1] * 0.97, arm, ha="center", va="top",
                    fontsize=9, color=COLOR[arm])

    fig.text(0.06, 0.945, "Rollouts treated as deterministic — the cell value is the model's behaviour",
             fontsize=11, color=INK, weight="bold")
    fig.text(0.06, 0.475, "Rollouts as Bernoulli draws — outer bars unchanged (spreads already contain rollout luck); "
                          "inner bars = if ALL spread were rollout luck (i.i.d. cells, Wilson)",
             fontsize=11, color=INK, weight="bold")
    fig.suptitle(f"ODCV-Bench, incentivized variant, first rollout per cell, {J} shared scenarios × 3 seeds per arm "
                 f"— 95% intervals under different sampling assumptions", fontsize=12, color=INK, y=0.985)
    handles = [Rectangle((0, 0), 1, 1, color=COLOR[a], alpha=0.85) for a in arms]
    fig.legend(handles, arms, loc="lower center", ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.005))
    fig.text(0.99, 0.005, "thin whisker: 95% interval (±1.96 in (a); t with exact df in (b), (c)) · "
                          "thick inner bar (bottom row): Wilson, all cells i.i.d.",
             ha="right", fontsize=8.5, color=INK2)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = REPO / out / stamp
    dest.mkdir(parents=True, exist_ok=True)
    png = dest / f"odcv_ci_comparison_{stamp}.png"
    fig.savefig(png, dpi=160)

    md = [f"# ODCV numina control vs 5% difficult advice — error bars under different assumptions ({stamp})", "",
          f"Incentivized only; first judged pass per cell; {J} shared scenarios; seeds {list(SEEDS)} per arm; "
          f"violation = median judge severity ≥ {VIOLATION_THRESHOLD}. Intervals are 95%, t-based.", "",
          "| arm | method | MR | 95% CI | SE | df | parts |", "|---|---|---|---|---|---|---|"]
    for arm, r in res.items():
        for m in ("both_random", "scenarios_fixed", "naive_binomial"):
            iv = r[m]
            parts = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in iv["parts"].items()}
            md.append(f"| {arm} | {iv['method']} | {100 * iv['mean']:.1f}% | [{100 * iv['lo']:.1f}, {100 * iv['hi']:.1f}] "
                      f"| {100 * iv['se']:.2f}pp | {iv['df']:.1f} | `{parts}` |")
        for s, iv in r["model_fixed"].items():
            w = r["per_model_wilson"][s]
            md.append(f"| {arm} seed {s} | {iv['method']} | {100 * iv['mean']:.1f}% | [{100 * iv['lo']:.1f}, {100 * iv['hi']:.1f}] "
                      f"| {100 * iv['se']:.2f}pp | {iv['df']:.0f} | Wilson [{100 * w[0]:.1f}, {100 * w[1]:.1f}] |")
    md += ["", "## Difference (5% difficult advice − numina control)", "",
           "| method | Δ MR | 95% CI | df |", "|---|---|---|---|"]
    for m, iv in diff.items():
        md.append(f"| {iv['method']} | {100 * iv['mean']:+.1f} pp | [{100 * iv['lo']:+.1f}, {100 * iv['hi']:+.1f}] | {iv['df']:.1f} |")
    md += ["", "## Cells", "", f"shared scenarios ({J}): " + ", ".join(shared), "",
           "passes used per model: " + json.dumps(meta["passes_used"]), "",
           f"figure: `{png.name}` · git `{git_sha()}` · regenerate: `uv run python scratch/stats/plot_odcv_ci_comparison.py`", ""]
    (dest / "results.md").write_text("\n".join(md))
    (dest / "results.json").write_text(json.dumps({"results": res, "diff": diff, "meta": meta,
                                                   "tables": {a: t.tolist() for a, t in tables.items()}}, indent=2))
    write_run_meta(dest, {"script": "scratch/stats/plot_odcv_ci_comparison.py", "arms": ARMS,
                          "variant": VARIANT, "shared_scenarios": J})
    print("\n".join(md))
    print(f">>> saved {dest.relative_to(REPO)}")


if __name__ == "__main__":
    fire.Fire(main)
