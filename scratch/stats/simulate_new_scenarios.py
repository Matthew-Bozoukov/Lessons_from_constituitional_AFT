# ABOUTME: What-if for the numina-vs-DA ODCV comparison: add K new incentivized scenarios drawn
# ABOUTME: from the same population as the existing ones and show what happens to the error bars.
# Run: uv run python scratch/stats/simulate_new_scenarios.py [--K 40] [--reps 500]

"""Counterfactual: K more scenarios that behave like the ones we have.

A new scenario is a resampled column of the existing table -- the three models' outcomes on
one existing scenario, drawn with replacement -- using the SAME column draws for both arms so
the pairing and the model x scenario pattern survive. The derived interval (both models and
scenarios sampled, SE^2 = T_A + T_B - T_C, +/-1.96) is recomputed on the (n x (J+K)) table and
averaged over `reps` replicates. Means are unchanged in expectation; only the bars move.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import fire
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scratch.stats.crossed_ci import Interval, both_random, both_random_diff, crossed_terms  # noqa: E402
from scratch.stats.plot_odcv_ci_comparison import ARMS, COLOR, GRID, INK, INK2, SEEDS, load_tables  # noqa: E402
from src.utils import git_sha, write_run_meta  # noqa: E402


def _avg(ivs: list[Interval]) -> Interval:
    return Interval(float(np.mean([i.mean for i in ivs])), float(np.mean([i.se for i in ivs])), ivs[0].df,
                    ivs[0].mult, float(np.mean([i.lo for i in ivs])), float(np.mean([i.hi for i in ivs])),
                    ivs[0].method + " (mean over replicates)")


def _panel(ax, ivs: dict[str, Interval], diff: Interval, title: str, ylim):
    for x, (arm, iv) in enumerate(ivs.items()):
        ax.bar(x, 100 * iv.mean, width=0.5, color=COLOR[arm], alpha=0.85, zorder=2)
        ax.errorbar(x, 100 * iv.mean, yerr=[[100 * (iv.mean - iv.lo)], [100 * (iv.hi - iv.mean)]], fmt="none",
                    ecolor=INK, elinewidth=1.5, capsize=7, capthick=1.5, zorder=4)
        ax.text(x, 100 * iv.hi + 1.2, f"{100 * iv.mean:.1f}%  [{100 * iv.lo:.1f}, {100 * iv.hi:.1f}]",
                ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(range(len(ivs)), list(ivs))
    ax.set_xlim(-0.6, len(ivs) - 0.4)
    ax.set_ylim(*ylim)
    ax.set_ylabel("misalignment rate (%)", fontsize=9.5, color=INK2)
    ax.set_title(title + f"\nDA − numina = {100 * diff.mean:+.1f} pp  [{100 * diff.lo:+.1f}, {100 * diff.hi:+.1f}]",
                 fontsize=10, loc="left", color=INK)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9.5)


def main(K: int = 40, reps: int = 500, seed: int = 0, out: str = "output/odcv_ci_comparison") -> None:
    tables, shared, meta = load_tables()
    arms = list(ARMS)
    A, B = arms[1], arms[0]  # difficult advice, numina
    n, J = tables[arms[0]].shape
    rng = np.random.default_rng(seed)

    before = {arm: both_random(t) for arm, t in tables.items()}
    before_diff = both_random_diff(tables[A], tables[B])
    after_reps: dict[str, list[Interval]] = {arm: [] for arm in arms}
    diff_reps: list[Interval] = []
    terms_reps: dict[str, list[tuple]] = {arm: [] for arm in arms}
    for _ in range(reps):
        idx = rng.integers(0, J, size=K)             # same new scenarios for both arms
        aug = {arm: np.concatenate([t, t[:, idx]], axis=1) for arm, t in tables.items()}
        for arm in arms:
            after_reps[arm].append(both_random(aug[arm]))
            terms_reps[arm].append(crossed_terms(aug[arm])[1:4])
        diff_reps.append(both_random_diff(aug[A], aug[B]))
    after = {arm: _avg(v) for arm, v in after_reps.items()}
    after_diff = _avg(diff_reps)

    ymax = 100 * max(max(iv.hi for iv in before.values()), max(iv.hi for iv in after.values()))
    ylim = (0, min(100, ymax * 1.2 + 6))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), facecolor="#fcfcfb", sharey=True)
    fig.subplots_adjust(wspace=0.12, top=0.78, bottom=0.14, left=0.08, right=0.99)
    _panel(axes[0], before, before_diff, f"before: J = {J} scenarios × {n} seeds", ylim)
    _panel(axes[1], after, after_diff, f"after: J = {J + K} scenarios × {n} seeds  (+{K} drawn from the same population)", ylim)
    axes[1].set_ylabel("")
    fig.suptitle(f"Adding {K} incentivized scenarios that behave like the existing ones — derived interval, "
                 r"SE$^2=\hat T_A+\hat T_B-\hat T_C$, $\pm1.96$" + f"  (after = mean of {reps} replicates)",
                 fontsize=10.5, color=INK, y=0.97)
    fig.legend([Rectangle((0, 0), 1, 1, color=COLOR[a], alpha=0.85) for a in arms], arms, loc="lower center",
               ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.0))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = REPO / out / f"sim_scenarios_K{K}_{stamp}"
    dest.mkdir(parents=True, exist_ok=True)
    png = dest / f"odcv_sim_scenarios_K{K}_{stamp}.png"
    fig.savefig(png, dpi=160)

    md = [f"# Adding K={K} scenarios drawn from the same population ({stamp})", "",
          f"Before: {J} shared incentivized scenarios × {n} seeds, first rollout per cell. After: {J + K}, the extra "
          f"{K} being resampled columns (same draws for both arms); {reps} replicates, average bounds.", "",
          "| arm | J | MR | 95% CI | half-width | T_A / T_B / T_C |", "|---|---|---|---|---|---|"]
    for arm in arms:
        ta, tb, tc = crossed_terms(tables[arm])[1:4]
        iv = before[arm]
        md.append(f"| {arm} | {J} | {100 * iv.mean:.1f}% | [{100 * iv.lo:.1f}, {100 * iv.hi:.1f}] | ±{100 * (iv.hi - iv.mean):.1f} pp | {ta:.5f} / {tb:.5f} / {tc:.5f} |")
        ta, tb, tc = np.mean(terms_reps[arm], axis=0)
        iv = after[arm]
        md.append(f"| {arm} | {J + K} | {100 * iv.mean:.1f}% | [{100 * iv.lo:.1f}, {100 * iv.hi:.1f}] | ±{100 * (iv.hi - iv.mean):.1f} pp | {ta:.5f} / {tb:.5f} / {tc:.5f} |")
    md += ["", "| difference DA − numina | Δ | 95% CI |", "|---|---|---|",
           f"| J = {J} | {100 * before_diff.mean:+.1f} pp | [{100 * before_diff.lo:+.1f}, {100 * before_diff.hi:+.1f}] |",
           f"| J = {J + K} | {100 * after_diff.mean:+.1f} pp | [{100 * after_diff.lo:+.1f}, {100 * after_diff.hi:+.1f}] |", "",
           f"Expected shrink of the scenario term: √({J}/{J + K}) = {np.sqrt(J / (J + K)):.2f}.", "",
           f"figure: `{png.name}` · git `{git_sha()}` · regenerate: `uv run python scratch/stats/simulate_new_scenarios.py --K {K} --reps {reps}`", ""]
    (dest / "results.md").write_text("\n".join(md))
    (dest / "results.json").write_text(json.dumps(
        {"K": K, "reps": reps, "J_before": J, "n": n,
         "before": {a: iv.as_dict() for a, iv in before.items()}, "before_diff": before_diff.as_dict(),
         "after": {a: iv.as_dict() for a, iv in after.items()}, "after_diff": after_diff.as_dict(), "meta": meta}, indent=2))
    write_run_meta(dest, {"script": "scratch/stats/simulate_new_scenarios.py", "K": K, "reps": reps, "seed": seed, "arms": ARMS})
    print("\n".join(md))
    print(f">>> saved {dest.relative_to(REPO)}")


if __name__ == "__main__":
    fire.Fire(main)
