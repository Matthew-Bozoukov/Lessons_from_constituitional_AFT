# ABOUTME: What-if for the numina-vs-DA ODCV comparison: give every cell R rollouts with
# ABOUTME: rollout noise, keep each arm's mean where it was observed, recompute every error bar.
# Run: uv run python scratch/stats/simulate_rollouts.py [--R 5] [--flip 0.2] [--reps 500]

"""Counterfactual R rollouts per cell that preserve the observed arm means.

Model: a cell observed as a violation has true rate 1 - delta_1; a clean cell has delta_0,
with mu * delta_1 = (1 - mu) * delta_0 so the table of true rates has exactly the observed
mean mu, and mu * delta_1 + (1 - mu) * delta_0 = flip so a rollout disagrees with the observed
one with probability `flip` on average (0.2 = "one in five"). R Bernoulli draws per cell give
a rate table; the three interval estimators run on it exactly as on the 0/1 table (they only
see cell means). Repeated `reps` times; the reported intervals are the average bounds.

The constraint has a consequence worth reading: an arm with few violations (DA, mu = 0.17)
must give its violation cells a high flip rate (delta_1 = flip / (2 mu)) to offset the many
clean cells drifting up. That is arithmetic, not a modelling choice -- any mean-preserving
noise on a low-rate arm looks like this.

`--mode flip_one` keeps the earlier deterministic rule (exactly one of R disagrees in every
cell), which does NOT preserve the means (a 20% floor on every clean cell).
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

from scratch.stats.crossed_ci import (  # noqa: E402
    Interval, both_random, both_random_diff, crossed_terms, model_fixed, scenarios_fixed,
    scenarios_fixed_diff,
)
from scratch.stats.plot_odcv_ci_comparison import (  # noqa: E402
    ARMS, COLOR, GRID, INK, INK2, SEEDS, load_tables,
)
from src.utils import git_sha, write_run_meta  # noqa: E402


def true_rates(table: np.ndarray, flip: float) -> tuple[np.ndarray, float, float]:
    """Per-cell true rates preserving the table mean, with average flip probability `flip`."""
    mu = float(table.mean())
    d1, d0 = flip / (2 * mu), flip / (2 * (1 - mu))
    assert 0 < d1 < 1 and 0 < d0 < 1, f"flip={flip} infeasible for mu={mu:.3f}"
    return np.where(table > 0.5, 1 - d1, d0), d1, d0


def draw_rates(pi: np.ndarray, R: int, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Mean of R Bernoulli(pi) draws per cell, and the mean within-cell sample variance."""
    draws = rng.random((R, *pi.shape)) < pi[None]
    rates = draws.mean(axis=0)
    within = draws.astype(float).var(axis=0, ddof=1).mean()
    return rates, float(within)


def flip_one(table: np.ndarray, R: int) -> np.ndarray:
    return (R - 1) / R * table + 1 / R * (1 - table)


def _avg(ivs: list[Interval]) -> Interval:
    """Average the point estimate and bounds of intervals across replicates."""
    return Interval(float(np.mean([i.mean for i in ivs])), float(np.mean([i.se for i in ivs])),
                    ivs[0].df, ivs[0].mult, float(np.mean([i.lo for i in ivs])),
                    float(np.mean([i.hi for i in ivs])), ivs[0].method + " (mean over replicates)")


def _bar(ax, x, iv, color, hatch=None, width=0.36):
    ax.bar(x, 100 * iv.mean, width=width, color=color, alpha=0.85 if hatch is None else 0.35,
           hatch=hatch, edgecolor=color if hatch else "none", zorder=2)
    ax.errorbar(x, 100 * iv.mean, yerr=[[100 * (iv.mean - iv.lo)], [100 * (iv.hi - iv.mean)]],
                fmt="none", ecolor=INK, elinewidth=1.3, capsize=5, capthick=1.3, zorder=4)
    ax.text(x, 100 * iv.hi + 1.0, f"{100 * iv.mean:.0f}", ha="center", va="bottom", fontsize=8, color=INK2)


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


def main(R: int = 5, flip: float = 0.2, reps: int = 500, seed: int = 0,
         mode: str = "mean_preserving", out: str = "output/odcv_ci_comparison") -> None:
    tables, shared, meta = load_tables()
    arms = list(ARMS)
    J, n = len(shared), len(SEEDS)
    rng = np.random.default_rng(seed)

    # Simulated rate tables: one per replicate per arm (mean_preserving) or one (flip_one).
    sims: dict[str, list[np.ndarray]] = {}
    within: dict[str, float] = {}
    deltas: dict[str, tuple[float, float]] = {}
    for arm in arms:
        if mode == "flip_one":
            sims[arm] = [flip_one(tables[arm], R)]
            within[arm] = float(np.array([1.0] * (R - 1) + [0.0]).var(ddof=1))
        else:
            pi, d1, d0 = true_rates(tables[arm], flip)
            deltas[arm] = (d1, d0)
            rs, ws = zip(*(draw_rates(pi, R, rng) for _ in range(reps)))
            sims[arm], within[arm] = list(rs), float(np.mean(ws))
    sim_label = f"R={R}, {'one flips' if mode == 'flip_one' else f'flip≈{flip:.2f}, mean kept'}"

    res: dict[tuple[str, str], dict] = {}
    rows = []
    for arm in arms:
        for label, tabs in (("R=1 observed", [tables[arm]]), (sim_label, sims[arm])):
            br = _avg([both_random(t) for t in tabs])
            sf = _avg([scenarios_fixed(t) for t in tabs])
            mf = [_avg([model_fixed(t[i]) for t in tabs]) for i in range(n)]
            T = np.mean([crossed_terms(t)[1:4] for t in tabs], axis=0)
            res[(arm, label)] = dict(both_random=br, scenarios_fixed=sf, model_fixed=mf, T=tuple(map(float, T)))
            rows.append(f"| {arm} | {label} | {100 * br.mean:.1f}% | [{100 * br.lo:.1f}, {100 * br.hi:.1f}] "
                        f"| [{100 * sf.lo:.1f}, {100 * sf.hi:.1f}] | "
                        + ", ".join(f"[{100 * m.lo:.0f}, {100 * m.hi:.0f}]" for m in mf)
                        + f" | {T[0]:.5f} / {T[1]:.5f} / {T[2]:.5f} |")
    A, B = arms[1], arms[0]
    diffs = {"R=1 observed": (both_random_diff(tables[A], tables[B]), scenarios_fixed_diff(tables[A], tables[B])),
             sim_label: (_avg([both_random_diff(a, b) for a, b in zip(sims[A], sims[B])]),
                         _avg([scenarios_fixed_diff(a, b) for a, b in zip(sims[A], sims[B])]))}

    md = [f"# Counterfactual R={R} rollouts per cell — {sim_label}", "",
          f"Same 25×3 tables as the main comparison ({J} shared incentivized scenarios, seeds {list(SEEDS)}).", ""]
    if mode != "flip_one":
        md += ["Per-cell true rates (mean-preserving): violation cells 1−δ₁, clean cells δ₀ with μδ₁ = (1−μ)δ₀.", "",
               "| arm | μ | δ₁ (violation cell flips) | δ₀ (clean cell flips) |", "|---|---|---|---|"]
        md += [f"| {arm} | {tables[arm].mean():.3f} | {deltas[arm][0]:.3f} | {deltas[arm][1]:.3f} |" for arm in arms]
        md += ["", f"{reps} replicates of {R} Bernoulli draws per cell; intervals below are the average bounds.", ""]
    md += ["| arm | rollouts | MR | (a) both sampled ±1.96 | (b) scenarios fixed t₂ | (c) per model t₂₄ (seeds 0/42/69) | T_A / T_B / T_C |",
           "|---|---|---|---|---|---|---|", *rows, "",
           "## Difference (DA − numina)", "", "| rollouts | (a) both sampled | (b) Welch |", "|---|---|---|"]
    for label, (d1, d2) in diffs.items():
        md.append(f"| {label} | {100 * d1.mean:+.1f} [{100 * d1.lo:+.1f}, {100 * d1.hi:+.1f}] "
                  f"| {100 * d2.mean:+.1f} [{100 * d2.lo:+.1f}, {100 * d2.hi:+.1f}] |")
    md += ["", f"## Rollout-noise term, estimable once R={R}", "",
           "| arm | ŝ²_within (per rollout) | ŝ²_within/(nJR) | T_A+T_B−T_C | noise share |", "|---|---|---|---|---|"]
    for arm in arms:
        ta, tb, tc = res[(arm, sim_label)]["T"]
        noise, total = within[arm] / (n * J * R), ta + tb - tc
        md.append(f"| {arm} | {within[arm]:.3f} | {noise:.6f} | {total:.6f} | {100 * noise / total:.1f}% |")
    md.append("")

    ymax = 100 * max(max(v["both_random"].hi, v["scenarios_fixed"].hi, max(m.hi for m in v["model_fixed"]))
                     for v in res.values())
    ylim = (0, min(100, ymax * 1.2 + 4))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), facecolor="#fcfcfb")
    fig.subplots_adjust(wspace=0.3, top=0.8, bottom=0.2, left=0.06, right=0.99)
    labels = ("R=1 observed", sim_label)
    for p, (key, title) in enumerate((("both_random", r"(a) models AND scenarios sampled — SE$^2=\hat T_A+\hat T_B-\hat T_C$, $\pm$1.96"),
                                      ("scenarios_fixed", r"(b) scenarios fixed, models sampled — SE$^2=\hat T_A$, $t_2$"),
                                      ("model_fixed", r"(c) each model fixed, scenarios sampled — $s^2/J$, $t_{24}$"))):
        ax = axes[p]
        xs, xl = [], []
        for a_i, arm in enumerate(arms):
            if key == "model_fixed":
                for s_i, sd in enumerate(SEEDS):
                    base = a_i * (n + 0.8) + s_i
                    for k, lab in enumerate(labels):
                        _bar(ax, base + (k - 0.5) * 0.4, res[(arm, lab)][key][s_i], COLOR[arm], hatch=None if k else "///")
                    xs.append(base)
                    xl.append(f"seed {sd}")
            else:
                for k, lab in enumerate(labels):
                    _bar(ax, a_i + (k - 0.5) * 0.4, res[(arm, lab)][key], COLOR[arm], hatch=None if k else "///")
                xs.append(a_i)
                xl.append(arm)
        _style(ax, title, ylim)
        ax.set_xticks(xs, xl, fontsize=8.5)
        if key == "model_fixed":
            for a_i, arm in enumerate(arms):
                ax.text(a_i * (n + 0.8) + 1, ylim[1] * 0.97, arm, ha="center", va="top", fontsize=9, color=COLOR[arm])
    fig.suptitle(f"Counterfactual: {sim_label} (hatched = observed R=1, solid = simulated"
                 + ("" if mode == "flip_one" else f", mean of {reps} replicates") + ")", fontsize=11.5, color=INK, y=0.97)
    handles = [Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK2, hatch="///"), Rectangle((0, 0), 1, 1, facecolor=INK2)]
    fig.legend(handles + [Rectangle((0, 0), 1, 1, color=COLOR[a], alpha=0.85) for a in arms],
               list(labels) + arms, loc="lower center", ncol=4, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.01))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = REPO / out / f"sim_{mode}_R{R}_{stamp}"
    dest.mkdir(parents=True, exist_ok=True)
    png = dest / f"odcv_sim_{mode}_R{R}_{stamp}.png"
    fig.savefig(png, dpi=160)
    md += [f"figure: `{png.name}` · git `{git_sha()}` · regenerate: `uv run python scratch/stats/simulate_rollouts.py "
           f"--R {R} --flip {flip} --reps {reps} --mode {mode}`", ""]
    (dest / "results.md").write_text("\n".join(md))
    (dest / "results.json").write_text(json.dumps(
        {"R": R, "flip": flip, "reps": reps, "mode": mode, "within_cell_var": within, "deltas": deltas,
         "intervals": {f"{a} | {l}": {"both_random": v["both_random"].as_dict(),
                                      "scenarios_fixed": v["scenarios_fixed"].as_dict(),
                                      "model_fixed": [m.as_dict() for m in v["model_fixed"]], "T": v["T"]}
                       for (a, l), v in res.items()},
         "diff": {l: {"both_random": d1.as_dict(), "scenarios_fixed": d2.as_dict()} for l, (d1, d2) in diffs.items()},
         "meta": meta}, indent=2))
    write_run_meta(dest, {"script": "scratch/stats/simulate_rollouts.py", "R": R, "flip": flip, "reps": reps,
                          "mode": mode, "seed": seed, "arms": ARMS})
    print("\n".join(md))
    print(f">>> saved {dest.relative_to(REPO)}")


if __name__ == "__main__":
    fire.Fire(main)
