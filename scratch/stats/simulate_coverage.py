# ABOUTME: End-to-end sanity check on src/eval/stats: simulate the ODCV process from a KNOWN
# ABOUTME: truth and measure whether our "95%" interval actually covers it 95% of the time.
# Run: uv run python scratch/stats/simulate_coverage.py [--reps 2000]

"""Does our 95% interval actually cover 95% of the time?

The whole pipeline is under test -- `collapse`, the Design, the three spreads, Satterthwaite,
the t multiplier -- not just the algebra. Each replicate runs the real code path.

Generative process, mirroring ODCV on a latent logit scale so rates stay in [0, 1] without
clipping (clipping would break the known truth):

    scenario   D_j ~ (any `draw` kind)             40 per experiment
    checkpoint A_i ~ (any `draw` kind)             n per experiment
    variant    gamma_k = +/- 0.5                   BOTH always run, weights 1/2 each
    interaction C_ij ~ N(0, sC^2)
    rate       pi = logistic(D_j + A_i + gamma_k + C_ij),  V ~ Bernoulli(pi), R rollouts

The INGREDIENTS are deliberately not all Gaussian. Real ODCV scenarios are nothing like a
normal: most stories a decent arm never violates on at all, a handful it always violates on,
and the rest sit in between. The `low_rate` suite therefore uses zero-inflation (a point mass
at pi = 0), all-or-nothing scenarios, Student-t tails, Beta rates piled against 0, and a
"one bad seed in three" checkpoint effect. All of them sit at mu ~ 0.02-0.10, the regime where
the interval shape actually matters, and none of them satisfies any normality the estimator
might have been quietly leaning on.

Zero-inflation also lets this test the BOUNDARY path, which the Gaussian suite could not: a
replicate can return mu_hat exactly 0 while the true mu is 0.04, which is precisely the case
where every spread collapses and the interval falls back to a Wilson score bound.

The estimand mu = E[pi] has no closed form through the logistic, so it is taken from a huge
Monte Carlo of the same process; its own error is reported so you can see it is negligible
against the interval widths being tested.

Both the checkpoints and the scenarios are redrawn every replicate, because both are declared
`sampled` -- that IS the estimand. Reported per regime:

    coverage     how often mu falls inside [lo, hi]      target 95%
    calibration  mean SE^2 / Var(mu_hat) across reps     target 1.00  (isolates the SE from
                                                          the multiplier)
    naive        the same for "all n*J*2*R rollouts i.i.d." Wilson, as a control
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
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.naming import figure_path
from src.eval.stats import Design, NotEstimable, interval, wilson  # noqa: E402
from src.utils import git_sha, write_run_meta  # noqa: E402

DESIGN = Design(item="scenario", enumerated={"variant": "equal"}, subsamples=("pass",))
VARIANTS = ("mandated", "incentivized")
logistic = lambda x: 1.0 / (1.0 + np.exp(-x))

J = 40
DEAD, ALIVE = -60.0, 60.0   # logits that ARE 0 and 1 to float precision, without inf arithmetic


def draw(rng, spec, size):
    """One ingredient on the logit scale. `spec` is (kind, *params)."""
    kind, *a = spec
    if kind == "normal":                       # the well-behaved baseline
        m, s = a
        return rng.normal(m, s, size)
    if kind == "mixture":                      # bimodal: some stories always, some never
        w, m1, s1, m2, s2 = a
        return np.where(rng.random(size) < w, rng.normal(m1, s1, size), rng.normal(m2, s2, size))
    if kind == "zero_inflated":                # a POINT MASS at pi = 0, plus a live component
        p0, m, s = a
        return np.where(rng.random(size) < p0, DEAD, rng.normal(m, s, size))
    if kind == "bernoulli":                    # all-or-nothing: pi is exactly 0 or exactly 1
        p1, = a
        return np.where(rng.random(size) < p1, ALIVE, DEAD)
    if kind == "t":                            # heavy tails; df=2 has INFINITE variance on
        m, s, df = a                           # the logit scale (pi is still bounded, so the
        return m + s * rng.standard_t(df, size)    # estimand is fine -- the tail is the test)
    if kind == "beta":                         # rates drawn directly, piled against 0
        aa, bb = a
        q = np.clip(rng.beta(aa, bb, size), 1e-12, 1 - 1e-12)
        return np.log(q / (1 - q))
    if kind == "bad_seed":                     # most seeds fine, occasionally one much worse
        pr, shift, s = a
        return np.where(rng.random(size) < pr, shift, 0.0) + rng.normal(0, s, size)
    raise ValueError(f"unknown ingredient kind {kind!r}")


# Each regime: scenario ingredient D, checkpoint ingredient A, interaction sd, n checkpoints, R.
BASE = {
    "base  (mu~0.4, bimodal scenarios)":  dict(D=("mixture", .6, -1.5, .5, 1.0, .5), A=("normal", 0, .35), sC=.30, n=3, R=5),
    "checkpoint-dominated (big sigma_A)": dict(D=("mixture", .6, -1.5, .2, 1.0, .2), A=("normal", 0, 1.20), sC=.15, n=3, R=5),
    "near-zero (mu~0.04)":                dict(D=("mixture", .85, -4.5, .5, -1.5, .5), A=("normal", 0, .35), sC=.30, n=3, R=5),
    # n=1 is `checkpoints="fixed"`: a claim about ONE model over sampled scenarios. So the
    # checkpoint is drawn ONCE and held for all replicates, and the truth is conditional on
    # it -- exactly the question the interval answers. (Redrawing it each replicate and
    # scoring against the pipeline mean is a different, unfair target; that gave 79%.)
    "single fixed checkpoint (n=1)":      dict(D=("mixture", .6, -1.5, .5, 1.0, .5), A=("normal", 0, .35), sC=.30, n=1, R=5, fixed_ckpt=True),
    "single fixed checkpoint, near-zero": dict(D=("mixture", .85, -4.5, .5, -1.5, .5), A=("normal", 0, .35), sC=.30, n=1, R=5, fixed_ckpt=True),
    "base but R=1":                       dict(D=("mixture", .6, -1.5, .5, 1.0, .5), A=("normal", 0, .35), sC=.30, n=3, R=1),
}

# All low-rate (mu ~ 0.02-0.10) and all deliberately non-Gaussian -- this is where the shape of
# the interval decides whether "95%" is true, and real ODCV scenarios look like these, not like
# a normal.
LOW_RATE = {
    "zero-inflated: 85% of stories never violate": dict(D=("zero_inflated", .85, -1.0, .8), A=("normal", 0, .35), sC=.30, n=3, R=5),
    "all-or-nothing stories (pi is 0 or 1)":        dict(D=("bernoulli", .06), A=("normal", 0, .35), sC=.30, n=3, R=5),
    "heavy tails (Student t, df=2, infinite var)":  dict(D=("t", -4.0, 1.2, 2), A=("normal", 0, .35), sC=.30, n=3, R=5),
    "Beta(0.3, 12) rates: mass at 0, long tail":    dict(D=("beta", .3, 12.), A=("normal", 0, .35), sC=.30, n=3, R=5),
    "one bad seed in three":                        dict(D=("normal", -5.0, .8), A=("bad_seed", .30, 3.0, .15), sC=.30, n=3, R=5),
    "zero-inflated, R=1":                           dict(D=("zero_inflated", .85, -1.0, .8), A=("normal", 0, .35), sC=.30, n=3, R=1),
    "zero-inflated, one fixed checkpoint":          dict(D=("zero_inflated", .85, -1.0, .8), A=("normal", 0, .35), sC=.30, n=1, R=5, fixed_ckpt=True),
}
SUITES = {"base": BASE, "low_rate": LOW_RATE}


def truth(rng, p, draws=20_000_000, held_a=None) -> tuple[float, float]:
    """mu and its Monte Carlo error. With `held_a`, mu is CONDITIONAL on that checkpoint."""
    tot, tot2, done = 0.0, 0.0, 0
    while done < draws:
        k = min(2_000_000, draws - done)
        D = draw(rng, p["D"], k)
        A = np.full(k, held_a) if held_a is not None else draw(rng, p["A"], k)
        C = rng.normal(0, p["sC"], k)
        g = np.where(rng.random(k) < .5, .5, -.5)      # the 1/2-1/2 variant mixture
        pi = logistic(D + A + g + C)
        tot += pi.sum(); tot2 += (pi ** 2).sum(); done += k
    mu = tot / draws
    return float(mu), float(np.sqrt(max(tot2 / draws - mu ** 2, 0) / draws))


def experiment(rng, p, held_a=None) -> list[dict]:
    """One run: fresh scenarios, and fresh checkpoints unless `held_a` fixes the model."""
    n, R = p["n"], p["R"]
    D = draw(rng, p["D"], J)
    A = np.full(n, held_a) if held_a is not None else draw(rng, p["A"], n)
    rows = []
    for i in range(n):
        for j in range(J):
            for k, g in zip(VARIANTS, (.5, -.5)):
                pi = logistic(D[j] + A[i] + g + rng.normal(0, p["sC"]))
                for r in range(R):
                    rows.append({"checkpoint": f"m{i}", "scenario": f"s{j}", "variant": k,
                                 "pass": r, "value": float(rng.random() < pi)})
    return rows


def _figure(payload: dict, reps: int, png: Path) -> None:
    """Draw the per-regime histograms with both interval shapes overlaid.

    Kept separate from `main` so `replot` can redraw a finished run from its saved
    results.json -- a labelling fix must never cost another hour of simulation.
    """
    # The distribution of mu_hat itself. Each interval shape is a CLAIM about this histogram:
    # the symmetric interval claims it is normal; the logit interval claims logit(mu_hat) is
    # normal, i.e. that mu_hat is logit-normal. Drawing both against the truth shows which
    # claim holds, and so explains the coverage numbers rather than just reporting them.
    INK, INK2, GRID, BLUE, RED = "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6", "#c2410c"
    ncol = 4 if len(payload) > 6 else 3
    nrow = -(-len(payload) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 4.2 * nrow), facecolor="#fcfcfb",
                             squeeze=False)
    for ax, (label, d) in zip(axes.ravel(), payload.items()):
        m = np.array(d["means"]); mu = d["mu"]
        ax.hist(100 * m, bins=45, color=BLUE, alpha=.85, density=True, zorder=2)
        xs = np.linspace(m.min(), m.max(), 400)
        se, pbar = np.sqrt(d["mean_se2"]), float(m.mean())
        gauss = lambda x, c, sd: np.exp(-((x - c) ** 2) / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi))
        ax.plot(100 * xs, gauss(xs, pbar, se) / 100, color=INK2, lw=1.3, ls="--", zorder=4,
                label=f"normal — symmetric claim ({d['coverage_pct']:.0f}%)")
        # Logit-normal: logit(mu_hat) ~ N(logit(pbar), s) with s = SE/(p(1-p)), the delta-method
        # scale the interval itself uses. Change of variables gives the 1/(x(1-x)) Jacobian.
        if 0 < pbar < 1:
            s_l = se / (pbar * (1 - pbar))
            xin = xs[(xs > 0) & (xs < 1)]
            dens = gauss(np.log(xin / (1 - xin)), np.log(pbar / (1 - pbar)), s_l) / (xin * (1 - xin))
            ax.plot(100 * xin, dens / 100, color=RED, lw=2.0, zorder=5,
                    label=f"logit-normal — logit claim ({d['coverage_logit_pct']:.0f}%)")
        ax.axvline(100 * mu, color="#eb6834", lw=2, zorder=6, label=f"true mu = {100*mu:.1f}%")
        bd = f" · boundary {d['boundary_pct']:.0f}%" if d.get("boundary_pct") else ""
        ax.set_title(f"{label}\nsym {d['coverage_pct']:.1f}%  ·  logit {d['coverage_logit_pct']:.1f}%\n"
                     f"skew {d['skew']:+.2f} · SE²/Var {d['calibration_ratio']:.2f}{bd}",
                     fontsize=8.5, loc="left", color=INK, linespacing=1.45)
        ax.set_xlabel(f"sample mean of the {len(m)} experiments (%)", fontsize=8.5, color=INK2)
        ax.set_yticks([])
        ax.legend(fontsize=7.5, frameon=False, loc="upper right")
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8.5)
    for ax in axes.ravel()[len(payload):]:
        ax.set_visible(False)
    fig.suptitle(f"Sampling distribution of mu_hat over {reps} simulated experiments — each curve is "
                 "the shape an interval assumes; the histogram is the truth",
                 fontsize=11.5, color=INK, y=0.98)
    fig.subplots_adjust(top=0.80, bottom=0.08, left=0.04, right=0.99, hspace=0.72, wspace=0.16)
    fig.savefig(png, dpi=150)


def main(reps: int = 2000, truth_draws: int = 20_000_000, seed: int = 0, suite: str = "base",
         out: str = "output/stats_coverage") -> None:
    assert suite in SUITES, f"suite must be one of {sorted(SUITES)}"
    rows_md = []
    payload = {}
    for label, p in SUITES[suite].items():
        rng = np.random.default_rng(seed)
        held = float(draw(rng, p["A"], 1)[0]) if p.get("fixed_ckpt") else None
        mu, mc_se = truth(rng, p, truth_draws, held_a=held)
        hit = naive_hit = logit_hit = 0
        means, se2s, widths, lwidths, dfs, skipped = [], [], [], [], [], 0
        shapes: dict[str, int] = {}
        for _ in range(reps):
            obs = experiment(rng, p, held_a=held)
            try:
                # Both are the SHIPPED code path; `bounds` is the only difference between them.
                r = interval(obs, DESIGN)
                rl = interval(obs, DESIGN, bounds=(0.0, 1.0))
            except NotEstimable:
                skipped += 1
                continue
            hit += r.lo <= mu <= r.hi
            logit_hit += rl.lo <= mu <= rl.hi
            shapes[rl.shape] = shapes.get(rl.shape, 0) + 1
            lwidths.append(rl.hi - rl.lo)
            means.append(r.mean); se2s.append(r.se ** 2)
            widths.append(r.hi - r.lo); dfs.append(r.df)
            k = sum(o["value"] for o in obs)
            lo, hi = wilson(k, len(obs))            # control: every rollout i.i.d.
            naive_hit += lo <= mu <= hi
        n_ok = len(means)
        emp_var = float(np.var(means, ddof=1))
        cov, ncov = 100 * hit / n_ok, 100 * naive_hit / n_ok
        lcov = 100 * logit_hit / n_ok
        # Coverage is itself an estimate from n_ok Bernoulli trials, so report its own error --
        # otherwise a 0.5pp gap between two shapes cannot be told from simulation noise.
        cov_mc = 100 * np.sqrt(lcov / 100 * (1 - lcov / 100) / n_ok)
        bpct = 100 * shapes.get("wilson-at-boundary", 0) / n_ok
        rows_md.append(
            f"| {label} | {mu:.4f} | {cov:.1f}% | **{lcov:.1f}%** | ±{cov_mc:.2f} | {np.mean(se2s)/emp_var:.3f} | "
            f"{100*np.mean(widths):.1f} / {100*np.mean(lwidths):.1f}pp | {np.median(dfs):.0f} | "
            f"{bpct:.1f}% | {ncov:.1f}% |")
        payload[label] = {"mu": mu, "mu_mc_se": mc_se, "coverage_pct": cov,
                          "coverage_logit_pct": lcov, "held_checkpoint": held,
                          "coverage_mc_se_pp": float(cov_mc), "shapes": shapes,
                          "boundary_pct": float(bpct), "suite": suite,
                          "mean_width_logit_pp": float(100 * np.mean(lwidths)),
                          "calibration_ratio": float(np.mean(se2s) / emp_var),
                          "mean_width_pp": float(100 * np.mean(widths)),
                          "median_df": float(np.median(dfs)), "naive_wilson_coverage_pct": ncov,
                          "empirical_var_of_mean": emp_var, "mean_se2": float(np.mean(se2s)),
                          "reps": n_ok, "skipped": skipped, "means": [float(m) for m in means],
                          "skew": float(((np.array(means)-np.mean(means))**3).mean()
                                        / np.std(means)**3), **{k: v for k, v in p.items()}}
        print(rows_md[-1], flush=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = REPO / out / stamp
    dest.mkdir(parents=True, exist_ok=True)

    png = figure_path(dest, f"mu hat distributions {stamp}")
    _figure(payload, reps, png)
    md = [f"# Does our 95% interval cover 95% of the time? ({stamp})", "",
          f"{reps} replicates per regime; truth from a {truth_draws:,}-draw Monte Carlo of the "
          f"same process. 3 checkpoints x {J} scenarios x 2 variants x R rollouts, checkpoints "
          "AND scenarios redrawn every replicate.", "",
          "| regime | true mu | coverage sym | **coverage logit** | MC err | SE²/Var(mu_hat) | width sym/logit | median df | boundary | naive Wilson |",
          "|---|---|---|---|---|---|---|---|---|---|", *rows_md, "",
          "`coverage` should be 95%. `sym` is the symmetric interval `mu_hat +/- t*SE`; `logit` "
          "does the same step on the log-odds scale and maps back — same estimand, same SE, "
          "boundary-aware shape. `SE²/Var(mu_hat)` should be 1.00; it isolates the standard error "
          "from the multiplier. `naive Wilson` treats every rollout as independent and is the "
          "control for what we would get wrong.", "",
          f"git `{git_sha()}` · regenerate: `uv run python scratch/stats/simulate_coverage.py --reps {reps}`", ""]
    (dest / "results.md").write_text("\n".join(md))
    (dest / "results.json").write_text(json.dumps(payload, indent=2, default=float))
    write_run_meta(dest, {"script": "scratch/stats/simulate_coverage.py", "reps": reps,
                          "truth_draws": truth_draws, "J": J})
    print("\n".join(md))
    print(f">>> saved {dest.relative_to(REPO)}  (figure: {png.name})")


def replot(run_dir: str) -> None:
    """Redraw the figure from a finished run's results.json -- no simulation re-run.

    Args:
        run_dir: An `output/stats_coverage/<stamp>` directory.
    """
    d = Path(run_dir)
    payload = json.loads((d / "results.json").read_text())
    payload = payload.get("regimes", payload)
    payload = {k: v for k, v in payload.items() if isinstance(v, dict) and "means" in v}
    reps = max(len(v["means"]) for v in payload.values())
    png = figure_path(d, f"mu hat distributions {d.name}")
    _figure(payload, reps, png)
    print(f">>> {png}")


if __name__ == "__main__":
    fire.Fire({"run": main, "replot": replot})
