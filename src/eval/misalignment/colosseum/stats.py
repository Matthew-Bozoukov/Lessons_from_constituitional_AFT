# ABOUTME: Colosseum statistics on src/eval/stats.py: per-arm episode summaries (Wilson or t over
# ABOUTME: episodes), the arm difference paired on seed, and the transfer ratio's seed bootstrap.

"""Statistics for the Colosseum experiments.

Colosseum is the one eval here whose unit really is an independent draw: a seed is one
episode of the environment with its own ticket (or patient) set, each arm runs the same
seeds, and no two episodes share anything. So the per-arm interval is over EPISODES — Wilson
for a yes/no measure (`episode_summary` is told which those are), a plain t for a continuous
one — and the arm contrast is paired on seed, because the seed's own generosity (some sets
simply leave more to capture) is the largest source of variance and the pairing removes it.

`transfer_ratio` is the quantity Experiment 2 exists to produce: the multi-agent effect
divided by the single-agent effect. It is a RATIO OF TWO NOISY ESTIMATES and so has no
usable closed-form interval — when the denominator's interval covers zero the ratio's does
not converge, and a naive standard error would report a tight interval around a number that
means nothing. It is bootstrapped over seeds instead, both effects recomputed on each
resample so their shared seed draw stays shared, and it refuses to report at all unless the
single-agent effect is separated from zero.
"""

from __future__ import annotations

from typing import Collection

import numpy as np

from src.eval.stats import t_cdf, t_mean, wilson

__all__ = ["arm_difference", "episode_summary", "transfer_ratio"]


def episode_summary(per_seed: dict[str, dict[str, dict]], binary: Collection[str] = ()) -> dict:
    """One arm's measures as a mean over episodes with an interval over episodes.

    Args:
        per_seed: `{measure: {cell: {seed: value}}}`, the shape `per_seed.json` holds.
        binary: Measures whose per-episode value is 0/1 (a refusal happened, a report was
            made); these get a Wilson interval. Every other measure gets a t over episodes.
            A measure named here whose values are not all 0/1 is an error, not a guess.

    Returns:
        `{measure: {cell: {"mean", "ci95", "n_episodes", "ci_method"}}}`.
    """
    out: dict[str, dict[str, dict]] = {}
    for measure, cells in per_seed.items():
        out[measure] = {}
        for cell, seeds in cells.items():
            values = [float(v) for v in seeds.values() if v is not None]
            if not values:
                continue
            if measure in binary:
                assert all(v in (0.0, 1.0) for v in values), (
                    f"{measure}/{cell} is declared binary but holds {sorted(set(values))[:5]}")
                k, n = int(sum(values)), len(values)
                lo, hi = wilson(k, n)
                out[measure][cell] = {"mean": round(k / n, 4), "ci95": [round(lo, 4), round(hi, 4)],
                                      "n_episodes": n, "ci_method": f"Wilson over {n} episodes"}
            else:
                r = t_mean(values)
                out[measure][cell] = {
                    "mean": round(r["mean"], 4),
                    "ci95": None if r["lo"] is None else [round(r["lo"], 4), round(r["hi"], 4)],
                    "n_episodes": r["n"],
                    "ci_method": r["method"].replace("passes", "episodes"),
                }
    return out


def arm_difference(
    treatment: dict[int, float], control: dict[int, float], *, label: str
) -> dict:
    """Treatment minus control on the seeds both arms ran, paired on seed.

    Args:
        treatment: {seed: value} for the arm under test.
        control: The same for the control arm.
        label: What the value is (e.g. "coalition_advantage"), carried into the result so
            a summary holding several contrasts stays readable.

    Returns:
        The difference, its 95% t-interval over the per-seed differences, a two-sided
        p-value, and the seeds it rests on.
    """
    shared = sorted(set(treatment) & set(control))
    assert len(shared) >= 2, (
        f"{label}: arms share {len(shared)} seeds; a paired interval needs at least 2. "
        "Seeds are meant to be identical across blocks — check that both arms ran the "
        "same --seeds list and that no episode was dropped."
    )
    r = t_mean([treatment[s] - control[s] for s in shared])
    n = len(shared)
    sd = float(np.std(r["values"], ddof=1))
    se = sd / np.sqrt(n)
    if se == 0:
        p = 1.0 if r["mean"] == 0 else 0.0
    else:
        p = 2.0 * (1.0 - t_cdf(abs(r["mean"]) / se, n - 1))
    return {
        "measure": label,
        "treatment_mean": round(float(np.mean([treatment[s] for s in shared])), 4),
        "control_mean": round(float(np.mean([control[s] for s in shared])), 4),
        "diff": round(r["mean"], 4),
        "diff_ci95": [round(r["lo"], 4), round(r["hi"], 4)],
        "p_two_sided": round(p, 4),
        "n_seeds": n,
        "method": f"paired on seed: {r['method'].replace('passes', 'seeds')}",
    }


def transfer_ratio(
    multi_treatment: dict[int, float],
    multi_control: dict[int, float],
    single_treatment: dict[int, float],
    single_control: dict[int, float],
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Multi-agent effect / single-agent effect, bootstrapped over a shared seed draw.

    A ratio near 1 means the trained disposition is worth as much against a colluding
    peer as it is alone; well below 1 means it leaks exactly where the pressure comes
    from another agent, which is the multi-agent-risk hypothesis this experiment exists
    to test.

    The numerator and denominator are both differences over the SAME seeds, so each
    bootstrap resample draws one set of seeds and recomputes both — resampling them
    independently would break the pairing that makes the ratio a statement about the same
    task instances.

    Args:
        multi_treatment / multi_control: {seed: value} from Experiment 1.
        single_treatment / single_control: {seed: value} from Experiment 2.
        n_boot, seed, alpha: Resamples, RNG seed, two-sided level.

    Returns:
        The ratio and its percentile interval, plus `interpretable` and, when False,
        `refused_because`. A caller must not report the ratio when `interpretable` is
        False: the interval is arithmetically real but scientifically empty.
    """
    seeds = sorted(
        set(multi_treatment)
        & set(multi_control)
        & set(single_treatment)
        & set(single_control)
    )
    assert len(seeds) >= 2, (
        f"transfer ratio rests on {len(seeds)} seeds shared by all four cells; needs >= 2"
    )
    idx = np.arange(len(seeds))
    mt, mc, st, sc = (
        np.array([d[s] for s in seeds], float)
        for d in (multi_treatment, multi_control, single_treatment, single_control)
    )

    def effects(take):
        """Both effects on one seed draw. Control minus treatment, so a REDUCTION in
        collusion is a positive effect and a ratio of positives is the readable case."""
        return (
            float(np.mean(mc[take] - mt[take])),
            float(np.mean(sc[take] - st[take])),
        )

    multi_effect, single_effect = effects(idx)

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    single_draws = np.empty(n_boot)
    for b in range(n_boot):
        take = rng.integers(0, len(seeds), len(seeds))
        m, s = effects(take)
        single_draws[b] = s
        # A resample whose denominator lands on ~0 sends the ratio to +-inf. Those draws
        # are real — they are exactly what "the denominator is not separated from zero"
        # looks like — so they are kept in the quantiles rather than dropped, and the
        # interpretability gate below is what stops a meaningless interval being read.
        draws[b] = m / s if s != 0 else np.inf * np.sign(m) if m != 0 else np.nan

    finite = draws[np.isfinite(draws)]
    single_lo, single_hi = (
        float(np.quantile(single_draws, alpha / 2)),
        float(np.quantile(single_draws, 1 - alpha / 2)),
    )
    interpretable = single_lo > 0 or single_hi < 0
    out = {
        "multi_agent_effect": round(multi_effect, 4),
        "single_agent_effect": round(single_effect, 4),
        "single_agent_effect_ci95": [round(single_lo, 4), round(single_hi, 4)],
        "ratio": round(multi_effect / single_effect, 3) if single_effect else None,
        "ratio_ci95": (
            [
                round(float(np.quantile(finite, alpha / 2)), 3),
                round(float(np.quantile(finite, 1 - alpha / 2)), 3),
            ]
            if finite.size == draws.size
            else None
        ),
        "n_seeds": len(seeds),
        "n_boot": n_boot,
        "non_finite_resamples": int(draws.size - finite.size),
        "interpretable": bool(interpretable),
        "method": (
            f"percentile bootstrap over {len(seeds)} seeds, {n_boot} resamples, "
            "both effects recomputed on each shared seed draw"
        ),
    }
    if not interpretable:
        out["refused_because"] = (
            "the single-agent effect's 95% interval covers zero "
            f"([{single_lo:.4f}, {single_hi:.4f}]), so the ratio's denominator is not "
            "separated from zero and the ratio is not interpretable. Read the two "
            "effects separately: a transfer ratio only means something once Experiment 2 "
            "has shown there is a single-agent effect to transfer."
        )
    return out
