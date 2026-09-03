# ABOUTME: Colosseum contrast statistics on src/eval/stats.py: the arm difference paired on
# ABOUTME: seed, and the multi-agent/single-agent transfer ratio with a joint seed bootstrap.

"""Statistics for the three Colosseum experiments.

Everything here is paired on SEED. The same seeds are run in every block, so a seed is a
task instance both arms faced, and the seed-to-seed spread — which is large in this
environment, because the seed picks the ticket set and therefore how much surplus is
available to capture at all — is variance the pairing removes rather than variance the
interval has to carry.

Two estimators:

`arm_difference` is the ordinary treatment-minus-control interval, delegated to
`src.eval.stats.difference`, which knows how to pair on the item axis and gets the
degrees of freedom right.

`transfer_ratio` is the quantity Experiment 2 exists to produce: the multi-agent effect
divided by the single-agent effect. It is a RATIO OF TWO NOISY ESTIMATES and so has no
usable closed-form interval — when the denominator's interval covers zero the ratio's
does not converge, and a naive standard error would report a tight interval around a
number that means nothing. It is bootstrapped over seeds instead, both effects
recomputed on each resample so their shared seed draw stays shared, and it refuses to
report at all unless the single-agent effect is separated from zero. That refusal is the
point: "the disposition transfers" and "neither experiment moved" produce similar-looking
ratios, and only the denominator check tells them apart.
"""

from __future__ import annotations

import numpy as np

from src.eval.stats import Design, Result, difference, t_cdf

__all__ = ["SEED_DESIGN", "arm_difference", "transfer_ratio"]

# A seed is a sampled draw: we generalise to the environment's task distribution, not to
# these forty ticket sets. There are no enumerated factors and no within-cell subsamples
# — one episode per (arm, condition, seed) — so the design is just the pairing axis.
SEED_DESIGN = Design(
    item="seed", item_sampling="sampled", checkpoint="checkpoint", value="value"
)


def _rows(per_seed: dict[int, float], checkpoint: str) -> list[dict]:
    """{seed: value} -> the long rows src.eval.stats consumes."""
    return [
        {"checkpoint": checkpoint, "seed": int(s), "value": float(v)}
        for s, v in sorted(per_seed.items())
    ]


def _p_two_sided(r: Result) -> float:
    """Two-sided p-value from an interval result (0 when the SE is zero and the gap is not)."""
    if r.se == 0:
        return 1.0 if r.mean == 0 else 0.0
    return 2.0 * (1.0 - t_cdf(abs(r.mean) / r.se, r.df))


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
        The difference, its 95% interval, a two-sided p-value, the seeds it rests on, and
        the full `src.eval.stats` result under "stats".
    """
    shared = sorted(set(treatment) & set(control))
    assert len(shared) >= 2, (
        f"{label}: arms share {len(shared)} seeds; a paired interval needs at least 2. "
        "Seeds are meant to be identical across blocks — check that both arms ran the "
        "same --seeds list and that no episode was dropped."
    )
    r = difference(
        _rows({s: treatment[s] for s in shared}, "treatment"),
        _rows({s: control[s] for s in shared}, "control"),
        SEED_DESIGN,
    )
    return {
        "measure": label,
        "treatment_mean": round(float(np.mean([treatment[s] for s in shared])), 4),
        "control_mean": round(float(np.mean([control[s] for s in shared])), 4),
        "diff": round(r.mean, 4),
        "diff_ci95": [round(r.lo, 4), round(r.hi, 4)],
        "p_two_sided": round(_p_two_sided(r), 4),
        "n_seeds": len(shared),
        "method": r.method,
        "stats": r.as_dict(),
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
