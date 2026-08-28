# ABOUTME: Unit tests for src/eval/stats.py: the three spreads are unbiased for Var(mu_hat),
# ABOUTME: the special cases collapse to the textbook formulas, and R=1 refuses what it cannot claim.

from __future__ import annotations

import numpy as np
import pytest

from src.eval.stats import (
    Design, NotEstimable, cluster_bootstrap, collapse, crossed_terms, difference, interval,
    mcnemar_exact, satterthwaite, t_quantile, wilson,
)

D = Design(unit="scenario", nested=("pass",))


def synth(rng, n, J, R, sA=0.05, sB=0.1, sC=0.03, noise=0.2, mu=0.4, model_prefix="m"):
    """Long table from the four-piece model: cell rate = mu + A_i + B_j + C_ij, R Bernoulli-ish draws.

    Values are continuous (rate + N(0, noise)/sqrt-free) so the four sigma^2 are exactly the
    inputs; rollout noise is Gaussian with variance `noise`^2 per draw.
    """
    A = rng.normal(0, sA, n)
    B = rng.normal(0, sB, J)
    C = rng.normal(0, sC, (n, J))
    rows = []
    for i in range(n):
        for j in range(J):
            rate = mu + A[i] + B[j] + C[i, j]
            for r in range(R):
                rows.append({"model": f"{model_prefix}{i}", "scenario": f"s{j}", "pass": r,
                             "value": rate + rng.normal(0, noise)})
    return rows


# --------------------------------------------------------------------------- core algebra

def test_three_spreads_are_unbiased_for_var_of_mean():
    """E[T_A + T_B - T_C] == Var(mu_hat) = sA^2/n + sB^2/J + sC^2/(nJ) + noise^2/(nJR)."""
    rng = np.random.default_rng(0)
    n, J, R = 4, 20, 2
    sA, sB, sC, noise = 0.05, 0.10, 0.04, 0.15
    truth = sA ** 2 / n + sB ** 2 / J + sC ** 2 / (n * J) + noise ** 2 / (n * J * R)
    est, means = [], []
    for _ in range(400):
        t = collapse(synth(rng, n, J, R, sA, sB, sC, noise), D)
        c = crossed_terms(t.values)
        est.append(c["T_A"] + c["T_B"] - c["T_C"])
        means.append(c["mu"])
    assert np.mean(est) == pytest.approx(truth, rel=0.15)
    assert np.var(means, ddof=1) == pytest.approx(truth, rel=0.25)   # the estimator's actual wobble


def test_each_spread_estimates_its_own_term_plus_beta():
    rng = np.random.default_rng(1)
    n, J, R = 3, 30, 1
    sA, sB, sC, noise = 0.08, 0.06, 0.05, 0.1
    beta = sC ** 2 / (n * J) + noise ** 2 / (n * J * R)
    TA, TB, TC = [], [], []
    for _ in range(400):
        c = crossed_terms(collapse(synth(rng, n, J, R, sA, sB, sC, noise), D).values)
        TA.append(c["T_A"]); TB.append(c["T_B"]); TC.append(c["T_C"])
    assert np.mean(TA) == pytest.approx(sA ** 2 / n + beta, rel=0.15)
    assert np.mean(TB) == pytest.approx(sB ** 2 / J + beta, rel=0.15)
    assert np.mean(TC) == pytest.approx(beta, rel=0.15)


def test_one_model_collapses_to_miller_s2_over_J():
    rng = np.random.default_rng(2)
    obs = synth(rng, 1, 25, 3)
    r = interval(obs, D)
    t = collapse(obs, D)
    assert r.models == "fixed" and r.n_models == 1
    assert r.se ** 2 == pytest.approx(t.values[0].var(ddof=1) / 25)
    assert r.df == 24 and r.mult == pytest.approx(t_quantile(0.975, 24))


def test_fixed_units_uses_T_A_only():
    rng = np.random.default_rng(3)
    obs = synth(rng, 3, 20, 1)
    r = interval(obs, Design(unit="scenario", units="fixed", nested=("pass",)))
    t = collapse(obs, D)
    assert r.se ** 2 == pytest.approx(t.values.mean(axis=1).var(ddof=1) / 3)
    assert r.df == 2 and r.units == "fixed" and r.models == "random"


def test_random_random_uses_all_three():
    rng = np.random.default_rng(4)
    r = interval(synth(rng, 3, 20, 1), D)
    assert set(r.terms) >= {"T_A", "T_B", "T_C"}
    assert r.se ** 2 == pytest.approx(r.terms["T_A"] + r.terms["T_B"] - r.terms["T_C"])
    # Satterthwaite: the scenario term dominates here, so nu sits near J-1, not n-1.
    assert 2 < r.df < 19 and r.mult == pytest.approx(t_quantile(0.975, r.df))


def test_satterthwaite_tracks_the_dominant_term():
    """One part dominating -> its own df; equal parts -> more df than either alone."""
    assert satterthwaite([(1.0, 5), (1e-9, 100)]) == pytest.approx(5, rel=1e-3)
    assert satterthwaite([(1e-9, 5), (1.0, 100)]) == pytest.approx(100, rel=1e-3)
    assert satterthwaite([(1.0, 10), (1.0, 10)]) == pytest.approx(20)      # 4 / (2 * 1/10)
    assert satterthwaite([(1.0, 2), (1.0, 40)]) < 10                        # the noisy part rules
    assert satterthwaite([(1.0, 2), (-0.5, 8)]) < satterthwaite([(1.0, 2)]) + 1e-9  # subtraction costs df


def test_model_dominated_composite_gets_a_small_df():
    """Big seed-to-seed spread, tiny unit spread: nu must fall back toward n-1, not stay at J-1."""
    rng = np.random.default_rng(21)
    r = interval(synth(rng, 3, 20, 1, sA=0.5, sB=0.005, sC=0.005, noise=0.005), D)
    assert r.terms["T_A"] > 20 * r.terms["T_B"]
    assert r.df < 4 and r.mult > 3.0      # +/-1.96 here would be an ~81% interval


# --------------------------------------------------------------------------- rollouts

def test_nested_factor_never_adds_a_term():
    """Three identical rollouts per cell give exactly the R=1 interval."""
    rng = np.random.default_rng(5)
    one = synth(rng, 3, 15, 1)
    three = [{**row, "pass": k} for row in one for k in range(3)]
    a, b = interval(one, D), interval(three, D)
    assert a.se == pytest.approx(b.se) and a.mean == pytest.approx(b.mean)
    assert a.noise["estimable"] is False and b.noise["estimable"] is True
    assert b.noise["term"] == pytest.approx(0.0)


def test_one_rollout_claims_and_noise_flags():
    rng = np.random.default_rng(6)
    r = interval(synth(rng, 2, 10, 1), D)
    assert r.rollouts == {"min": 1, "max": 1, "mean": 1.0}
    assert r.noise["estimable"] is False
    assert any("one rollout per cell" in c for c in r.claims)


def test_multiple_rollouts_report_noise_share():
    rng = np.random.default_rng(7)
    r = interval(synth(rng, 2, 10, 5, noise=0.3), D)
    assert r.noise["estimable"] and 0 < r.noise["share"] < 1
    assert any("5-5 draws per cell" in c for c in r.claims)


def test_both_fixed_refuses_one_rollout_but_works_with_repeats():
    rng = np.random.default_rng(8)
    fixed = Design(unit="scenario", units="fixed", nested=("pass",))
    with pytest.raises(NotEstimable, match="one rollout"):
        interval(synth(rng, 1, 10, 1), fixed, models="fixed")
    r = interval(synth(rng, 1, 10, 4, noise=0.2), fixed, models="fixed")
    assert r.se > 0 and "rollout noise only" in r.method


def test_random_models_needs_two_checkpoints():
    rng = np.random.default_rng(9)
    with pytest.raises(NotEstimable, match="one checkpoint"):
        interval(synth(rng, 1, 10, 1), D, models="random")


# --------------------------------------------------------------------------- crossed_fixed (variants)

def _two_variant_obs(rng, n, J, R=1):
    rows = []
    for i in range(n):
        for j in range(J):
            for k, base in (("mandated", 0.6), ("incentivized", 0.3)):
                for r in range(R):
                    rows.append({"model": f"m{i}", "scenario": f"s{j}", "variant": k, "pass": r,
                                 "value": float(rng.random() < base)})
    return rows


def test_fixed_factor_collapses_with_equal_weights():
    rng = np.random.default_rng(10)
    obs = _two_variant_obs(rng, 2, 12)
    d = Design(unit="scenario", crossed_fixed={"variant": "equal"}, nested=("pass",))
    t = collapse(obs, d)
    by_level = {k: collapse([o for o in obs if o["variant"] == k], D).values for k in ("mandated", "incentivized")}
    assert np.allclose(t.values, 0.5 * (by_level["mandated"] + by_level["incentivized"]))
    assert t.counts.min() == 2


def test_balanced_fixed_factor_matches_miller_clustered_se():
    """Miller's clustered SE (clusters = scenarios of 2 cells) == (J-1)/J * T_B."""
    rng = np.random.default_rng(11)
    obs = _two_variant_obs(rng, 1, 20)
    d = Design(unit="scenario", crossed_fixed={"variant": "equal"}, nested=("pass",))
    t = collapse(obs, d)
    J = t.J
    cells = np.array([[o["value"] for o in obs if o["scenario"] == u] for u in t.units])   # (J, 2)
    s_bar = cells.mean()
    miller = ((cells - s_bar).sum(axis=1) ** 2).sum() / (2 * J) ** 2
    assert miller == pytest.approx((J - 1) / J * crossed_terms(t.values)["T_B"])


def test_variant_contrast_is_paired_on_both_axes():
    rng = np.random.default_rng(12)
    obs = _two_variant_obs(rng, 3, 15)
    mand = [o for o in obs if o["variant"] == "mandated"]
    inc = [o for o in obs if o["variant"] == "incentivized"]
    r = difference(mand, inc, D, paired_models=True)
    tm, ti = collapse(mand, D), collapse(inc, D)
    assert r.mean == pytest.approx(tm.values.mean() - ti.values.mean())
    assert "paired on units and checkpoints" in r.estimand
    assert r.mean > 0   # mandated is built to violate more


def test_missing_fixed_level_drops_unit_or_errors():
    rng = np.random.default_rng(13)
    obs = [o for o in _two_variant_obs(rng, 2, 8) if not (o["scenario"] == "s3" and o["variant"] == "mandated")]
    d = Design(unit="scenario", crossed_fixed={"variant": "equal"}, nested=("pass",))
    t = collapse(obs, d)
    assert t.dropped_units == ["s3"] and t.J == 7
    with pytest.raises(NotEstimable, match="missing"):
        collapse(obs, Design(unit="scenario", crossed_fixed={"variant": "equal"}, nested=("pass",), incomplete="error"))


def test_unit_missing_for_one_model_is_dropped_everywhere():
    rng = np.random.default_rng(14)
    obs = [o for o in synth(rng, 3, 10, 1) if not (o["model"] == "m2" and o["scenario"] == "s7")]
    t = collapse(obs, D)
    assert t.dropped_units == ["s7"] and t.values.shape == (3, 9)
    assert not np.isnan(t.values).any()


# --------------------------------------------------------------------------- differences between arms

def test_difference_of_identical_arms_is_zero_and_pairs_units():
    rng = np.random.default_rng(15)
    a = synth(rng, 3, 20, 1, model_prefix="a")
    b = [dict(o, model="b" + o["model"][1:]) for o in a]
    r = difference(a, b, D)
    assert r.mean == pytest.approx(0.0)
    assert r.terms["T_B_d"] == pytest.approx(0.0)         # shared unit difficulty cancels exactly
    assert "paired on units" in r.estimand and r.n_models == 6


def test_paired_difference_is_tighter_than_unpaired_when_units_shared():
    rng = np.random.default_rng(16)
    n, J = 3, 25
    A = rng.normal(0, 0.05, n); B = rng.normal(0, 0.2, J)     # big shared unit effects
    def arm(prefix, shift):
        return [{"model": f"{prefix}{i}", "scenario": f"s{j}", "pass": 0,
                 "value": 0.4 + shift + A[i] + B[j] + rng.normal(0, 0.05)} for i in range(n) for j in range(J)]
    a, b = arm("a", 0.1), arm("b", 0.0)
    paired = difference(a, b, D)
    ta, tb = crossed_terms(collapse(a, D).values), crossed_terms(collapse(b, D).values)
    unpaired_se2 = ta["T_A"] + tb["T_A"] + ta["T_B"] + tb["T_B"] - ta["T_C"] - tb["T_C"]
    assert paired.se ** 2 < 0.5 * unpaired_se2
    assert paired.lo > 0


def test_difference_single_models_uses_per_unit_differences():
    rng = np.random.default_rng(17)
    a, b = synth(rng, 1, 15, 1, model_prefix="a"), synth(rng, 1, 15, 1, model_prefix="b")
    r = difference(a, b, D)
    d = collapse(a, D).values[0] - collapse(b, D).values[0]
    assert r.se ** 2 == pytest.approx(d.var(ddof=1) / 15) and r.df == 14 and r.models == "fixed"


# --------------------------------------------------------------------------- bootstrap + helpers

def test_cluster_bootstrap_of_mean_agrees_with_closed_form():
    rng = np.random.default_rng(18)
    obs = synth(rng, 1, 40, 1, sB=0.15)
    closed = interval(obs, D)
    boot = cluster_bootstrap(obs, lambda v: float(v.mean()), D, n_boot=4000, seed=0)
    assert boot["se"] == pytest.approx(closed.se, rel=0.15)
    assert boot["mean"] == pytest.approx(closed.mean)


def test_cluster_bootstrap_resamples_rows_only_when_models_random():
    rng = np.random.default_rng(19)
    obs = synth(rng, 3, 20, 1)
    both = cluster_bootstrap(obs, lambda v: float(v.mean()), D, n_boot=2000, seed=0)
    units_only = cluster_bootstrap(obs, lambda v: float(v.mean()), D, models="fixed", n_boot=2000, seed=0)
    assert "models and units" in both["method"] and units_only["method"].startswith("cluster bootstrap over units")


def test_t_quantile_matches_tables():
    assert t_quantile(0.975, 2) == pytest.approx(4.3027, abs=1e-3)
    assert t_quantile(0.975, 24) == pytest.approx(2.0639, abs=1e-3)
    assert t_quantile(0.975, 1000) == pytest.approx(1.9623, abs=1e-3)
    assert t_quantile(0.975, float("inf")) == pytest.approx(1.95996, abs=1e-4)
    assert t_quantile(0.025, 5) == pytest.approx(-2.5706, abs=1e-3)


def test_wilson_stays_inside_the_unit_interval_at_the_edges():
    lo, hi = wilson(0, 40)
    assert lo == 0.0 and 0.08 < hi < 0.09          # not a zero-width "certainly 0%"
    lo, hi = wilson(40, 40)
    assert hi == 1.0 and 0.91 < lo < 0.92


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 0) == pytest.approx(2 / 32)
    assert mcnemar_exact(3, 3) == 1.0


def test_design_from_config_rejects_unknown_keys_and_parses_lists():
    d = Design.from_config({"unit": "scenario", "crossed_fixed": {"variant": "equal"}, "nested": ["pass"]})
    assert d.nested == ("pass",) and d.crossed_fixed == {"variant": "equal"}
    with pytest.raises(AssertionError, match="unknown Design keys"):
        Design.from_config({"unit": "scenario", "models": "random"})


def test_result_as_dict_is_json_safe():
    import json
    rng = np.random.default_rng(20)
    d = interval(synth(rng, 2, 10, 1), D).as_dict()
    json.dumps(d)
    assert d["ci95"] == [d["lo"], d["hi"]] and d["noise"]["sigma_eps2"] is None
