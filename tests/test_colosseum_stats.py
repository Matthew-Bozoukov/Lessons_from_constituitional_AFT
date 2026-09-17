# ABOUTME: Unit tests for the Colosseum contrast statistics — seed pairing, and the
# ABOUTME: transfer ratio's refusal to report when its denominator is not separated from zero.

import numpy as np
import pytest

from src.eval.misalignment.colosseum.stats import arm_difference, transfer_ratio


def _cell(mean, n=40, sd=0.05, seed=0):
    rng = np.random.default_rng(seed)
    return {s: float(v) for s, v in enumerate(rng.normal(mean, sd, n))}


def test_arm_difference_recovers_a_planted_gap():
    control = _cell(0.30, seed=1)
    treatment = _cell(0.18, seed=2)
    r = arm_difference(treatment, control, label="coalition_advantage")
    assert r["n_seeds"] == 40
    assert r["diff"] == pytest.approx(-0.12, abs=0.03)
    lo, hi = r["diff_ci95"]
    assert lo < r["diff"] < hi
    assert hi < 0, "a planted 0.12 reduction at sd 0.05 must exclude zero"
    assert r["p_two_sided"] < 0.001


def test_arm_difference_refuses_arms_that_share_almost_no_seeds():
    # Seeds are meant to be identical across blocks; near-disjoint seed sets mean an arm
    # silently ran a different list, which would otherwise produce an interval over one
    # accidental overlap.
    with pytest.raises(AssertionError, match="share 1 seeds"):
        arm_difference({7: 0.1}, {7: 0.2, 8: 0.3}, label="coalition_advantage")


def test_transfer_ratio_near_one_when_both_effects_match():
    # Treatment reduces the measure by 0.10 in BOTH the multi-agent and single-agent
    # settings: the disposition carries into the team unchanged.
    mc, mt = _cell(0.30, seed=1), _cell(0.20, seed=2)
    sc, st = _cell(0.25, seed=3), _cell(0.15, seed=4)
    r = transfer_ratio(mt, mc, st, sc, n_boot=2000)
    assert r["interpretable"] is True
    assert r["ratio"] == pytest.approx(1.0, abs=0.25)
    lo, hi = r["ratio_ci95"]
    assert lo < r["ratio"] < hi


def test_transfer_ratio_well_below_one_when_the_effect_leaks_in_the_team():
    # The hypothesis under test: a large single-agent effect, a small multi-agent one.
    mc, mt = _cell(0.30, seed=1), _cell(0.28, seed=2)  # multi effect ~0.02
    sc, st = _cell(0.25, seed=3), _cell(0.15, seed=4)  # single effect ~0.10
    r = transfer_ratio(mt, mc, st, sc, n_boot=2000)
    assert r["interpretable"] is True
    assert r["ratio"] < 0.5


def test_transfer_ratio_refuses_when_the_single_agent_effect_covers_zero():
    # No single-agent effect to divide by. The ratio is arithmetically computable and
    # scientifically empty, so the gate must fire rather than let it be reported.
    mc, mt = _cell(0.30, seed=1), _cell(0.20, seed=2)
    sc, st = _cell(0.25, seed=3), _cell(0.25, seed=4)  # single effect ~0
    r = transfer_ratio(mt, mc, st, sc, n_boot=2000)
    assert r["interpretable"] is False
    assert "not separated from zero" in r["refused_because"]
    lo, hi = r["single_agent_effect_ci95"]
    assert lo < 0 < hi


def test_transfer_ratio_pairs_the_two_experiments_on_one_seed_draw():
    # Both effects must be recomputed on the SAME resampled seeds. If they were drawn
    # independently the ratio's interval would be wider than the pairing warrants; this
    # pins the shared-draw behaviour by checking the reported seed count is the
    # intersection of all four cells, not the union.
    mc, mt = _cell(0.30, n=40, seed=1), _cell(0.20, n=40, seed=2)
    sc, st = _cell(0.25, n=25, seed=3), _cell(0.15, n=25, seed=4)
    r = transfer_ratio(mt, mc, st, sc, n_boot=500)
    assert r["n_seeds"] == 25
