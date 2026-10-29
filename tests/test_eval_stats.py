# ABOUTME: Unit tests for src/eval/stats.py: one pass gives no interval, n passes give a t over
# ABOUTME: them (log-odds for a bounded score, plain for the rest), Wilson and McNemar as before.

from __future__ import annotations

import math

import pytest

from src.eval.stats import Z_95, logit_t, mcnemar_exact, t_mean, t_quantile, wilson


def test_one_pass_reports_its_number_and_no_interval():
    for r in (t_mean([84.9]), logit_t([26.2], 80)):
        assert r["mean"] == pytest.approx(84.9 if r["values"] == [84.9] else 26.2)
        assert r["lo"] is None and r["hi"] is None and r["n"] == 1
        assert r["method"].startswith("none: one pass")


def test_t_mean_is_the_textbook_t_interval():
    r = t_mean([80.0, 84.0, 88.0])
    # sd 4, t_{0.975, 2} = 4.303: half-width 4.303 * 4 / sqrt(3) = 9.94
    assert r["mean"] == 84.0 and r["n"] == 3
    assert (r["hi"] - r["lo"]) / 2 == pytest.approx(9.94, abs=0.01)
    assert (r["hi"] + r["lo"]) / 2 == pytest.approx(84.0)


def test_logit_t_matches_the_worked_example():
    # Three ODCV passes over 80 cells: 26.2, 16.2, 20.0 -> mean 20.8, interval 11.2 to 35.6.
    r = logit_t([26.2, 16.2, 20.0], 80)
    assert r["mean"] == pytest.approx(20.8, abs=0.01)
    assert r["lo"] == pytest.approx(11.2, abs=0.05)
    assert r["hi"] == pytest.approx(35.6, abs=0.05)
    assert r["method"] == "t over 3 passes (df 2) on the log-odds scale"


def test_logit_t_is_asymmetric_and_stays_inside_the_range():
    r = logit_t([2.5, 5.0, 7.5], 80)
    assert 0.0 < r["lo"] < r["mean"] < r["hi"] < 100.0
    # near the floor the interval reaches further up than down
    assert r["hi"] - r["mean"] > r["mean"] - r["lo"]
    top = logit_t([97.5, 95.0, 92.5], 80)
    assert top["hi"] < 100.0 and top["mean"] - top["lo"] > top["hi"] - top["mean"]


def test_logit_t_handles_zero_and_full_passes_through_the_continuity_correction():
    r = logit_t([0.0, 0.0, 1.25], 80)
    assert r["lo"] >= 0.0 and r["hi"] < 100.0 and math.isfinite(r["hi"])
    both = logit_t([0.0, 0.0, 0.0], 80)      # no spread: a zero-width interval at the floor
    assert both["lo"] == pytest.approx(both["hi"], abs=1e-9)


def test_logit_t_on_another_scale():
    sev = logit_t([1.5, 1.7, 1.6], 80, scale=5.0)
    assert 0.0 < sev["lo"] < 1.6 < sev["hi"] < 5.0
    frac = logit_t([0.10, 0.20, 0.15], 18, scale=1.0)
    assert 0.0 < frac["lo"] < 0.15 < frac["hi"] < 1.0
    with pytest.raises(AssertionError):
        logit_t([101.0, 50.0], 80)


def test_identical_passes_give_a_zero_width_interval_not_a_crash():
    r = t_mean([50.0, 50.0, 50.0])
    assert r["lo"] == r["hi"] == 50.0
    r = logit_t([50.0, 50.0, 50.0], 80)
    assert r["lo"] == pytest.approx(50.0) and r["hi"] == pytest.approx(50.0)


def test_t_quantile_matches_tables():
    assert t_quantile(0.975, 2) == pytest.approx(4.3027, abs=1e-3)
    assert t_quantile(0.975, 9) == pytest.approx(2.2622, abs=1e-3)
    assert t_quantile(0.975, 39) == pytest.approx(2.0227, abs=1e-3)
    assert t_quantile(0.975, math.inf) == pytest.approx(Z_95, abs=1e-9)
    assert t_quantile(0.025, 2) == pytest.approx(-4.3027, abs=1e-3)


def test_wilson_stays_inside_the_unit_interval_at_the_edges():
    lo, hi = wilson(0, 30)
    assert lo == pytest.approx(0.0, abs=1e-12) and 0.0 < hi < 0.2
    lo, hi = wilson(30, 30)
    assert hi == 1.0 and 0.8 < lo < 1.0
    lo, hi = wilson(15, 30)
    assert lo < 0.5 < hi and hi - 0.5 == pytest.approx(0.5 - lo)


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 0) == 2 * 1 / 2**5
    assert mcnemar_exact(3, 3) == 1.0
    assert mcnemar_exact(0, 7) == mcnemar_exact(7, 0)
