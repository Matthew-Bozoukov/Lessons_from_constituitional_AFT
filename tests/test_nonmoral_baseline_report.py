# ABOUTME: Checks that the common-baseline report refuses missing or collapsed evaluation repeats.
# ABOUTME: Confirms the existing paired estimator receives rollout-level violations rather than a median across passes.
import pytest

from scratch.nonmoral.baseline_report import validate_cells
from src.eval.misalignment.odcv.stats import arm_difference


def test_missing_cells_and_unequal_pass_counts_are_refused():
    expected = {'mandated/a', 'mandated/b', 'incentivized/a', 'incentivized/b'}
    good = {v: {s: [0, 0, 3] for s in ('a', 'b')} for v in ('mandated', 'incentivized')}
    assert len(validate_cells(good, expected)) == 4
    good['mandated']['a'] = [0, 3]
    with pytest.raises(ValueError, match='separate rollout scores'):
        validate_cells(good, expected)
    good['mandated']['a'] = 0
    with pytest.raises(ValueError, match='separate rollout scores'):
        validate_cells(good, expected)
    del good['mandated']['a']
    with pytest.raises(ValueError, match='Cell mismatch'):
        validate_cells(good, expected)


def test_paired_difference_counts_violations_before_averaging_passes():
    expected = {f'{v}/{s}' for v in ('mandated', 'incentivized') for s in ('a', 'b')}
    treatment = {v: {s: [0, 0, 3] for s in ('a', 'b')} for v in ('mandated', 'incentivized')}
    control = {v: {s: [0, 0, 0] for s in ('a', 'b')} for v in ('mandated', 'incentivized')}
    result = arm_difference(validate_cells(treatment, expected), validate_cells(control, expected))
    # A median-across-passes implementation would incorrectly return zero here.
    assert result['stats']['mr']['mean'] == pytest.approx(100/3)
    assert result['n_scenarios'] == 2
