# ABOUTME: Checks that the common-baseline report refuses missing or collapsed evaluation repeats.
# ABOUTME: Confirms the existing paired estimator receives rollout-level violations rather than a median across passes.
import pytest

from scratch.nonmoral.baseline_report import health_evidence, summarise_health, validate_cells
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


def test_health_preserves_unknown_flags_and_separate_recovery_evidence():
    evidence = [
        health_evidence({'status': 'timeout', 'token_limit_hit': True, 'transcript_partial': True,
                         'transcript_source': 'docker_log_reconstruction'}, 'partial transcript'),
        health_evidence({'status': 'cached'}, '[Harness] transcript token limit\n'
                        '== [transcript truncated: container killed before completion] =='),
        health_evidence({'status': 'ok', 'token_limit_hit': False, 'transcript_partial': False,
                         'transcript_source': 'executor_archive'}, 'ordinary complete archive'),
    ]
    result = summarise_health(evidence, [{'reconstructed': 2}, {}])
    assert result['timeout_statuses'] == 1
    assert result['token_limit']['positive'] == 2
    assert result['partial']['positive'] == 2
    assert result['partial']['explicit_false'] == 1
    assert result['partial']['unknown_flag'] == 1  # Missing cached metadata is not false.
    assert result['identified_timeout_reconstructions'] == 1
    assert result['post_pass_reconstructions_reported'] == 2
    assert result['reconstruction_audits_unknown'] == 1  # Missing audit is not a recorded zero.
