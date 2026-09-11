# ABOUTME: Checks that the common-baseline report refuses missing or collapsed evaluation repeats.
# ABOUTME: Confirms the existing paired estimator receives rollout-level violations rather than a median across passes.
import pytest

from scratch.nonmoral.baseline_report import executor_terminal_events, health_evidence, scan_executor_logs, summarise_health, validate_cells
from src.eval.misalignment.odcv.stats import arm_difference


def test_matched_stakes_refuses_wrong_model_or_protocol(monkeypatch, tmp_path):
    from scratch.nonmoral import baseline_report as report
    pins = {k: {'repo': 'org/' + k, 'revision': c * 40}
            for k, c in [('low', 'a'), ('high', 'b')]}
    monkeypatch.setattr(report, 'scenario_names', lambda bench, variant: [str(i) for i in range(40)])
    def wrong_model(root, name, expected, **kwargs):
        return {'target': 'org/wrong'}, {}, {}
    monkeypatch.setattr(report, 'load_arm', wrong_model)
    with pytest.raises(ValueError, match='repository mismatch'):
        report.build_report({'low': 'low', 'high': 'high'}, tmp_path, matched_models=pins)
    def wrong_protocol(root, name, expected, **kwargs):
        assert kwargs['revision'] == pins[name]['revision']
        return {'target': pins[name]['repo']}, {}, {'temperature': 0.7 if name == 'low' else 0}
    monkeypatch.setattr(report, 'load_arm', wrong_protocol)
    with pytest.raises(ValueError, match='protocols differ'):
        report.build_report({'low': 'low', 'high': 'high'}, tmp_path, matched_models=pins)
    assert not (tmp_path / 'comparison.json').exists()


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


def test_executor_markers_are_anchored_and_missing_logs_stay_visible(tmp_path):
    text = '\n'.join([
        'executor-1 | [out]: {"stdout": "[AI API dead]: quoted tool output"}',
        'executor-1 | [Reply]: Reached cycle limit. Closing.',
        'orchestrator-1 | [AI API dead]: wrong service',
        'executor-1 | "[AI API dead]: quoted string"',
        '\x1b[Kexecutor-1      | [AI API dead]: Request timed out.',
        'executor-1      | Reached cycle limit. Closing.',
    ])
    events = executor_terminal_events(text)
    assert [(e['kind'], e['line']) for e in events] == [('api_death', 5), ('cycle_limit', 6)]
    path = tmp_path/'rollouts/mandated/a/pass1'
    path.mkdir(parents=True)
    (path/'docker_output.log').write_text(text)
    result = scan_executor_logs(tmp_path, {'mandated/a/pass1', 'mandated/a/pass2'})
    assert result['api_death_ids'] == ['mandated/a/pass1']
    assert result['cycle_limit_ids'] == ['mandated/a/pass1']
    assert result['missing_log_ids'] == ['mandated/a/pass2']
    assert result['inspected_logs'] == 1
    assert len(result['log_sha256']['mandated/a/pass1']) == 64
