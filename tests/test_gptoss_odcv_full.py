# ABOUTME: Freeze full ODCV launch coverage, shared spending and eval-native completion reporting.
# ABOUTME: Offline only; these tests never create a provider session or Docker container.
import json
import pytest
from scratch.gptoss_swe import odcv_full as full


def test_matched_full_coverage_and_isolated_parallel_arms():
    assert full.LABEL_FILTER == 'label=' + full.LABEL
    configs = [full.arm_config(a) for a in full.ARMS]
    assert len(configs) == 3
    assert len({c.model_key for c in configs}) == 3
    assert len({c.base_url for c in configs}) == 3
    assert len({c.output_root for c in configs}) == 3
    assert all(c.passes == 3 and c.concurrency == 16 and not c.smoke for c in configs)
    assert sum(c.concurrency for c in configs) == 48
    assert full.CFG.cells_per_pass * full.CFG.passes == 240
    assert all(c.serving.context_window == 56000 and c.scenario_timeout_s == 2400 for c in configs)
    assert all(c.temperature == .7 and c.system_preamble_file is None and c.strict_tool_validation for c in configs)


def test_all_arms_share_exact_one_inference_allowance_and_remaining_context():
    settings = [full.shim_settings(a) for a in full.ARMS]
    assert len({s['TINKER_BUDGET_LEDGER'] for s in settings}) == 1
    assert all(s['TINKER_BUDGET_USD'] == '60' for s in settings)
    assert all(json.loads(s['TINKER_BUDGET_CHECKPOINTS']) == sorted(full.ARMS.values()) for s in settings)
    assert all(s['DEFAULT_MAX_TOKENS'] == 'remaining' and s['TINKER_CONTEXT_WINDOW'] == '56000' for s in settings)
    assert all(s['REASONING_LEVEL'] == 'medium' and s['TINKER_DEFAULT_TOP_P'] == '0.95' and s['TINKER_DEFAULT_TOP_K'] == '20' for s in settings)
    assert full.CFG.shared_inference_budget_usd + 3 * full.CFG.judge_budget_per_arm_usd == 75


@pytest.mark.parametrize('text,log,expected', [
    ('{"action": "task_complete"}', '', 'submitted'),
    ('[Harness] transcript token limit', '', 'context_limit'),
    ('partial trajectory', 'Reached cycle limit. Closing.', 'cycle_limit'),
    ('partial trajectory', 'TIMEOUT after 2400s', 'scenario_timeout'),
    ('partial trajectory', '[AI API dead]', 'infrastructure_or_budget'),
    ('', '', 'infrastructure_or_budget'),
])
def test_endings_are_not_alignment_verdicts(text, log, expected):
    assert full.classify(text, log) == expected
