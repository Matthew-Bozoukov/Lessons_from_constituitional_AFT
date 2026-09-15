# ABOUTME: Offline checks of blind critic calibration, exact evidence, shared cap, and no retry.
import json
from pathlib import Path

import pytest

from scratch.dataset_refresh import reviewer_probe as probe
from src.infra.endpoints.openrouter import ChatResult


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(probe.runtime, 'OpenRouterClient', lambda: pytest.fail('Paid transport forbidden'))
    monkeypatch.setattr(probe.runtime, 'provider_price', lambda _: {'in': 2., 'out': 10.})


def frozen(tmp_path):
    budget = tmp_path / 'budget'
    budget.mkdir()
    probe.runtime.write_json(budget / 'spend.json', [{'charged_or_reserved_usd': 5., 'status': 'settled'}])
    output = tmp_path / 'probe'
    conversation = {'system': 'Help.', 'user': 'Vote at the monthly meeting.',
                    'reasoning': 'Use another vote.', 'final': 'Hold a chat vote.'}
    manifest = {'models': list(probe.ALLOWED_MODELS), 'budget_root': str(budget),
                'allocation_key': 'narrow_critic_probe_allocation.json', 'max_probe_usd': 12,
                'prompt': probe.PROMPT.read_text(encoding='utf-8'),
                'code_sha256': probe.runtime.digest(Path(probe.__file__).read_bytes()),
                'cases': [{'id': 'a', 'expected': 'pass', 'conversation': conversation,
                           'rationale': 'SECRET LABEL', 'source_path': 'SECRET SOURCE'}], 'scope': 'test'}
    probe.runtime.save_checkpoint(output / 'manifest.json', manifest)
    return output, manifest


def test_blind_request_contains_only_conversation_and_prompt(tmp_path):
    _, manifest = frozen(tmp_path)
    request = probe.messages(manifest, manifest['cases'][0])
    assert set(json.loads(request[1]['content'])) == set(probe.FIELDS)
    assert 'SECRET' not in json.dumps(request)
    assert 'expected' not in json.loads(request[1]['content'])


def test_balanced_labels_and_no_withdrawn_negative():
    assert [c[3] for c in probe.CASES].count('pass') == 4
    assert [c[3] for c in probe.CASES].count('reject') == 4
    assert not any(c[2] in ('t7_000_v0', 't8_001_v0') and c[3] == 'reject' for c in probe.CASES)


def verdict():
    return {'accepted': False, 'assessment': 'Explicit process is contradicted.', 'issues': [
        {'kind': 'constraint_violation', 'answer_field': 'final', 'answer_quote': 'Hold a chat vote.',
         'premise_field': 'user', 'premise_quote': 'Vote at the monthly meeting.',
         'material_consequence': 'Proposed decision process violates the supplied requirement.',
         'alternative_reading': 'No request to authorize an exception is present.'}]}


@pytest.mark.parametrize('mutation', ['answer_quote', 'premise_quote', 'material_consequence', 'alternative_reading'])
def test_invalid_unsupported_or_unexplained_evidence_rejected(tmp_path, mutation):
    _, manifest = frozen(tmp_path)
    value = verdict()
    value['issues'][0][mutation] = ''
    with pytest.raises(ValueError):
        probe.validate_verdict(value, manifest['cases'][0]['conversation'])


def test_valid_exact_quoted_failure(tmp_path):
    _, manifest = frozen(tmp_path)
    assert probe.validate_verdict(verdict(), manifest['cases'][0]['conversation'])['accepted'] is False


def test_shared_allocation_cannot_reset_for_second_model_or_probe(tmp_path):
    _, manifest = frozen(tmp_path)
    assert probe.allocation(manifest)['absolute_ceiling_usd'] == 17
    probe.runtime.write_json(Path(manifest['budget_root']) / 'spend.json', [{'charged_or_reserved_usd': 9., 'status': 'settled'}])
    assert probe.allocation(dict(manifest))['absolute_ceiling_usd'] == 17


def test_no_haiku_even_with_injected_transport(tmp_path):
    output, _ = frozen(tmp_path)
    with pytest.raises(ValueError, match='no Haiku'):
        probe.run_probe(output, 'anthropic/claude-haiku-4.5', send=lambda **_: pytest.fail('Must not dispatch'))


def test_success_resume_does_not_repeat_paid_call(tmp_path):
    output, _ = frozen(tmp_path)
    calls = []
    def send(**kw):
        calls.append(kw)
        return ChatResult(content=json.dumps({'accepted': True, 'issues': [], 'assessment': 'No material defect.'}),
                          prompt_tokens=100, completion_tokens=30, finish_reason='stop', cost=.001, provider='test')
    first = probe.run_probe(output, send=send)
    second = probe.run_probe(output, send=lambda **_: pytest.fail('Must not retry'))
    assert first['correct'] == second['correct'] == 1
    assert len(calls) == 1


def test_failed_physical_call_retained_without_retry(tmp_path):
    output, manifest = frozen(tmp_path)
    def fail(**kw):
        raise TimeoutError('Uncertain billing')
    first = probe.run_probe(output, send=fail)
    second = probe.run_probe(output, send=lambda **_: pytest.fail('Must not retry'))
    assert first['complete'] == second['complete'] == 0
    entries = json.loads((Path(manifest['budget_root']) / 'spend.json').read_text())
    assert len(entries) == 2
    assert entries[-1]['status'] == 'uncertain_failure'


def test_malformed_model_result_remains_failure_not_false_negative(tmp_path):
    output, _ = frozen(tmp_path)
    response = ChatResult(content='{"accepted":true}', prompt_tokens=100, completion_tokens=10,
                          finish_reason='stop', cost=.001, provider='test')
    result = probe.run_probe(output, send=lambda **kw: response)
    assert result['complete'] == 0
    assert result['false_rejects'] == result['missed_defects'] == []
