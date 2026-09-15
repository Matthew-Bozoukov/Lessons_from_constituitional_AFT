# ABOUTME: Offline tests for blind Luna/Terra calibration, retained failures and nonresetting budget limits.
import json
from pathlib import Path

import pytest

from scratch.dataset_refresh import reviewer_probe_openai as mod
from src.infra.endpoints.openrouter import ChatResult


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(mod.runtime, 'OpenRouterClient', lambda: pytest.fail('No network in tests'))
    monkeypatch.setattr(mod.runtime, 'provider_price', lambda _: {'in': 2., 'out': 12.})


def frozen(tmp_path):
    output, budget = tmp_path / 'probe', tmp_path / 'budget'
    cases = [{'id': f'case_{i}', 'expected': 'reject' if i % 2 == 0 else 'pass',
              'independent_annotation': {'secret_expected': 'DO NOT SHOW'},
              'conversation': {'system': 'Be helpful.', 'user': 'Advise on a choice.',
                               'reasoning': 'Consider tradeoffs.', 'final': 'Choose an option.'}} for i in range(20)]
    manifest = {'models': list(mod.MODELS), 'cases': cases, 'budget_root': str(budget),
        'prompts': {'system': 'Blind critique only.', 'user': '{conversation_json}'}, 'scope': 'Test fixture.',
        'code_sha256': mod.runtime.digest(Path(mod.__file__).read_bytes()),
        'runtime_sha256': mod.runtime.digest(Path(mod.runtime.__file__).read_bytes()),
        'validator_sha256': mod.runtime.digest(Path(mod.__file__).with_name('reviewer_probe.py').read_bytes())}
    mod.runtime.save_checkpoint(output / 'manifest.json', manifest)
    mod.runtime.save_checkpoint(budget / mod.ALLOCATION, {'probe_root': str(output.resolve()),
        'allowance_usd': 2, 'manifest_sha256': mod.runtime.digest(manifest), 'max_calls': 40})
    mod.runtime.write_json(budget / 'spend.json', [])
    return output, budget


def test_exact40_blind_calls_default_reasoning_no_repeat(tmp_path):
    output, _ = frozen(tmp_path)
    calls = []
    def send(**kw):
        calls.append(kw)
        assert 'extra_body' not in kw and kw['max_tokens'] == 6000
        assert 'DO NOT SHOW' not in json.dumps(kw)
        assert set(json.loads(kw['messages'][1]['content'])) == {'system', 'user', 'reasoning', 'final'}
        return ChatResult(content=json.dumps({'accepted': True, 'issues': [], 'assessment': 'No material defect.'}),
                          prompt_tokens=100, completion_tokens=20, cost=.001, finish_reason='stop', provider='test')
    result = mod.run_probe(output, send=send)
    assert len(calls) == result['physical_calls'] == 40
    assert all(x['complete'] == 20 for x in result['models'].values())
    assert mod.run_probe(output, send=lambda **kw: pytest.fail('No repeat'))['physical_calls'] == 40


def test_unknown_failed_call_is_retained_and_not_repeated(tmp_path):
    output, _ = frozen(tmp_path)
    count = 0
    def send(**kw):
        nonlocal count
        count += 1
        if count == 1:
            raise RuntimeError('Unknown provider failure')
        return ChatResult(content=json.dumps({'accepted': True, 'issues': [], 'assessment': 'Pass.'}),
                          prompt_tokens=100, completion_tokens=20, cost=.001, finish_reason='stop', provider='test')
    report = mod.run_probe(output, send=send)
    assert count == 40
    assert len(report['models'][mod.MODELS[0]]['incomplete']) == 1
    assert report['charged_or_reserved_usd'] > report['reported_cost_usd']
    mod.run_probe(output, send=lambda **kw: pytest.fail('No uncertain retry'))


def test_probe2_limit_and_global250_both_hold(tmp_path):
    with pytest.raises(mod.runtime.BudgetStop, match='ceiling'):
        mod.reserve_check([{'arm': mod.ARM, 'charged_or_reserved_usd': 1.99}], mod.MODELS[1], [])
    output, budget = frozen(tmp_path)
    mod.runtime.write_json(budget / 'spend.json', [{'arm': 'production', 'charged_or_reserved_usd': 249.99, 'status': 'settled'}])
    with pytest.raises(mod.runtime.BudgetStop, match='Budget stop'):
        mod.run_probe(output, send=lambda **kw: pytest.fail('Do not cross250'))


def test_shared_allocation_cannot_move_or_change(tmp_path):
    output, budget = frozen(tmp_path)
    value = mod.runtime.load_checkpoint(budget / mod.ALLOCATION)
    value['allowance_usd'] = 3
    mod.runtime.save_checkpoint(budget / mod.ALLOCATION, value)
    with pytest.raises(ValueError, match='allocation'):
        mod.run_probe(output, send=lambda **kw: pytest.fail('No call'))
