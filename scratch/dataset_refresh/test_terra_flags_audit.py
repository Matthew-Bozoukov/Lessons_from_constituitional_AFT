# ABOUTME: Offline checks for deterministic unaudited sampling and isolated blind requests.
# ABOUTME: Exercise concurrent local/global budget bounds and retained nonretryable outcomes.
import json
from pathlib import Path

import pytest

from scratch.dataset_refresh import terra_flags_audit as mod
from src.infra.endpoints.openrouter import ChatResult


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(mod.runtime, 'OpenRouterClient', lambda: pytest.fail('No network in tests'))
    monkeypatch.setattr(mod.runtime, 'provider_price', lambda _: {'in': 2., 'out': 12.})


def frozen(tmp_path, monkeypatch, count=16):
    output, budget = tmp_path / 'probe', tmp_path / 'budget'
    monkeypatch.setattr(mod, 'BUDGET', budget)
    cases = [{'id': f'case_{i}', 'source_sha256': 'sourcehash',
              'author_metadata': 'SECRET TARGET', 'prior_verdict': 'SECRET VERDICT',
              'conversation': {'system': 'Be helpful.', 'user': 'Advise on a choice.',
                               'reasoning': 'Consider tradeoffs.', 'final': 'Choose an option.'}}
             for i in range(count)]
    mod.runtime.save_checkpoint(output / 'selection.json', {'selected': list(range(count))})
    manifest = {'model': mod.MODEL, 'cases': cases, 'budget_root': str(budget.resolve()),
                'max_tokens': 6000, 'max_calls': count, 'price': {'in': 2., 'out': 12.},
                'prompts': {'system': 'Blind critique only.', 'user': '{conversation_json}'},
                'selection_sha256': mod.runtime.digest((output / 'selection.json').read_bytes()),
                'frozen_code_sha256': {n: mod.runtime.digest(p.read_bytes()) for n, p in mod.code_files().items()}}
    mod.runtime.save_checkpoint(output / 'manifest.json', manifest)
    mod.runtime.save_checkpoint(budget / mod.ALLOWANCE, {'output_root': str(output.resolve()),
        'allowance_usd': 6., 'manifest_sha256': mod.runtime.digest(manifest), 'entries': []})
    mod.runtime.write_json(budget / 'spend.json', [])
    return output, budget


def response(content=None):
    return ChatResult(content=content or json.dumps({'accepted': True, 'issues': [], 'assessment': 'No defect.'}),
                      prompt_tokens=100, completion_tokens=20, cost=.001, finish_reason='stop', provider='OpenAI')


def test_selection_reproducible_excludes_reviewed_and_reports_shortage():
    pool = [{'arm': a, 'trait_id': f't{t}', 'candidate_id': f't{t}_{i}'}
            for a in (mod.LOW, mod.NON) for t in range(1, 10) for i in range(11 if t != 9 else 2)]
    excluded = {a: {'t1_0', 't1_1'} for a in (mod.LOW, mod.NON)}
    selected, counts = mod.select_pool(pool, excluded)
    assert (selected, counts) == mod.select_pool(list(reversed(pool)), excluded)
    assert all(e['candidate_id'] not in excluded[e['arm']] for e in selected)
    assert counts[mod.LOW + '/t9'] == {'available': 2, 'selected': 2, 'shortage': 7}
    assert len(selected) == 148


def test_parallel_blind_calls_and_no_retries(tmp_path, monkeypatch):
    output, budget = frozen(tmp_path, monkeypatch)
    calls = []
    def send(**kw):
        calls.append(kw)
        assert 'SECRET' not in json.dumps(kw)
        assert kw['model'] == mod.MODEL and kw['max_tokens'] == 6000 and 'extra_body' not in kw
        assert set(json.loads(kw['messages'][1]['content'])) == set(mod.FIELDS)
        return response()
    summary = mod.run_audit(output, workers=12, send=send)
    assert len(calls) == summary['complete'] == 16
    assert len(list((output / 'audit/raw_calls').glob('*.json'))) == 16
    assert mod.run_audit(output, send=lambda **_: pytest.fail('No repeat'))['physical_calls'] == 16
    local = mod.runtime.load_checkpoint(budget / mod.ALLOWANCE)
    assert sum(e['charged_or_reserved_usd'] for e in local['entries']) < 6


@pytest.mark.parametrize('mode', ['unknown', 'invalid'])
def test_failed_outcomes_never_repeat(tmp_path, monkeypatch, mode):
    output, _ = frozen(tmp_path, monkeypatch, count=1)
    def send(**kw):
        if mode == 'unknown':
            raise RuntimeError('uncertain completion')
        return response('invalid json')
    summary = mod.run_audit(output, send=send)
    assert summary['physical_calls'] == 1 and len(summary['incomplete']) == 1
    mod.run_audit(output, send=lambda **_: pytest.fail('No failed retry'))


def test_global_limit_stops_without_call(tmp_path, monkeypatch):
    output, budget = frozen(tmp_path, monkeypatch, count=1)
    mod.runtime.write_json(budget / 'spend.json', [{'arm': 'production', 'charged_or_reserved_usd': 249.99, 'status': 'settled'}])
    summary = mod.run_audit(output, send=lambda **_: pytest.fail('No call over250'))
    assert summary['physical_calls'] == 0 and len(summary['incomplete']) == 1


def test_local_reservations_include_unresolved_calls_and_cannot_reset(tmp_path, monkeypatch):
    allocation = {'entries': [{'id': 'old', 'charged_or_reserved_usd': 5.99}]}
    with pytest.raises(mod.runtime.BudgetStop, match='ceiling'):
        mod.admit(allocation, 'new', .02, 162)
    with pytest.raises(mod.runtime.BudgetStop, match='retry'):
        mod.admit(allocation, 'old', .001, 162)
    output, budget = frozen(tmp_path, monkeypatch, count=1)
    a = mod.runtime.load_checkpoint(budget / mod.ALLOWANCE)
    a['allowance_usd'] = 7
    mod.runtime.save_checkpoint(budget / mod.ALLOWANCE, a)
    with pytest.raises(ValueError, match='Allocation'):
        mod.run_audit(output, send=lambda **_: pytest.fail('No new allocation'))


def test_parallel_workers_cannot_overreserve_local_allowance(tmp_path, monkeypatch):
    output, budget = frozen(tmp_path, monkeypatch)
    allocation = mod.runtime.load_checkpoint(budget / mod.ALLOWANCE)
    allocation['entries'].append({'id': 'previous_unknown', 'charged_or_reserved_usd': 5.85})
    mod.runtime.save_checkpoint(budget / mod.ALLOWANCE, allocation)
    calls = []
    def send(**kw):
        calls.append(kw)
        raise RuntimeError('unknown charge retains full reservation')
    summary = mod.run_audit(output, workers=12, send=send)
    current = mod.runtime.load_checkpoint(budget / mod.ALLOWANCE)
    assert len(calls) == summary['physical_calls'] == 1
    assert sum(e['charged_or_reserved_usd'] for e in current['entries']) <= 6
