# ABOUTME: Offline anti-retry-shopping and evidence-preservation tests for verified empty critic failures.
# ABOUTME: Billing proof integrity itself is covered by test_billing_evidence; these test the recovery boundary.
import json
import pytest
from scratch.dataset_refresh import retry_billing_verified_review as mod
from scratch.dataset_refresh.test_retry_technical_review import fixture as ordinary_fixture, snapshot


def fixture(tmp_path, monkeypatch):
    root, row, budget = ordinary_fixture(tmp_path, parse_error=True)
    entries = json.loads((budget / 'spend.json').read_text())
    raw_path = budget / 'raw_calls/000000.json'
    raw = json.loads(raw_path.read_text())
    error = 'Model anthropic/claude-sonnet-5 returned empty content (provider Anthropic): finish_reason=length'
    original = {**entries[0], 'model': 'anthropic/claude-sonnet-5', 'status': 'uncertain_failure', 'exception_type': 'EmptyCompletionError', 'error': error}
    raw.pop('response')
    raw['accounting'] = original
    raw['diagnostics'] = {'provider': 'Anthropic', 'provider_error': None,
        'choices': [{'finish_reason': 'length', 'dropped_content_chars': 0}]}
    entries[0] = {**original, 'status': 'billing_verified_failure'}
    mod.runtime.write_json(budget / 'spend.json', entries)
    mod.runtime.write_json(raw_path, raw)
    result = mod.runtime.load_checkpoint(row / 'result.json')
    result.update(error_type='EmptyCompletionError', error=error)
    mod.runtime.save_checkpoint(row / 'result.json', result)
    proof = budget / 'billing_reconciliations/proof'
    mod.runtime.write_json(proof / 'bound-proof.json', {'verified': True})
    monkeypatch.setattr(mod, 'validate_billing_call', lambda *args: proof)
    return root, row, budget, raw_path


def test_read_only_then_archive_without_touching_answer(tmp_path, monkeypatch):
    root, row, budget, _ = fixture(tmp_path, monkeypatch)
    before = snapshot(tmp_path)
    report = mod.recover(root)
    assert len(report['retryable']) == 1 and not report['refused']
    assert snapshot(tmp_path) == before
    mod.recover(root, apply=True)
    archive = row / 'recovered_failures/technical_retry_0/grounding_0'
    assert (archive / 'result.json').read_bytes() == before[str((row / 'result.json').relative_to(tmp_path))]
    assert not (row / 'result.json').exists()
    for name in ('scenario', 'preflight', 'draft', 'rewrite'):
        p = row / (name + '.json')
        assert p.read_bytes() == before[str(p.relative_to(tmp_path))]
    assert (budget / 'spend.json').read_bytes() == before[str((budget / 'spend.json').relative_to(tmp_path))]


@pytest.mark.parametrize('mutation', ['unverified', 'accepted', 'different_error', 'answer_changed', 'returned_verdict',
    'checkpoint', 'filter', 'refusal', 'dropped_content', 'provider_error', 'duplicate', 'later', 'retried', 'excluded', 'proof_failure'])
def test_refuse_unsafe_or_substantive_outcomes(tmp_path, monkeypatch, mutation):
    root, row, budget, raw_path = fixture(tmp_path, monkeypatch)
    raw = json.loads(raw_path.read_text())
    entries = json.loads((budget / 'spend.json').read_text())
    result = mod.runtime.load_checkpoint(row / 'result.json')
    if mutation == 'unverified': entries[0]['status'] = 'uncertain_failure'
    elif mutation == 'accepted': result['status'] = 'accepted'
    elif mutation == 'different_error': result['error'] = 'Something else'
    elif mutation == 'answer_changed': result['record']['response'] = 'Changed'
    elif mutation == 'returned_verdict': raw['response'] = {'content': '{"accepted":false}'}
    elif mutation == 'checkpoint': mod.runtime.save_checkpoint(row / 'grounding_0.json', {'accepted': False})
    elif mutation == 'filter': raw['diagnostics']['choices'][0]['finish_reason'] = 'content_filter'
    elif mutation == 'refusal': raw['diagnostics']['choices'][0]['refusal'] = 'No'
    elif mutation == 'dropped_content': raw['diagnostics']['choices'][0]['dropped_content_chars'] = 4
    elif mutation == 'provider_error': raw['diagnostics']['provider_error'] = 'Error'
    elif mutation in ('duplicate', 'later'):
        entries.append({**entries[0], 'call_id': 1, 'stage': 'review_0' if mutation == 'later' else 'grounding_0'})
    elif mutation == 'retried': mod.runtime.save_checkpoint(row / 'technical_retry_grounding_0.json', {})
    elif mutation == 'excluded': mod.runtime.save_checkpoint(row / 'independent_exclusion.json', {})
    elif mutation == 'proof_failure':
        def fail(*args): raise ValueError('Changed billing proof')
        monkeypatch.setattr(mod, 'validate_billing_call', fail)
    mod.runtime.save_checkpoint(row / 'result.json', result)
    mod.runtime.write_json(raw_path, raw)
    mod.runtime.write_json(budget / 'spend.json', entries)
    before = snapshot(tmp_path)
    assert not mod.recover(root)['retryable']
    assert snapshot(tmp_path) == before


def test_changed_proof_blocks_prepared_application(tmp_path, monkeypatch):
    root, _, budget, _ = fixture(tmp_path, monkeypatch)
    plan = mod.recover(root)['retryable'][0]
    mod.runtime.write_json(budget / 'billing_reconciliations/proof/bound-proof.json', {'changed': True})
    with pytest.raises(ValueError, match='Bound evidence changed'):
        mod.apply_plan(plan)
