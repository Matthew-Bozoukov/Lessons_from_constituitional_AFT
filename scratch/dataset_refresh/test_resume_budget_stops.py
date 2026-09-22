# ABOUTME: Offline tests of pure reservation-stop recovery and a single imported BudgetStop identity.
# ABOUTME: Tests cached-stage resume, refusal of unknown costs/quality failures, and immutable invocation evidence.
import json
from pathlib import Path

import pytest

from scratch.dataset_refresh import resume_budget_stops as mod, execute_imported
from scratch.dataset_refresh.test_recover_missing_changes import ready, RecordedClient
from scratch.dataset_refresh.test_per_row import case, offline, client_for, FakeClient


FOREIGN = type('BudgetStop', (RuntimeError,), {})
ERROR = 'Budget stop: $175.145 exposed, $0.158 next reservation, ceiling $175.0'


def failure(c, stage='revise_responses'):
    meta = mod.runtime.load_checkpoint(c.root / 'run_meta.json')
    for name, field in [('run.py', 'code_sha256'), ('per_row.py', 'code_per_row_sha256'), ('reviewer_probe.py', 'critic_validator_sha256')]:
        meta[field] = mod.byte_sha(Path(mod.__file__).with_name(name).read_bytes())
    mod.runtime.save_checkpoint(c.root / 'run_meta.json', meta)
    replies = client_for().replies
    replies[stage] = FOREIGN(ERROR)
    result = mod.per_row.generate_one(c.root, c.arm, c.candidate, RecordedClient(replies, c.budget))
    assert result['status'] == 'failed' and result['error_type'] == 'BudgetStop'
    if not (c.budget / 'spend.json').exists():
        mod.runtime.write_json(c.budget / 'spend.json', [])


@pytest.mark.parametrize('stage', ['scenario', 'revise_responses', 'grounding_0'])
def test_archive_only_terminal_then_resume_only_uncached_calls(ready, stage):
    c = ready
    failure(c, stage)
    before = {str(p): p.read_bytes() for p in c.root.rglob('*') if p.is_file()}
    report = mod.recover(c.root)
    assert len(report['recoverable']) == 1 and report['refused'] == []
    assert {str(p): p.read_bytes() for p in c.root.rglob('*') if p.is_file()} == before
    terminal = (c.out / 'result.json').read_bytes()
    receipt = (c.out / 'result.receipt.json').read_bytes()
    calls = json.loads((c.budget / 'spend.json').read_text(encoding='utf-8'))
    completed = {e['stage'] for e in calls}
    stages = {name: (c.out / (name + '.json')).read_bytes() for name in completed}
    mod.recover(c.root, apply=True)
    assert not (c.out / 'result.json').exists()
    assert (c.out / 'recovered_failures/budget_stop_0/result.json').read_bytes() == terminal
    assert (c.out / 'recovered_failures/budget_stop_0/result.receipt.json').read_bytes() == receipt
    assert all((c.out / (name + '.json')).read_bytes() == data for name, data in stages.items())
    client = client_for()
    result = mod.per_row.generate_one(c.root, c.arm, c.candidate, client)
    assert result['status'] == 'accepted'
    assert not completed.intersection(name for name, _ in client.calls)
    assert mod.recover(c.root)['recoverable'] == []


@pytest.mark.parametrize('change', ['unknown', 'uncertain', 'bound_exceeded', 'missing_checkpoint', 'bad_receipt', 'wrong_error', 'hard250', 'rejected', 'excluded', 'changed_identity'])
def test_unsafe_or_substantive_failures_refused(ready, change):
    c = ready
    failure(c)
    ledger = json.loads((c.budget / 'spend.json').read_text(encoding='utf-8'))
    result = mod.runtime.load_checkpoint(c.out / 'result.json')
    if change == 'unknown': ledger[0]['api_reported_cost_usd'] = None
    elif change in ('uncertain', 'bound_exceeded'): ledger[0]['status'] = change
    elif change == 'missing_checkpoint': (c.out / (ledger[0]['stage'] + '.json')).unlink()
    elif change == 'bad_receipt': (c.out / 'draft_responses.json').write_text('{}', encoding='utf-8')
    elif change == 'wrong_error': result['error'] = 'Invalid ledger amount; halt without dispatch'
    elif change == 'hard250': result['error'] = ERROR.replace('175.0', '250.0')
    elif change == 'rejected': result['status'] = 'rejected'
    elif change == 'excluded': mod.runtime.save_checkpoint(c.out / 'independent_exclusion.json', {'reason': 'A substantive defect.'})
    else: mod.runtime.save_checkpoint(c.out / 'identity.json', {'candidate_sha256': 'bad'})
    mod.runtime.write_json(c.budget / 'spend.json', ledger)
    mod.runtime.save_checkpoint(c.out / 'result.json', result)
    report = mod.recover(c.root)
    assert not report['recoverable']
    assert (c.out / 'result.json').exists()


def test_accounting_change_after_plan_refused(ready):
    c = ready
    failure(c)
    plan = mod.recover(c.root)['recoverable'][0]
    entries = json.loads((c.budget / 'spend.json').read_text(encoding='utf-8'))
    entries[0]['api_reported_cost_usd'] = .003
    mod.runtime.write_json(c.budget / 'spend.json', entries)
    with pytest.raises(ValueError, match='accounting changed'):
        mod.apply_plan(plan)
    assert (c.out / 'result.json').exists()


def test_imported_wrapper_preserves_one_exception_identity_and_prospective_receipt(ready, monkeypatch):
    c = ready
    observed = []
    def stopped(*args, **kwargs):
        observed.append((args, kwargs))
        raise mod.runtime.BudgetStop('Mock pre-dispatch stop')
    monkeypatch.setattr(execute_imported.runtime, 'execute', stopped)
    with pytest.raises(mod.per_row.base.BudgetStop):
        execute_imported.execute(c.root, 225, workers=3, arms=[c.arm], batch_limit=12)
    assert observed[0][0] == (c.root.resolve(), 'production', 225)
    files = [p for p in (c.root / 'phases').glob('imported_invocation_*.json') if not p.name.endswith('.receipt.json')]
    note = mod.runtime.load_checkpoint(files[0])
    assert note['budget_root'] == str(c.budget)
    assert note['ceiling'] == 225 and note['batch_limit'] == 12
    assert note['wrapper_sha256'] == mod.runtime.digest(Path(execute_imported.__file__).read_bytes())
    assert execute_imported.runtime is mod.per_row.base
