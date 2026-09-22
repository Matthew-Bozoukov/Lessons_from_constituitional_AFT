# ABOUTME: Offline tests for exact per-trait top-ups, partial-stage preference and preserved core exception handling.
# ABOUTME: No external clients or publication; all generation uses saved fixture responses.
from copy import deepcopy
from pathlib import Path

import pytest

from scratch.dataset_refresh import execute_targeted as mod
from scratch.dataset_refresh.test_recover_missing_changes import ready
from scratch.dataset_refresh.test_per_row import case, offline, client_for


@pytest.fixture
def target(ready):
    c = ready
    c.cfg['source']['unique_parent_ids'] = True
    mod.runtime.write_json(c.root / c.arm / 'config.json', c.cfg)
    c.candidates = []
    for i in range(3):
        candidate = deepcopy(c.candidate)
        candidate['candidate_id'] = f't1_{i:03d}_v0'
        candidate['source_id'] = f'parent_{i}'
        c.candidates.append(candidate)
    mod.runtime.write_rows(c.root / c.arm / 'candidates.jsonl', c.candidates)
    meta = mod.runtime.load_checkpoint(c.root / 'run_meta.json')
    meta['arms'][c.arm].update(config_sha256=mod.runtime.digest(c.cfg),
                             candidates_sha256=mod.runtime.digest((c.root / c.arm / 'candidates.jsonl').read_bytes()))
    for name, field in [('run.py', 'code_sha256'), ('per_row.py', 'code_per_row_sha256'), ('reviewer_probe.py', 'critic_validator_sha256')]:
        meta[field] = mod.runtime.digest(Path(mod.__file__).with_name(name).read_bytes())
    mod.runtime.save_checkpoint(c.root / 'run_meta.json', meta)
    c.plan = {'root': str(c.root), 'arm': c.arm, 'ceiling': 225, 'workers': 1,
              'candidate_counts': {t: int(t == 't1') for t in mod.runtime.quotas()}}
    c.row = lambda i: c.root / c.arm / 'records' / c.candidates[i]['candidate_id']
    return c


def partial(c, i, names):
    candidate = c.candidates[i]
    mod.runtime.save_checkpoint(c.row(i) / 'identity.json', {'candidate_sha256': mod.runtime.digest(candidate),
                                                           'config_sha256': mod.runtime.digest(c.cfg)})
    for name in names:
        mod.runtime.save_checkpoint(c.row(i) / (name + '.json'), {'fixture_stage': name})


def test_exact_counts_prioritize_most_paid_stages_before_fresh_candidates(target):
    c = target
    partial(c, 1, ['scenario'])
    partial(c, 2, ['scenario', 'preflight', 'draft_responses'])
    before = {str(p): p.read_bytes() for p in c.root.rglob('*') if p.is_file()}
    phase, cfg, chosen = mod.resolve(c.plan)
    assert [x['candidate_id'] for x in chosen] == ['t1_002_v0']
    assert phase['candidate_counts'] == c.plan['candidate_counts']
    assert len(phase['candidates'][0]['saved_stage_sha256']) == 3
    assert {str(p): p.read_bytes() for p in c.root.rglob('*') if p.is_file()} == before


def test_never_reopens_failed_or_excluded_candidates(target):
    c = target
    mod.runtime.save_checkpoint(c.row(0) / 'result.json', {'status': 'failed', 'trait_id': 't1', 'candidate_id': 't1_000_v0'})
    mod.runtime.save_checkpoint(c.row(1) / 'independent_exclusion.json', {'reason': 'Do not reopen.'})
    assert mod.resolve(c.plan)[2] == [c.candidates[2]]
    c.plan.pop('candidate_counts')
    c.plan['candidate_ids'] = ['t1_000_v0']
    with pytest.raises(ValueError, match='completed, excluded'):
        mod.resolve(c.plan)


@pytest.mark.parametrize('mutation', ['wrong_core', 'changed_receipt', 'shortfall', 'over250', 'two_modes'])
def test_unsafe_targeted_plan_refused(target, mutation):
    c = target
    if mutation == 'wrong_core':
        meta = mod.runtime.load_checkpoint(c.root / 'run_meta.json')
        meta['code_sha256'] = 'bad'
        mod.runtime.save_checkpoint(c.root / 'run_meta.json', meta)
    elif mutation == 'changed_receipt':
        partial(c, 2, ['scenario'])
        (c.row(2) / 'scenario.json').write_text('{}', encoding='utf-8')
    elif mutation == 'shortfall': c.plan['candidate_counts']['t2'] = 1
    elif mutation == 'over250': c.plan['ceiling'] = 251
    else: c.plan['candidate_ids'] = ['t1_000_v0']
    with pytest.raises((ValueError, mod.runtime.BudgetStop)):
        mod.resolve(c.plan)


def test_targeted_execution_uses_actual_imported_pipeline_and_freezes_exact_jobs(target, monkeypatch):
    c = target
    client = client_for()
    monkeypatch.setattr(mod.runtime, 'BudgetClient', lambda *a, **kw: client)
    monkeypatch.setattr(mod.runtime, 'status', lambda root: {'fixture': 'status collected'})
    result = mod.execute(c.plan)
    assert result['jobs'] == 1 and result['completed'][0]['status'] == 'accepted'
    assert not result['stopped_before_next_dispatch']
    phase = mod.runtime.load_checkpoint(c.root / 'phases' / result['phase_file'])
    assert phase['candidate_counts'] == c.plan['candidate_counts']
    assert phase['candidates'][0]['candidate_sha256'] == mod.runtime.digest(c.candidates[0])
    assert phase['wrapper_sha256'] == mod.runtime.digest(Path(mod.__file__).read_bytes())
    assert phase['budget_root'] == str(c.budget)
    assert len(list((c.root / c.arm / 'records').glob('*/result.json'))) == 1


def test_imported_budgetstop_creates_no_failed_terminal_and_stops_following_jobs(target, monkeypatch):
    c = target
    c.plan['candidate_counts']['t1'] = 3
    client = client_for()
    client.replies['revise_responses'] = mod.runtime.BudgetStop('Mock pre-dispatch ceiling reached')
    monkeypatch.setattr(mod.runtime, 'BudgetClient', lambda *a, **kw: client)
    monkeypatch.setattr(mod.runtime, 'status', lambda root: {'fixture': 'status collected'})
    result = mod.execute(c.plan)
    assert result['stopped_before_next_dispatch'] and result['completed'] == []
    assert not list((c.root / c.arm / 'records').glob('*/result.json'))
    assert len([stage for stage, _ in client.calls if stage == 'scenario']) == 1
    assert (c.row(0) / 'draft_responses.json').exists()
