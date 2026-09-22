# ABOUTME: Offline adversarial tests for audit-tag-only recovery and unchanged native/external resume.
import json
from pathlib import Path

import pytest

from scratch.dataset_refresh import recover_missing_changes as mod
from scratch.dataset_refresh import repair_independent
from scratch.dataset_refresh.test_per_row import case, offline, FakeClient, client_for, tagged, clean_verdict, GOOD, BAD


class RecordedClient(FakeClient):
    def __init__(self, replies, budget):
        super().__init__(replies)
        self.budget = budget

    def chat(self, **request):
        response = super().chat(**request)
        path = self.budget / 'spend.json'
        entries = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
        entry = {'call_id': len(entries), 'run_root': self.local.run_root, 'arm': self.local.arm,
                 'candidate_id': self.local.candidate_id, 'stage': self.local.stage,
                 'model': request['model'], 'status': 'settled', 'api_reported_cost_usd': .001,
                 'request_sha256': mod.runtime.digest(request)}
        entries.append(entry)
        mod.runtime.write_json(path, entries)
        mod.runtime.write_json(self.budget / 'raw_calls' / f'{entry["call_id"]:06d}.json',
            {'request': request, 'accounting': entry, 'response': {'content': response.content, 'finish_reason': 'stop'}})
        return response


@pytest.fixture
def ready(case):
    constitution = case.root / 'constitution.md'
    constitution.write_text('Preserve actual requirements.', encoding='utf-8')
    case.cfg.update(constitution=str(constitution), constitution_sha256=mod.runtime.digest(constitution.read_bytes()))
    mod.runtime.write_json(case.root / case.arm / 'config.json', case.cfg)
    mod.runtime.write_rows(case.root / case.arm / 'source.jsonl', [case.candidate['source']])
    meta = {'budget_root': str(case.root / 'budget'), 'arms': {case.arm: {
        'config_sha256': mod.runtime.digest(case.cfg),
        'source_snapshot_sha256': mod.runtime.digest((case.root / case.arm / 'source.jsonl').read_bytes()),
        'candidates_sha256': mod.runtime.digest((case.root / case.arm / 'candidates.jsonl').read_bytes())}}}
    mod.runtime.save_checkpoint(case.root / 'run_meta.json', meta)
    case.budget = case.root / 'budget'
    return case


def native_failure(c, repair=False):
    replies = client_for(response=BAD if repair else GOOD, repair=repair).replies
    stage = 'repair_1' if repair else 'revise_responses'
    replies[stage] = tagged('Keep exactly this detailed reasoning.', GOOD)
    client = RecordedClient(replies, c.budget)
    result = mod.per_row.generate_one(c.root, c.arm, c.candidate, client)
    assert result['status'] == 'failed' and result['error'] == mod.ERROR
    return stage


@pytest.mark.parametrize('repair', [False, True])
def test_native_format_recovery_preserves_bytes_and_resume_calls_only_new_critics(ready, repair):
    c = ready
    stage = native_failure(c, repair)
    before = {str(p): p.read_bytes() for p in c.root.rglob('*') if p.is_file()}
    report = mod.recover(c.root)
    assert len(report['recoverable']) == 1 and not report['refused']
    assert {str(p): p.read_bytes() for p in c.root.rglob('*') if p.is_file()} == before
    original = (c.out / 'result.json').read_bytes()
    original_receipt = (c.out / 'result.receipt.json').read_bytes()
    mod.recover(c.root, apply=True)
    saved = mod.runtime.load_checkpoint(c.out / (stage + '.json'))
    assert saved == {'reasoning': 'Keep exactly this detailed reasoning.', 'response': GOOD, 'rewrite_changes': ''}
    archive = c.out / 'recovered_failures' / ('missing_changes_' + stage + '_0')
    assert (archive / 'result.json').read_bytes() == original
    assert (archive / 'result.receipt.json').read_bytes() == original_receipt
    n = 1 if repair else 0
    reviewer = FakeClient({f'grounding_{n}': clean_verdict(), f'review_{n}': {'accepted': True, 'factual_reasoning': True, 'stakes': 1}})
    result = mod.per_row.generate_one(c.root, c.arm, c.candidate, reviewer)
    assert result['status'] == 'accepted'
    assert [name for name, _ in reviewer.calls] == [f'grounding_{n}', f'review_{n}']
    assert result['record']['reasoning'] == saved['reasoning'] and result['record']['response'] == saved['response']
    assert mod.recover(c.root)['recoverable'] == []


def test_external_failed_candidate_recovery_keeps_original_excluded_and_resume_reuses_author(ready, monkeypatch):
    c = ready
    mod.per_row.generate_one(c.root, c.arm, c.candidate, RecordedClient(client_for(response=BAD).replies, c.budget))
    original = (c.out / 'result.json').read_bytes()
    mod.runtime.save_checkpoint(c.out / 'independent_exclusion.json', {'result_sha256': mod.runtime.digest(original), 'reason': 'Wrong authorization.'})
    client = RecordedClient({'independent_rewrite_100': tagged('Correct the actual premise.', GOOD)}, c.budget)
    monkeypatch.setattr(mod.runtime, 'BudgetClient', lambda *a, **k: client)
    result = repair_independent.propose(c.root, c.arm, c.candidate['candidate_id'], 250)
    assert result['error'] == mod.ERROR
    failed = (c.out / 'independent_candidate_100.json').read_bytes()
    mod.recover(c.root, apply=True)
    assert (c.out / 'result.json').read_bytes() == original
    assert mod.runtime.load_result(c.out / 'result.json')['status'] == 'rejected'
    assert (c.out / 'recovered_failures/missing_changes_independent_rewrite_100_0/independent_candidate_100.json').read_bytes() == failed
    reviewer = FakeClient({'grounding_100': clean_verdict(), 'review_100': {'accepted': True, 'factual_reasoning': True, 'stakes': 1}})
    monkeypatch.setattr(mod.runtime, 'BudgetClient', lambda *a, **k: reviewer)
    repaired = repair_independent.propose(c.root, c.arm, c.candidate['candidate_id'], 250)
    assert repaired['status'] == 'accepted'
    assert [stage for stage, _ in reviewer.calls] == ['grounding_100', 'review_100']
    assert mod.runtime.load_result(c.out / 'result.json')['status'] == 'rejected'


@pytest.mark.parametrize('raw', [
    '<reasoning>One.</reasoning><response>Two.',
    '<reasoning>One.<response>Two.</response>',
    '<reasoning>One.</reasoning><response></response>',
    '<reasoning>One.</reasoning><response>Two.</response><changes>',
    '<reasoning>One.</reasoning><response>Two.</response><changes/>',
    '<reasoning>One.</reasoning><response>Two.</response><response>Three.</response>',
    'Preface <reasoning>One.</reasoning><response>Two.</response>',
    '<reasoning>Nested <response>thing</response></reasoning><response>Two.</response>',
])
def test_partial_empty_duplicate_or_extraneous_output_refused(raw):
    with pytest.raises(ValueError):
        mod.parse_complete_answer(raw)


def test_text_escapes_unicode_and_reverse_order_preserved():
    raw = '<response>Use "quotes", \\n, and café.</response>\n<reasoning>Line1\nLine2.</reasoning>'
    assert mod.parse_complete_answer(raw) == {'response': 'Use "quotes", \\n, and café.', 'reasoning': 'Line1\nLine2.'}


@pytest.mark.parametrize('mutation', ['truncated', 'uncertain', 'unknown_cost', 'duplicate', 'wrong_error', 'rejected', 'changed_raw', 'changed_identity'])
def test_provenance_and_substantive_failures_are_never_format_recovered(ready, mutation):
    c = ready
    native_failure(c)
    entries = json.loads((c.budget / 'spend.json').read_text(encoding='utf-8'))
    rawpath = c.budget / 'raw_calls' / f'{entries[-1]["call_id"]:06d}.json'
    raw = json.loads(rawpath.read_text(encoding='utf-8'))
    if mutation == 'truncated':
        raw['response']['finish_reason'] = 'length'
    elif mutation in ('uncertain', 'unknown_cost'):
        entries[-1]['status' if mutation == 'uncertain' else 'api_reported_cost_usd'] = 'uncertain_failure' if mutation == 'uncertain' else None
        raw['accounting'] = entries[-1]
    elif mutation == 'duplicate':
        entries.append({**entries[-1], 'call_id': len(entries)})
    elif mutation in ('wrong_error', 'rejected'):
        result = mod.runtime.load_checkpoint(c.out / 'result.json')
        result['error' if mutation == 'wrong_error' else 'status'] = 'lint failure' if mutation == 'wrong_error' else 'rejected'
        mod.runtime.save_checkpoint(c.out / 'result.json', result)
    elif mutation == 'changed_raw':
        raw['request']['messages'][0]['content'] = 'Tampered'
    else:
        mod.runtime.save_checkpoint(c.out / 'identity.json', {'config_sha256': 'changed'})
    mod.runtime.write_json(c.budget / 'spend.json', entries)
    mod.runtime.write_json(rawpath, raw)
    assert not mod.recover(c.root)['recoverable']


def test_recovery_refuses_active_production_or_changed_evidence(ready):
    native_failure(ready)
    plan = mod.recover(ready.root)['recoverable'][0]
    with mod.FileLock(str(ready.root / 'execution.lock')):
        with pytest.raises(Exception, match='could not be acquired'):
            mod.recover(ready.root, apply=True)
    mod.runtime.save_checkpoint(ready.out / 'preflight.json', {'eligible': False, 'stakes': 1})
    with pytest.raises(ValueError, match='changed since planning'):
        mod.apply_plan(plan)
