# ABOUTME: Offline evidence-preservation and anti-judge-shopping tests for technical review retries.
import json
from pathlib import Path

import pytest

from scratch.dataset_refresh import retry_technical_review as mod


def fixture(tmp_path, parse_error=False, finish='stop'):
    root, budget = tmp_path / 'run', tmp_path / 'budget'
    row = root / 'nonmoral-advice' / 'records' / 't4_001_v0'
    row.mkdir(parents=True)
    cfg = {'models': {'grounding': {'model': 'anthropic/claude-sonnet-5', 'temperature': 0, 'max_tokens': 6000}},
           'grounding_review': {'model': 'grounding', 'prompts': {'system': 'Check concrete claims.', 'user': '{conversation_json}'}},
           'preflight': {'acceptance': {'required_true': ['accepted']}},
           'response_stages': [{'name': 'draft'}, {'name': 'rewrite'}], 'max_response_repairs': 1}
    mod.runtime.write_json(root / 'nonmoral-advice' / 'config.json', cfg)
    mod.runtime.save_checkpoint(root / 'run_meta.json', {'budget_root': str(budget),
        'critic_validator_sha256': mod.runtime.digest(Path(mod.__file__).with_name('reviewer_probe.py').read_bytes())})
    mod.runtime.save_checkpoint(row / 'identity.json', {'config_sha256': mod.runtime.digest(cfg)})
    scenario = {'system': 'Offer advice.', 'user': 'Keep the meeting on Monday.'}
    answer = {'reasoning': 'Monday matters.', 'response': 'Meet on Monday.'}
    for name, value in [('scenario', scenario), ('preflight', {'accepted': True}),
                        ('draft', {'draft_reasoning': 'Draft.', 'draft_response': 'Draft.'}), ('rewrite', answer)]:
        mod.runtime.save_checkpoint(row / (name + '.json'), value)
    narrow = {**scenario, 'reasoning': answer['reasoning'], 'final': answer['response']}
    verdict = {'accepted': False, 'assessment': 'A timing problem.', 'issues': [{
        'kind': 'constraint_violation', 'answer_field': 'final', 'answer_quote': 'Meet on Tuesday.',
        'premise_field': 'user', 'premise_quote': 'Keep the meeting on Monday.',
        'material_consequence': 'Changes the date.', 'alternative_reading': 'No waiver given.'}]}
    text = '{"accepted":' if parse_error else json.dumps(verdict)
    try:
        mod.validate_verdict(mod.runtime._parse_json(text), narrow)
    except ValueError as exc:
        error_type, error = type(exc).__name__, str(exc)
    request = {**cfg['models']['grounding'], 'messages': [
        {'role': 'system', 'content': cfg['grounding_review']['prompts']['system']},
        {'role': 'user', 'content': json.dumps(narrow, ensure_ascii=False)}]}
    entry = {'call_id': 0, 'run_root': str(root.resolve()), 'arm': 'nonmoral-advice',
             'candidate_id': row.name, 'stage': 'grounding_0', 'status': 'settled',
             'request_sha256': mod.runtime.digest(request), 'api_reported_cost_usd': .03}
    raw = {'accounting': entry, 'request': request, 'response': {'finish_reason': finish, 'content': text}}
    mod.runtime.write_json(budget / 'raw_calls' / '000000.json', raw)
    mod.runtime.write_json(budget / 'spend.json', [entry])
    if not parse_error:
        mod.runtime.save_checkpoint(row / 'grounding_0.json', verdict)
    mod.runtime.save_checkpoint(row / 'result.json', {'candidate_id': row.name, 'status': 'failed',
        'record': {**scenario, **answer}, 'error_type': error_type, 'error': error})
    return root, row, budget


def snapshot(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob('*') if p.is_file()}


def test_dry_run_and_apply_archive_exact_evidence_without_answer_edits(tmp_path):
    root, row, _ = fixture(tmp_path)
    before = snapshot(tmp_path)
    result = mod.recover(root)
    assert len(result['retryable']) == 1 and not result['refused']
    assert snapshot(tmp_path) == before
    preserved = {name: (row / name).read_bytes() for name in
                 ('grounding_0.json', 'grounding_0.receipt.json', 'result.json', 'result.receipt.json')}
    mod.recover(root, apply=True)
    archive = row / 'recovered_failures' / 'technical_retry_0' / 'grounding_0'
    for name, data in preserved.items():
        assert (archive / name).read_bytes() == data
        assert not (row / name).exists()
    for name in ('scenario', 'preflight', 'draft', 'rewrite', 'identity'):
        assert (row / (name + '.json')).read_bytes() == before[str((row / (name + '.json')).relative_to(tmp_path))]
    assert mod.runtime.load_checkpoint(row / 'technical_retry_grounding_0.json')['call_id'] == 0


@pytest.mark.parametrize('finish,allow,expected', [('stop', False, True), ('length', False, False), ('length', True, True), ('content_filter', True, False)])
def test_missing_parse_checkpoint_and_explicit_truncation(tmp_path, finish, allow, expected):
    root, row, _ = fixture(tmp_path, parse_error=True, finish=finish)
    report = mod.recover(root, allow_truncated=allow)
    assert bool(report['retryable']) == expected
    if expected:
        mod.recover(root, apply=True, allow_truncated=allow)
        assert not (row / 'grounding_0.json').exists()
        assert (row / 'recovered_failures' / 'technical_retry_0' / 'grounding_0' / 'result.json').exists()


@pytest.mark.parametrize('mutation', ['rejected', 'accepted', 'unknown', 'missing_cost', 'duplicate_call',
    'later_call', 'changed_request', 'changed_answer', 'changed_checkpoint', 'different_error',
    'review_exists', 'already_claimed', 'excluded', 'changed_validator', 'bad_preflight'])
def test_fail_closed_provenance_and_no_second_retry(tmp_path, mutation):
    root, row, budget = fixture(tmp_path)
    ledger = json.loads((budget / 'spend.json').read_text())
    rawpath = budget / 'raw_calls' / '000000.json'
    raw = json.loads(rawpath.read_text())
    result = mod.runtime.load_checkpoint(row / 'result.json')
    if mutation in ('rejected', 'accepted'):
        result['status'] = mutation
    elif mutation in ('unknown', 'missing_cost'):
        ledger[0]['status' if mutation == 'unknown' else 'api_reported_cost_usd'] = 'uncertain' if mutation == 'unknown' else None
        raw['accounting'] = ledger[0]
    elif mutation in ('duplicate_call', 'later_call'):
        ledger.append({**ledger[0], 'call_id': 1, 'stage': 'review_0' if mutation == 'later_call' else 'grounding_0'})
    elif mutation == 'changed_request':
        raw['request']['messages'][1]['content'] = 'Different answer.'
        ledger[0]['request_sha256'] = mod.runtime.digest(raw['request'])
        raw['accounting'] = ledger[0]
    elif mutation == 'changed_answer':
        result['record']['response'] = 'Different answer.'
    elif mutation == 'changed_checkpoint':
        mod.runtime.save_checkpoint(row / 'grounding_0.json', {'accepted': True})
    elif mutation == 'different_error':
        result['error'] = 'Not the same error.'
    elif mutation == 'review_exists':
        mod.runtime.save_checkpoint(row / 'review_0.json', {'accepted': True})
    elif mutation == 'already_claimed':
        mod.runtime.save_checkpoint(row / 'technical_retry_grounding_0.json', {})
    elif mutation == 'excluded':
        mod.runtime.save_checkpoint(row / 'independent_exclusion.json', {'reason': 'Content failed.'})
    elif mutation == 'changed_validator':
        meta = mod.runtime.load_checkpoint(root / 'run_meta.json')
        meta['critic_validator_sha256'] = 'different'
        mod.runtime.save_checkpoint(root / 'run_meta.json', meta)
    else:
        mod.runtime.save_checkpoint(row / 'preflight.json', {'accepted': False})
    mod.runtime.save_checkpoint(row / 'result.json', result)
    mod.runtime.write_json(budget / 'spend.json', ledger)
    mod.runtime.write_json(rawpath, raw)
    before = snapshot(tmp_path)
    assert not mod.recover(root)['retryable']
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize('accepted', [True, False])
def test_valid_verdict_is_never_retried_even_if_terminal_claims_schema_failure(tmp_path, accepted):
    root, row, budget = fixture(tmp_path)
    rawpath = budget / 'raw_calls' / '000000.json'
    raw = json.loads(rawpath.read_text())
    verdict = json.loads(raw['response']['content'])
    verdict['accepted'] = accepted
    if accepted:
        verdict['issues'] = []
    else:
        verdict['issues'][0]['answer_quote'] = 'Meet on Monday.'
    raw['response']['content'] = json.dumps(verdict)
    mod.runtime.write_json(rawpath, raw)
    mod.runtime.save_checkpoint(row / 'grounding_0.json', verdict)
    report = mod.recover(root)
    assert not report['retryable']
    assert 'Valid verdict' in report['refused'][0]['reason']


def test_changed_evidence_between_plan_apply_and_active_lock_refused(tmp_path):
    root, row, _ = fixture(tmp_path)
    plan = mod.recover(root)['retryable'][0]
    with mod.FileLock(str(root / 'execution.lock')):
        with pytest.raises(Exception, match='could not be acquired'):
            mod.recover(root, apply=True)
    mod.runtime.save_checkpoint(row / 'rewrite.json', {'reasoning': 'Changed.', 'response': 'Changed.'})
    with pytest.raises(ValueError, match='changed after planning'):
        mod.apply_plan(plan)
    assert (row / 'result.json').exists()


def test_known_length_failure_before_parser_requires_flag_and_never_retries_valid_adverse_json(tmp_path):
    root, row, budget = fixture(tmp_path, parse_error=True, finish='length')
    result = mod.runtime.load_checkpoint(row / 'result.json')
    result.update(error_type='ValueError', error='Excluded incomplete/provider-filtered output: length')
    mod.runtime.save_checkpoint(row / 'result.json', result)
    assert not mod.recover(root)['retryable']
    assert len(mod.recover(root, allow_truncated=True)['retryable']) == 1
    rawpath = budget / 'raw_calls' / '000000.json'
    raw = json.loads(rawpath.read_text())
    raw['response']['content'] = json.dumps({'accepted': False, 'assessment': 'A concern.', 'issues': [{
        'kind': 'constraint_violation', 'answer_field': 'final', 'answer_quote': 'Meet on Monday.',
        'premise_field': 'user', 'premise_quote': 'Keep the meeting on Monday.',
        'material_consequence': 'Concern.', 'alternative_reading': 'No waiver.'}]})
    mod.runtime.write_json(rawpath, raw)
    report = mod.recover(root, allow_truncated=True)
    assert not report['retryable'] and 'Valid verdict' in report['refused'][0]['reason']


def test_second_response_attempt_preserves_prior_adverse_review_and_repair(tmp_path):
    root, row, budget = fixture(tmp_path)
    rawpath = budget / 'raw_calls' / '000000.json'
    raw = json.loads(rawpath.read_text())
    malformed = json.loads(raw['response']['content'])
    prior = json.loads(raw['response']['content'])
    prior['issues'][0]['answer_quote'] = 'Meet on Monday.'
    mod.runtime.save_checkpoint(row / 'grounding_0.json', prior)
    mod.runtime.save_checkpoint(row / 'review_0.json', {'accepted': False})
    actual = mod.runtime.load_checkpoint(row / 'result.json')['record']
    mod.runtime.save_checkpoint(row / 'answer_0.json', actual)
    mod.runtime.save_checkpoint(row / 'repair_1.json', {k: actual[k] for k in ('reasoning', 'response')})
    mod.runtime.save_checkpoint(row / 'grounding_1.json', malformed)
    raw['accounting'].update(call_id=1, stage='grounding_1')
    mod.runtime.write_json(budget / 'raw_calls' / '000001.json', raw)
    entries = json.loads((budget / 'spend.json').read_text()) + [raw['accounting']]
    mod.runtime.write_json(budget / 'spend.json', entries)
    report = mod.recover(root)
    assert len(report['retryable']) == 1 and report['retryable'][0]['stage'] == 'grounding_1'
    mod.recover(root, apply=True)
    assert mod.runtime.load_checkpoint(row / 'grounding_0.json') == prior
    assert mod.runtime.load_checkpoint(row / 'review_0.json') == {'accepted': False}
    assert mod.runtime.load_checkpoint(row / 'repair_1.json') == {k: actual[k] for k in ('reasoning', 'response')}
