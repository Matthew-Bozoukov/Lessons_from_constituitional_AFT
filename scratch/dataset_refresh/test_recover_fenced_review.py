# ABOUTME: Test strict lossless review extraction, scoped billing/request checks and no-network replay.
# ABOUTME: Exercise recovery rollback using temporary evidence only; never query models or modify run artifacts.
import json
from pathlib import Path

import pytest

from scratch.dataset_refresh import recover_fenced_review as recovery, run


def test_unique_terminal_fence_preserves_preamble_and_exact_json():
    content = 'Reasoned preamble.\n\n```json\n{ "accepted": true, "issues": [] }\n```\n'
    value, extraction = recovery.extract_final_json(content)
    assert value == {'accepted': True, 'issues': []}
    assert extraction['content'] == content
    assert extraction['preamble'] == 'Reasoned preamble.\n\n'
    a, b = extraction['json_span']
    assert content[a:b] == extraction['json_text']


@pytest.mark.parametrize('content', [
    '```json\n{}\n```\ntrailing explanation',
    '```json\n{}\n```\n```json\n{}\n```',
    'A quoted earlier {} verdict.\n```json\n{}\n```',
    '```python\n{}\n```',
    '```json\n{"accepted":true,"accepted":false}\n```',
    '```json\n{"x":NaN}\n```',
    '```json\n[]\n```',
    '```json\n{"accepted":true\n```',
])
def test_ambiguous_incomplete_or_nonjson_content_refused(content):
    with pytest.raises(ValueError):
        recovery.extract_final_json(content)


def call_fixture(tmp_path):
    cfg = {'models': {'review': {'model': 'anthropic/claude-sonnet-5', 'temperature': 0,
                                 'max_tokens': 4096}},
           'prompts': {'review_system': 'Judge {constitution}', 'review_user': '{conversation_json}'},
           'review_constitution_text': 'exact constitution', 'acceptance': {'required_true': ['accepted']}}
    record = {'system': 'Original system', 'user': 'Original user', 'reasoning': 'Original rationale',
              'response': 'Original final'}
    request = recovery.expected_request(record, cfg)
    call = {'call_id': 11275, 'stage': 'review_0', 'candidate_id': recovery.CANDIDATE,
            'arm': recovery.ARM, 'run_root': str(tmp_path), 'model': request['model'],
            'status': 'settled', 'request_sha256': run.digest(request)}
    raw = {'request': request, 'accounting': call.copy(),
           'response': {'finish_reason': 'stop', 'response_model': request['model'], 'tool_calls': [],
                        'content': 'All checks considered.\n```json\n{"accepted":true,"issues":[]}\n```'}}
    return cfg, record, call, raw


def test_exact_scoped_billed_request_validates(tmp_path):
    cfg, record, call, raw = call_fixture(tmp_path)
    value, _ = recovery.validate_call(raw, call, tmp_path, tmp_path, record, cfg)
    assert value['accepted'] is True


@pytest.mark.parametrize('field,value', [('stage', 'grounding_0'), ('candidate_id', 't2_008_v1'),
    ('arm', 'da-lowstakes-refresh'), ('call_id', 11276), ('model', 'other'), ('status', 'reserved')])
def test_wrong_scoped_call_refused(tmp_path, field, value):
    cfg, record, call, raw = call_fixture(tmp_path)
    call[field] = value
    with pytest.raises(ValueError, match='scoped/billed'):
        recovery.validate_call(raw, call, tmp_path, tmp_path, record, cfg)


def test_request_changed_after_authoring_refused(tmp_path):
    cfg, record, call, raw = call_fixture(tmp_path)
    record['response'] = 'A different final'
    with pytest.raises(ValueError, match='exactly match'):
        recovery.validate_call(raw, call, tmp_path, tmp_path, record, cfg)


def test_no_network_client_never_falls_back():
    client = recovery.NoNetworkClient()
    with pytest.raises(run.BudgetStop, match='No network'):
        client.chat(model='anything', messages=[])
    assert client.attempted_calls == 1


def replay_fixture(tmp_path):
    root = tmp_path / 'origin'
    arm = root / recovery.ARM
    row = arm / 'records' / recovery.CANDIDATE
    row.mkdir(parents=True)
    (arm / 'config.json').write_text('{}', encoding='utf-8')
    original = {'status': 'failed', 'record': {'user': 'do not edit'}}
    run.save_checkpoint(row / 'result.json', original)
    return root, row, original


def test_missing_cached_stage_fails_without_network(tmp_path, monkeypatch):
    root, row, original = replay_fixture(tmp_path)
    before = recovery.inventory(row)

    def missing_stage(root, arm, candidate, client):
        client.chat(messages=[], model='forbidden')

    monkeypatch.setattr(recovery.per_row, 'generate_one', missing_stage)
    with pytest.raises(run.BudgetStop, match='No network'):
        recovery.replay_cached(root, row, {}, {}, original, {'accepted': True})
    assert recovery.inventory(row) == before


def test_cached_replay_refuses_author_change(tmp_path, monkeypatch):
    root, row, original = replay_fixture(tmp_path)
    monkeypatch.setattr(recovery.per_row, 'generate_one', lambda *args: {
        'status': 'accepted', 'accepted_attempt': 0, 'record': {'user': 'changed'}})
    with pytest.raises(ValueError, match='changed original record'):
        recovery.replay_cached(root, row, {}, {}, original, {'accepted': True})


def transaction_fixture(tmp_path):
    root, row, original = replay_fixture(tmp_path)
    raw = tmp_path / 'raw.json'
    raw.write_text('{"untouched":"raw response"}', encoding='utf-8')
    evidence = tmp_path / 'evidence.json'
    evidence.write_text('{"untouched":"independent review"}', encoding='utf-8')
    cfg = {'raw_call_sha256': recovery.file_sha(raw), 'review_evidence_sha256': recovery.file_sha(evidence)}
    prepared = {'row': row, 'raw_path': raw, 'call': {'call_id': recovery.CALL_ID}, 'cfg': {},
                'bound_files': recovery.inventory(row), 'extraction': {}, 'review': {'accepted': True},
                'answer': {'user': 'do not edit'}, 'result': {'status': 'accepted', 'record': {'user': 'do not edit'}}}
    return row, prepared, cfg, evidence, tmp_path / 'output.json'


@pytest.mark.parametrize('failure_name', ['output.json', 'output.receipt.json'])
def test_final_receipt_failure_restores_failed_terminal(tmp_path, monkeypatch, failure_name):
    row, prepared, cfg, evidence, output = transaction_fixture(tmp_path)
    old = {name: (row / name).read_bytes() for name in prepared['bound_files']}
    monkeypatch.setattr(recovery, 'verify_history', lambda *args: True)
    write = run.write_json
    fired = False

    def fail_once(path, value):
        nonlocal fired
        if Path(path).name == failure_name and not fired:
            fired = True
            raise OSError('injected receipt failure')
        return write(path, value)

    monkeypatch.setattr(run, 'write_json', fail_once)
    with pytest.raises(OSError, match='injected'):
        recovery.apply_prepared(prepared, cfg, evidence, output)
    for name, value in old.items():
        assert (row / name).read_bytes() == value
    assert run.load_checkpoint(row / 'result.json')['status'] == 'failed'
    assert not (row / 'review_0.json').exists()
    assert not (row / recovery.HISTORY).exists()
    assert (row / recovery.ARCHIVE / 'result.json').read_bytes() == old['result.json']
    assert run.load_checkpoint(output)['rolled_back'] is True


def test_changed_independent_review_never_mutates_original(tmp_path):
    row, prepared, cfg, evidence, output = transaction_fixture(tmp_path)
    evidence.write_text('{}', encoding='utf-8')
    old = (row / 'result.json').read_bytes()
    with pytest.raises(ValueError, match='Changed bound file'):
        recovery.apply_prepared(prepared, cfg, evidence, output)
    assert (row / 'result.json').read_bytes() == old
    assert not (row / 'review_0.json').exists()


def history_fixture(tmp_path, monkeypatch):
    root, row, original = replay_fixture(tmp_path)
    cfg, record, call, raw = call_fixture(root)
    original = {'status': 'failed', 'error_type': 'JSONDecodeError', 'record': record}
    run.save_checkpoint(row / 'result.json', original)
    raw_path = tmp_path / 'raw_valid.json'
    run.write_json(raw_path, raw)
    evidence = tmp_path / 'review_valid.json'
    run.write_json(evidence, {'rows': [{'candidate_id': recovery.CANDIDATE,
        'result_path': str(row / 'result.json'), 'result_sha256': recovery.file_sha(row / 'result.json'),
        'decision': 'restore', 'review_scope': 'full_conversation', 'decision_owner': 'root',
        'reason': 'Read unchanged full conversation; approve only lossless review formatting recovery.'}]})
    review, extraction = recovery.validate_call(raw, call, tmp_path, root, record, cfg)
    settings = {'root': str(root), 'budget_root': str(tmp_path),
                'original_result_sha256': recovery.file_sha(row / 'result.json'),
                'raw_call_sha256': recovery.file_sha(raw_path), 'ledger_entry_sha256': run.digest(call),
                'review_evidence_path': str(evidence), 'review_evidence_sha256': recovery.file_sha(evidence)}
    result = {'status': 'accepted', 'accepted_attempt': 0, 'review': review,
              'record': dict(record, response_repair_count=0)}
    prepared = {'row': row, 'raw_path': raw_path, 'call': call, 'cfg': cfg,
                'bound_files': recovery.inventory(row), 'extraction': extraction, 'review': review,
                'answer': recovery.per_row.conversation(record), 'result': result}
    # Isolate the additional recovery history checks. Native accepted-stage receipts are tested by test_per_row.
    monkeypatch.setattr(recovery.per_row, 'verify_accepted', lambda *args: True)
    return row, prepared, settings, evidence, tmp_path / 'history_output.json'


def test_history_validates_lossless_origin_billing_and_evidence(tmp_path, monkeypatch):
    row, prepared, settings, evidence, output = history_fixture(tmp_path, monkeypatch)
    recovery.validate_independent_evidence(settings, prepared)
    report = recovery.apply_prepared(prepared, settings, evidence, output)
    assert report['applied'] and report['inference_calls'] == 0
    result = run.load_checkpoint(row / 'result.json')
    assert recovery.verify_history(row / 'result.json', result, prepared['cfg'])
    assert (row / recovery.ARCHIVE / 'result.json').read_bytes()
    assert result['record'] == prepared['result']['record']


@pytest.mark.parametrize('change', ['author_text', 'raw_call', 'ledger', 'independent_evidence', 'history'])
def test_history_refuses_tampered_recovery_chain(tmp_path, monkeypatch, change):
    row, prepared, settings, evidence, output = history_fixture(tmp_path, monkeypatch)
    recovery.apply_prepared(prepared, settings, evidence, output)
    result = run.load_checkpoint(row / 'result.json')
    if change == 'author_text':
        result['record']['response'] = 'Edited after recovery'
    elif change == 'raw_call':
        (row / recovery.ARCHIVE / 'raw_call_011275.json').write_text('{}', encoding='utf-8')
    elif change == 'ledger':
        run.save_checkpoint(row / recovery.ARCHIVE / 'ledger_call.json', {'call_id': 1})
    elif change == 'independent_evidence':
        (row / recovery.ARCHIVE / 'independent_review.json').write_text('{}', encoding='utf-8')
    else:
        history = run.load_checkpoint(row / recovery.HISTORY)
        history['inference_calls'] = 1
        run.save_checkpoint(row / recovery.HISTORY, history)
    with pytest.raises(ValueError):
        recovery.verify_history(row / 'result.json', result, prepared['cfg'])


@pytest.mark.parametrize('change', ['result_sha256', 'decision', 'review_scope', 'decision_owner', 'reason'])
def test_independent_approval_must_match_full_root_decision(tmp_path, monkeypatch, change):
    row, prepared, settings, evidence, output = history_fixture(tmp_path, monkeypatch)
    value = recovery.read(evidence)
    value['rows'][0][change] = ''
    run.write_json(evidence, value)
    settings['review_evidence_sha256'] = recovery.file_sha(evidence)
    with pytest.raises(ValueError, match='does not approve'):
        recovery.validate_independent_evidence(settings, prepared)


@pytest.mark.parametrize('name', ['review_0.receipt.json', 'answer_0.receipt.json',
    'fenced_review_recovery_11275.receipt.json', 'independent_exclusion.receipt.json'])
def test_orphan_receipts_cannot_be_overwritten_or_removed(tmp_path, name):
    original = b'{"orphan": "must remain unchanged"}'
    (tmp_path / name).write_bytes(original)
    with pytest.raises(ValueError, match='orphan receipt'):
        recovery.require_pristine_destination(tmp_path)
    assert (tmp_path / name).read_bytes() == original
