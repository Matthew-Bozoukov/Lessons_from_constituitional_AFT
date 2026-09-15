# ABOUTME: Offline adversarial tests for lossless scenario format repair and provenance-bound application.
import json
from pathlib import Path

import pytest

from scratch.dataset_refresh import recover_scenario_json as mod


def test_repair_preserves_escapes_literal_quotes_controls_unicode_and_reverse_order():
    system = r'Existing \"quote\"; newline\n; slash\\; unicode\u263a'
    user = 'I said "hello".\nNext\tline: café.'
    raw = '{"user":"' + user + '","system":"' + system + '"}'
    result, changes = mod.repair_two_string_json(raw)
    assert result == {'user': user, 'system': json.loads('"' + system + '"')}
    assert changes['fields']['user']['escaped_literal_quotes'] == 2
    assert changes['fields']['user']['escaped_literal_controls'] == 2
    assert json.loads(json.dumps(result)) == result


def test_exact_outer_fence_only():
    result, changes = mod.repair_two_string_json('```json\n{"system":"a", "user":"say "yes""}\n```')
    assert result['user'] == 'say "yes"'
    assert changes['removed_outer_markdown_fence'] is True


@pytest.mark.parametrize('raw', [
    '{"system":"a","user":"b","extra":"c"}',
    '{"system":"a","user":"b" "extra":"c"}',
    '{"system":"a","user":"b",extra:42}',
    '{"system":"a","user":"b","user":"c"}',
    '{"system":"a", "user":"inside", "user":"actual"}',
    '{"system":"a","user":true}',
    '{"system":"a","user":"bad\\q"}',
    '{"system":"a","user":"bad\\u12xx"}',
    '{"system":"a","user":"bad\\ud800"}',
    '{"system":"a","user":"b",}',
    'Prose {"system":"a","user":"b"}',
    '```json\n{"system":"a","user":"b"}\n```\nMore prose',
    '{"system":"a","user":""}',
])
def test_ambiguous_non_schema_or_invalid_escape_rejected(raw):
    with pytest.raises((ValueError, UnicodeError)):
        mod.repair_two_string_json(raw)


def fixture(tmp_path):
    root = tmp_path / 'run'
    row = root / 'nonmoral-advice' / 'records' / 't3_000_v0'
    row.mkdir(parents=True)
    budget = tmp_path / 'budget'
    cfg = {'prompts': {'scenario_user': mod.SCHEMA_INSTRUCTION}}
    mod.runtime.write_json(root / 'nonmoral-advice' / 'config.json', cfg)
    mod.runtime.write_json(root / 'run_meta.json', {'budget_root': str(budget)})
    mod.runtime.save_checkpoint(row / 'identity.json', {'config_sha256': mod.runtime.digest(cfg)})
    text = '{"system":"Be helpful.", "user":"Please discuss "this" choice.\nIt matters."}'
    try:
        mod.runtime._parse_json(text)
    except json.JSONDecodeError as exc:
        error = str(exc)
    request = {'model': 'test/no-call', 'messages': [{'role': 'user', 'content': mod.SCHEMA_INSTRUCTION}]}
    entry = {'call_id': 0, 'run_root': str(root.resolve()), 'arm': 'nonmoral-advice',
             'candidate_id': row.name, 'stage': 'scenario', 'status': 'settled',
             'request_sha256': mod.runtime.digest(request)}
    raw = {'request': request, 'accounting': entry, 'response': {'content': text, 'finish_reason': 'stop'}}
    mod.runtime.write_json(budget / 'raw_calls' / '000000.json', raw)
    mod.runtime.write_json(budget / 'spend.json', [entry])
    mod.runtime.save_checkpoint(row / 'result.json', {'candidate_id': row.name, 'status': 'failed', 'error_type': 'JSONDecodeError', 'error': error})
    return root, row, budget


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_dry_run_has_no_mutations_and_apply_preserves_failed_bytes_and_receipt(tmp_path):
    root, row, budget = fixture(tmp_path)
    before = snapshot(tmp_path)
    report = mod.recover(root)
    assert len(report['recoverable']) == 1 and not report['refused']
    assert snapshot(tmp_path) == before
    terminal, receipt = (row / 'result.json').read_bytes(), (row / 'result.receipt.json').read_bytes()
    mod.recover(root, apply=True)
    assert not (row / 'result.json').exists()
    archive = row / 'recovered_failures' / 'scenario_json_0'
    assert (archive / 'result.json').read_bytes() == terminal
    assert (archive / 'result.receipt.json').read_bytes() == receipt
    assert mod.runtime.load_checkpoint(row / 'scenario.json') == report['recoverable'][0]['payload']
    recovery = mod.runtime.load_checkpoint(row / 'scenario_recovery.json')
    assert recovery['raw_call_sha256'] == mod.byte_sha((budget / 'raw_calls' / '000000.json').read_bytes())
    assert mod.recover(root, apply=True)['recoverable'] == []


@pytest.mark.parametrize('mutation', ['duplicate_call', 'wrong_stage', 'raw_request', 'truncated', 'different_error', 'downstream_checkpoint', 'quality_reject'])
def test_provenance_or_quality_failures_cannot_be_recovered(tmp_path, mutation):
    root, row, budget = fixture(tmp_path)
    ledger = json.loads((budget / 'spend.json').read_text())
    rawpath = budget / 'raw_calls' / '000000.json'
    raw = json.loads(rawpath.read_text())
    if mutation == 'duplicate_call':
        ledger.append(dict(ledger[0]))
    elif mutation == 'wrong_stage':
        ledger[0]['stage'] = 'answer_0'
    elif mutation == 'raw_request':
        raw['request']['model'] = 'tampered'
    elif mutation == 'truncated':
        raw['response']['finish_reason'] = 'length'
    elif mutation in ('different_error', 'quality_reject'):
        result = mod.runtime.load_checkpoint(row / 'result.json')
        result['error' if mutation == 'different_error' else 'status'] = 'different' if mutation == 'different_error' else 'rejected'
        mod.runtime.save_checkpoint(row / 'result.json', result)
    else:
        mod.runtime.save_checkpoint(row / 'preflight.json', {'eligible': False})
    mod.runtime.write_json(budget / 'spend.json', ledger)
    mod.runtime.write_json(rawpath, raw)
    before = snapshot(tmp_path)
    report = mod.recover(root)
    assert not report['recoverable']
    assert snapshot(tmp_path) == before


def test_move_target_cannot_escape_row(tmp_path):
    with pytest.raises(ValueError, match='escapes'):
        mod.within(tmp_path / 'elsewhere' / 'result.json', tmp_path / 'row')


def test_apply_refuses_active_production_execution(tmp_path):
    root, _, _ = fixture(tmp_path)
    with mod.FileLock(str(root / 'execution.lock')):
        with pytest.raises(Exception, match='could not be acquired'):
            mod.recover(root, apply=True)
