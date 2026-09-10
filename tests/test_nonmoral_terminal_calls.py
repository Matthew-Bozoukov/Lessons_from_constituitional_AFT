# ABOUTME: Verify terminal-call accounting retains exposure and rejects uncertain evidence.
# ABOUTME: Uses local synthetic ledgers; never contacts a provider.
import hashlib
import json

import pytest

from scratch.nonmoral.reconcile_terminal_calls import reconcile


def setup_files(tmp_path):
    phase = tmp_path / 'phase'
    phase.mkdir()
    (tmp_path / 'raw_calls').mkdir()
    messages = [{'role': 'user', 'content': 'A complete request'}]
    entries = [dict(status='reserved', charged_or_reserved_usd=0.25,
                    request_sha256=hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest())]
    (tmp_path / 'spend.json').write_text(json.dumps(entries))
    (phase / 'status.json').write_text(json.dumps(dict(status='awaiting_local_source_review', cumulative_calls=1)))
    raw = dict(status='terminal_exception', messages=messages, exception_type='TimeoutError')
    (tmp_path / 'raw_calls/00000.json').write_text(json.dumps(raw))
    return phase, raw


def test_retains_full_exposure_and_original_entry(tmp_path):
    phase, _ = setup_files(tmp_path)
    result = reconcile(tmp_path, phase)
    assert result == dict(retained_calls=1, exposure_usd=0.25)
    receipt = json.loads((phase / 'terminal_charge_receipt.json').read_text())
    assert receipt['entries'][0]['original_entry']['status'] == 'reserved'
    ledger = json.loads((tmp_path / 'spend.json').read_text())
    assert ledger[0]['status'] == 'retained_terminal_reservation'
    assert ledger[0]['charged_or_reserved_usd'] == 0.25


@pytest.mark.parametrize('defect', ['lock', 'uncertain', 'different_request', 'stale_phase'])
def test_refuses_mutation_when_terminal_proof_is_insufficient(tmp_path, defect):
    phase, raw = setup_files(tmp_path)
    if defect == 'lock':
        (tmp_path / 'generation.lock').write_text('123')
    elif defect == 'uncertain':
        raw['status'] = 'request_reserved'
    elif defect == 'different_request':
        raw['messages'][0]['content'] = 'Another request'
    else:
        (phase / 'status.json').write_text(json.dumps(dict(status='awaiting_local_source_review', cumulative_calls=0)))
    (tmp_path / 'raw_calls/00000.json').write_text(json.dumps(raw))
    before = (tmp_path / 'spend.json').read_bytes()
    with pytest.raises(ValueError):
        reconcile(tmp_path, phase)
    assert (tmp_path / 'spend.json').read_bytes() == before
    assert not (phase / 'terminal_charge_receipt.json').exists()
