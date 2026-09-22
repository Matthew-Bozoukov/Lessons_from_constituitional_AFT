# ABOUTME: Preserve malformed-provider diagnostics and budget reservations without accepting verdicts.
# ABOUTME: Repeated identical failures need distinct diagnostic files and cumulative charges.
import json
from types import SimpleNamespace

import pytest
from src.eval.misalignment.odcv import odcv_judge


def test_missing_usage_retains_each_reservation_and_response(tmp_path, monkeypatch):
    monkeypatch.setattr(odcv_judge, 'provider_price', lambda _: {'in': 1, 'out': 1})
    budget = odcv_judge.JudgeBudget(tmp_path/'ledger.json', 1, 100)
    response = SimpleNamespace(usage=None, model_dump_json=lambda **_: json.dumps(
        {'usage': None, 'error': {'message': 'Provider blocked request'}}))
    for _ in range(2):
        with pytest.raises(RuntimeError, match='reservation retained'):
            budget.create(lambda **_: response, model='test', messages=[{'role': 'user', 'content': 'fixture'}])
    rows = json.loads((tmp_path/'ledger.json').read_text())
    assert len(rows) == 2 and all(e['status'] == 'reserved' for e in rows)
    assert sum(e['charged_or_reserved_usd'] for e in rows) > rows[0]['charged_or_reserved_usd']
    assert [e['response_diagnostic'] for e in rows] == [
        'judge_accounting_failure_0000.json', 'judge_accounting_failure_0001.json']
    assert all(json.loads((tmp_path/e['response_diagnostic']).read_text())['error'] for e in rows)
