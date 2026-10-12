# ABOUTME: Preserve malformed-provider diagnostics and budget reservations without accepting verdicts.
# ABOUTME: Repeated identical failures need distinct diagnostic files and cumulative charges.
import json
from types import SimpleNamespace
from pathlib import Path

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


def test_transient_replace_retries_without_resending(tmp_path, monkeypatch):
    monkeypatch.setattr(odcv_judge, 'provider_price', lambda _: {'in': 1, 'out': 1})
    monkeypatch.setattr(odcv_judge.time, 'sleep', lambda _: None)
    original = Path.replace
    attempts = []
    def replace(path, target):
        attempts.append(str(path))
        if len(attempts) == 1:
            raise PermissionError('transient sharing violation')
        return original(path, target)
    monkeypatch.setattr(Path, 'replace', replace)
    calls = []
    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10,completion_tokens=2),
                               choices=[SimpleNamespace(finish_reason='stop')])
    def send(**kw):
        calls.append(kw)
        return response
    budget = odcv_judge.JudgeBudget(tmp_path/'ledger.json', 1, 100)
    assert budget.create(send, model='test',messages=[{'role':'user','content':'fixture'}]) is response
    assert len(calls) == 1 and len(attempts) == 3
    assert json.loads((tmp_path/'ledger.json').read_text())[0]['status'] == 'settled'


def test_null_comparison_key_preserves_judged_results(tmp_path, monkeypatch):
    from omegaconf import OmegaConf
    cfg=tmp_path/'config.yaml'
    OmegaConf.save(OmegaConf.create(dict(bench_dir=str(tmp_path),model='base',model_key='base',
        judges={'judge':'model'},published_key=None)),cfg)
    monkeypatch.setattr(odcv_judge, '_load_upstream_judge', lambda _: None)
    monkeypatch.setattr(odcv_judge, 'transcripts', lambda *a: [('mandated','Task',tmp_path/'transcript')])
    monkeypatch.setattr(odcv_judge, 'openrouter_usage', lambda **k: None)
    monkeypatch.setattr(odcv_judge, 'score_with_judge', lambda *a,**k: {'mandated/Task':{'score':0,'reasoning':'cached'}})
    monkeypatch.setattr(odcv_judge, 'summarise', lambda _: {'overall':{'mr_pct':0,'mr_ci95':[0,0],'mean_severity':0}})
    monkeypatch.setattr(odcv_judge, 'write_run_meta', lambda *a,**k: None)
    def forbidden(*a): raise AssertionError('No external comparison should be loaded')
    monkeypatch.setattr(odcv_judge, 'load_published_medians', forbidden)
    odcv_judge.main(str(tmp_path),str(cfg))
    result=json.loads((tmp_path/'results.json').read_text())
    assert result['n_judged']==1 and result['published'] is None and result['delta_mr_pct'] is None
