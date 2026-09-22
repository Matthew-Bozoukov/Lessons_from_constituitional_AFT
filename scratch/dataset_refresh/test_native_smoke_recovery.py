# ABOUTME: Tests Windows ledger write recovery and prevention of uncertain-call redispatch.
# ABOUTME: Run: uv run --no-sync python -m pytest -q scratch/dataset_refresh/test_native_smoke_recovery.py
import json
from pathlib import Path

import pytest

from scratch.dataset_refresh import run
from scratch.dataset_refresh.run_native_smoke import GuardedClient
from src.infra.endpoints.openrouter import EmptyCompletionError


def test_atomic_replace_retries_only_filesystem_operation(tmp_path, monkeypatch):
    original = Path.replace
    calls = []
    def busy_once(self, target):
        calls.append(target)
        if len(calls) == 1:
            raise PermissionError('Windows sharing violation')
        return original(self, target)
    monkeypatch.setattr(Path, 'replace', busy_once)
    monkeypatch.setattr(run.time, 'sleep', lambda _: None)
    target = tmp_path / 'spend.json'
    run.write_json(target, [{'reserved': 0.1}])
    assert json.loads(target.read_text()) == [{'reserved': 0.1}]
    assert len(calls) == 2


def test_uncertain_request_is_not_resent_or_unreserved(tmp_path, monkeypatch):
    root = tmp_path / 'run'
    request = dict(model='anthropic/claude-sonnet-5', messages=[{'role':'user','content':'test'}], temperature=0, max_tokens=100)
    entry = dict(run_root=str(root.resolve()), request_sha256=run.digest(request), status='reserved', charged_or_reserved_usd=0.1)
    run.write_json(tmp_path / 'spend.json', [entry])
    monkeypatch.setattr(run, 'OpenRouterClient', lambda: None)
    original = run.BudgetClient.__init__
    def init(self, *args, **kwargs):
        return original(self, *args, send=lambda **_: pytest.fail('Redispatched uncertain paid request'), **kwargs)
    monkeypatch.setattr(run.BudgetClient, '__init__', init)
    client = GuardedClient(tmp_path, {'ceiling_usd':20,'max_physical_calls':378}, root)
    with pytest.raises(ValueError, match='without redispatch'):
        client.chat(**request)
    assert client.count == 1
    assert not client.stop.is_set()
    assert client.entries() == [entry]


def test_settled_response_is_reused_without_new_dispatch(tmp_path, monkeypatch):
    root = tmp_path / 'run'
    request = dict(model='anthropic/claude-sonnet-5', messages=[{'role':'user','content':'test'}], temperature=0, max_tokens=100)
    entry = dict(call_id=0,run_root=str(root.resolve()),request_sha256=run.digest(request),status='settled',charged_or_reserved_usd=0.01)
    run.write_json(tmp_path/'spend.json',[entry])
    run.write_json(tmp_path/'raw_calls/000000.json',dict(request=request,response=dict(content='saved response',prompt_tokens=2,completion_tokens=3,finish_reason='stop')))
    original=run.BudgetClient.__init__
    monkeypatch.setattr(run.BudgetClient,'__init__',lambda self,*a,**k: original(self,*a,send=lambda **_: pytest.fail('Repaid for saved response'),**k))
    client=GuardedClient(tmp_path,{'ceiling_usd':20,'max_physical_calls':378},root)
    assert client.chat(**request).content=='saved response'
    assert client.count==1 and client.reused==1 and client.entries()==[entry]


def test_explicit_provider_block_is_row_failure_but_unknown_failure_stops(tmp_path, monkeypatch):
    original=run.BudgetClient.__init__
    monkeypatch.setattr(run.BudgetClient,'__init__',lambda self,*a,**k: original(self,*a,send=lambda **_: None,**k))
    client=GuardedClient(tmp_path,{'ceiling_usd':20,'max_physical_calls':378},tmp_path/'run')
    request=dict(model='anthropic/claude-sonnet-5',messages=[{'role':'user','content':'test'}],temperature=0,max_tokens=100)
    def blocked(*a,**k):
        raise EmptyCompletionError('blocked',diagnostics={'choices':[{'finish_reason':'content_filter'}]})
    monkeypatch.setattr(run.BudgetClient,'chat',blocked)
    with pytest.raises(EmptyCompletionError): client.chat(**request)
    assert not client.stop.is_set()
    def unknown(*a,**k):
        raise EmptyCompletionError('unknown')
    monkeypatch.setattr(run.BudgetClient,'chat',unknown)
    with pytest.raises(EmptyCompletionError): client.chat(**request)
    assert client.stop.is_set()


def test_settled_truncated_response_is_not_retried_on_operational_resume(tmp_path, monkeypatch):
    root = tmp_path / 'run'
    request = dict(model='anthropic/claude-sonnet-5', messages=[{'role':'user','content':'test'}], temperature=0, max_tokens=100)
    entry = dict(call_id=0,run_root=str(root.resolve()),request_sha256=run.digest(request),status='settled',charged_or_reserved_usd=0.01)
    run.write_json(tmp_path/'spend.json',[entry])
    run.write_json(tmp_path/'raw_calls/000000.json',dict(request=request,response=dict(content='truncated',prompt_tokens=2,completion_tokens=100,finish_reason='length')))
    original=run.BudgetClient.__init__
    monkeypatch.setattr(run.BudgetClient,'__init__',lambda self,*a,**k: original(self,*a,send=lambda **_: pytest.fail('Repaid for truncated failure'),**k))
    client=GuardedClient(tmp_path,{'ceiling_usd':120,'max_physical_calls':6000},root)
    with pytest.raises(ValueError, match='without redispatch'):
        client.chat(**request)
