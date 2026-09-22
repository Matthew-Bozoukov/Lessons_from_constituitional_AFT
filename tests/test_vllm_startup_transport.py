# ABOUTME: A lost SSH acknowledgement or process probe must not kill a healthy vLLM server.
# ABOUTME: Transport recovery checks readiness without launching a second server or sampling requests.
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from src.infra.endpoints import vllm


def server(executor):
    obj=object.__new__(vllm.VllmServer)
    obj.executor=executor
    obj.port=8000
    return obj


def test_healthy_http_does_not_require_another_ssh_connection(monkeypatch):
    executor=SimpleNamespace(endpoint_host='localhost',alive=Mock(side_effect=AssertionError('SSH not needed')))
    monkeypatch.setattr(vllm.requests,'get',lambda *a,**k:SimpleNamespace(status_code=200))
    server(executor)._wait_healthy()
    executor.alive.assert_not_called()


def test_readiness_survives_one_unavailable_process_probe(monkeypatch):
    responses=iter([requests.ConnectionError('not ready'),SimpleNamespace(status_code=200)])
    def get(*a,**k):
        result=next(responses)
        if isinstance(result,Exception):
            raise result
        return result
    executor=SimpleNamespace(endpoint_host='localhost',alive=Mock(side_effect=subprocess.TimeoutExpired('ssh',10)))
    monkeypatch.setattr(vllm.requests,'get',get)
    monkeypatch.setattr(vllm.time,'sleep',lambda _:None)
    server(executor)._wait_healthy()
    executor.alive.assert_called_once()


def test_confirmed_missing_process_still_fails(monkeypatch):
    executor=SimpleNamespace(endpoint_host='localhost',alive=lambda:False,tail_log=lambda:'crashed')
    monkeypatch.setattr(vllm.requests,'get',lambda *a,**k:SimpleNamespace(status_code=503))
    with pytest.raises(RuntimeError,match='vLLM exited'):
        server(executor)._wait_healthy()


def test_lost_launch_acknowledgement_never_dispatches_twice(monkeypatch):
    executor=vllm.SshExec('test-host',port=8000)
    monkeypatch.setattr(executor,'write_file',lambda *a:'/workspace/start.sh')
    dispatch=Mock(side_effect=subprocess.TimeoutExpired('ssh',60))
    monkeypatch.setattr(executor,'_ssh',dispatch)
    tunnel=Mock()
    monkeypatch.setattr(vllm.subprocess,'Popen',tunnel)
    executor.start_server(['python','-m','vllm.entrypoints.openai.api_server'],{})
    dispatch.assert_called_once()
    tunnel.assert_called_once()
