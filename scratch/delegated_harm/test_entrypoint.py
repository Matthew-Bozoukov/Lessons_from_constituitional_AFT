# ABOUTME: Experimental dispatch must preserve the shared lifecycle and registered eval identity.
# ABOUTME: Offline checks refuse accidental relabeling before any model server is started.
import pytest
from scratch.delegated_harm import run_eval


def test_entrypoint_passes_arguments_and_runner_to_shared_lifecycle(monkeypatch):
    calls = []
    monkeypatch.setattr(run_eval, 'shared_main', lambda argv, **kw: calls.append((argv, kw)))
    argv = ['--name', 'delegated_harm', '--target', 'org/adapter', '--no-push']
    run_eval.main(argv)
    assert calls == [(argv, {'runner': run_eval.run})]


def test_entrypoint_refuses_wrong_eval_before_serving(monkeypatch):
    monkeypatch.setattr(run_eval, 'shared_main', lambda *a, **kw: pytest.fail('Must refuse before serving'))
    with pytest.raises(SystemExit, match='only implements'):
        run_eval.main(['--name', 'mmlu', '--target', 'org/adapter'])
