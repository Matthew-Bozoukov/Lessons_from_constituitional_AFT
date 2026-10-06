# ABOUTME: Mocked safety checks for the bounded plain-dose evaluation owner.
# ABOUTME: Exercises ownership, watchdog ordering, failure cleanup and lexical short paths without services.
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from scratch.plain_dose import eval_owner as owner


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.setattr(owner, 'ROOT', tmp_path)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / 'output' / 'run'
    plan = {'eval': 'mask', 'target': 'dougalldeepmind/2026-10-06-qwen36-0-da-5',
            'target_revision': 'a' * 40, 'base_revision': owner.BASE_REVISION,
            'pod_name': 'nika-test-only', 'port': 8097, 'output_dir': str(output),
            'max_hours': 2, 'hourly_ceiling_usd': 5}
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    spec = SimpleNamespace(revision=plan['target_revision'], base_revision=owner.BASE_REVISION,
                           mode='think', base_model='Qwen/Qwen3.6-27B')
    monkeypatch.setattr(owner, 'resolve_target', MagicMock(return_value=spec))
    monkeypatch.setattr(owner.socket, 'socket', MagicMock())
    remote = MagicMock()
    remote._ssh.return_value = 'booted\nREADY\n'
    monkeypatch.setattr(owner, 'SshExec', MagicMock(return_value=remote))
    monkeypatch.setattr(owner.time, 'sleep', lambda _: pytest.fail('Unexpected monitor wait'))
    events = []
    guard = MagicMock(pid=101)

    def watchdog(pod, seconds, log):
        state = json.loads((output / 'status.json').read_text())
        assert state['owned_pod'] == pod == 'owned-pod'
        assert state['phase'] == 'booting'
        events.append('watchdog')
        return guard

    monkeypatch.setattr(owner.runpod, 'start_watchdog', MagicMock(side_effect=watchdog))
    monkeypatch.setattr(owner.runpod, 'call', MagicMock(return_value={'costPerHr': 3.0}))
    teardown = MagicMock(side_effect=lambda pod: events.append(('teardown', pod)))
    monkeypatch.setattr(owner.runpod, 'teardown', teardown)

    def up(**kwargs):
        assert kwargs['eval'] == 'mask' and kwargs['target'] == plan['target']
        assert kwargs['count'] == 1
        events.append('provision')
        kwargs['on_provisioned']('owned-pod')
        events.append('callback-finished')
        return 'pod: owned-pod\nhost: root@example.invalid:2222'

    monkeypatch.setattr(owner.runpod, 'up', MagicMock(side_effect=up))
    child = MagicMock(pid=102, returncode=7)
    child.poll.return_value = 7

    def popen(argv, **kwargs):
        assert events.index('watchdog') < events.index('callback-finished')
        state = json.loads((output / 'status.json').read_text())
        assert state['watchdog_pid'] == guard.pid
        assert f"target_revision={plan['target_revision']}" in argv
        assert 'max_generation_error_rate=0.05' in argv
        events.append('child')
        return child

    monkeypatch.setattr(owner.subprocess, 'Popen', MagicMock(side_effect=popen))
    return SimpleNamespace(plan=plan, path=path, output=output, events=events,
                           child=child, guard=guard, teardown=teardown)


def test_watchdog_and_owned_receipt_precede_child(harness):
    assert owner.run(harness.path) == 1
    assert harness.events[:4] == ['provision', 'watchdog', 'callback-finished', 'child']
    owner.resolve_target.assert_called_once_with(harness.plan['target'], revision='a' * 40)


def test_child_failure_still_tears_down_only_owned_pod(harness):
    assert owner.run(harness.path) == 1
    harness.teardown.assert_called_once_with('owned-pod')
    harness.guard.terminate.assert_called_once()
    status = json.loads((harness.output / 'status.json').read_text())
    assert status['eval_exit'] == 7 and status['phase'] == 'failed'
    assert status['terminated'] is True


def test_failure_before_provision_never_tears_down_unrelated_pod(harness):
    owner.runpod.up.side_effect = RuntimeError('No capacity before allocation')
    assert owner.run(harness.path) == 1
    harness.teardown.assert_not_called()
    owner.runpod.start_watchdog.assert_not_called()
    owner.subprocess.Popen.assert_not_called()


def test_watchdog_start_failure_still_tears_down_registered_pod(harness):
    owner.runpod.start_watchdog.side_effect = RuntimeError('Guard failed to start')
    assert owner.run(harness.path) == 1
    harness.teardown.assert_called_once_with('owned-pod')
    owner.subprocess.Popen.assert_not_called()


def test_duplicate_claim_refused_before_any_rental(harness):
    harness.output.mkdir(parents=True)
    status = harness.output / 'status.json'
    original = '{"owned_pod": "unrelated-existing-owner"}'
    status.write_text(original)
    with pytest.raises(FileExistsError):
        owner.run(harness.path)
    assert status.read_text() == original
    owner.runpod.up.assert_not_called()
    harness.teardown.assert_not_called()


def test_teardown_failure_keeps_guard_and_reports_failure(harness):
    harness.teardown.side_effect = RuntimeError('Provider did not confirm termination')
    assert owner.run(harness.path) == 1
    harness.guard.terminate.assert_not_called()
    status = json.loads((harness.output / 'status.json').read_text())
    assert 'teardown_error' in status and not status.get('terminated')


def test_short_paths_survive_command_and_run_directory(monkeypatch, tmp_path):
    from src.naming import run_dir
    # Treat this lexical alias like the campaign's Windows junction: resolution must not occur.
    alias = tmp_path / 'short'
    monkeypatch.setattr(Path, 'resolve', lambda *_a, **_kw: pytest.fail('Expanded short path'))
    plan = {'eval': 'odcv', 'target': 'org/model', 'target_revision': 'a' * 40,
            'port': 8098, 'resume_from': str(alias / 'prior')}
    argv = owner.command(plan, 'root@example.invalid:2222', alias / 'eval')
    assert f'output_root={(alias / "eval").as_posix()}' in argv
    assert f'resume_from={(alias / "prior").as_posix()}' in argv
    assert 'concurrency=16' in argv and 'passes=3' in argv
    assert run_dir(alias / 'eval', 'qwen36-0-da-5', date='2026-10-06').parent == alias / 'eval'
    if owner.os.name == 'nt':
        assert argv[argv.index('--server-bind') + 1] == '0.0.0.0'
