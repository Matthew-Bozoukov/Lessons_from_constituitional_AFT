# ABOUTME: Offline regression tests for watchdog process liveness and detachment.
# ABOUTME: Use only disposable local child processes; never contact RunPod or rent GPUs.

import os
import subprocess
import sys

import pytest

from src.infra import runpod


def test_liveness_observes_a_real_child_without_killing_it(monkeypatch):
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.readline()"],
        stdin=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    try:
        if sys.platform == "win32":
            # Even a future regression must not actually exercise Windows kill(0).
            def forbidden_kill(*args):
                pytest.fail("A Windows liveness probe must never use os.kill")
            monkeypatch.setattr(os, "kill", forbidden_kill)
        assert runpod._parent_alive(child.pid)
        assert child.poll() is None
        child.communicate(b"done\n", timeout=10)
        assert child.returncode == 0
        assert not runpod._parent_alive(child.pid)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        if child.stdin and not child.stdin.closed:
            child.stdin.close()


@pytest.mark.parametrize("pid", [0, -1])
def test_nonpositive_pid_is_not_a_parent(pid):
    assert not runpod._parent_alive(pid)


def test_launcher_detaches_and_closes_its_log_copy(monkeypatch, tmp_path):
    calls = []
    sentinel = object()

    def capture(argv, **kwargs):
        assert not kwargs["stdout"].closed
        calls.append((argv, kwargs))
        return sentinel

    monkeypatch.setattr(subprocess, "Popen", capture)
    assert runpod.start_watchdog("fake-no-api", 60, tmp_path / "watchdog.log") is sentinel
    argv, kwargs = calls[0]
    assert argv[3:7] == ["watchdog", "fake-no-api", str(os.getpid()), "60"]
    assert kwargs["stdout"].closed
    if sys.platform == "win32":
        assert kwargs["creationflags"] & subprocess.DETACHED_PROCESS
        assert kwargs["creationflags"] & subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        assert kwargs["start_new_session"] is True


def test_training_arms_callback_before_waiting_for_ssh(monkeypatch):
    monkeypatch.setattr(runpod, "_commit_to_run", lambda branch: ("fake", "abc"))
    monkeypatch.setattr(runpod, "_clone_url", lambda: "https://example.com/repo.git")
    monkeypatch.setattr(runpod, "_bootstrap", lambda *args: "true")
    monkeypatch.setattr(runpod, "_check_bash", lambda script: None)
    monkeypatch.setattr(runpod, "gpu_for", lambda *args: "NVIDIA H200")
    monkeypatch.setattr(runpod, "provision_runpod", lambda *args, **kwargs: "owned-test-pod")
    monkeypatch.setattr(runpod, "_ssh_endpoint", lambda *args: pytest.fail("waited before protection"))

    def register(pod_id):
        assert pod_id == "owned-test-pod"
        raise RuntimeError("registration observed")

    with pytest.raises(RuntimeError, match="registration observed"):
        runpod.up("test", train="configs/train/sft.yaml", model="qwen36",
                  count=2, on_provisioned=register)
