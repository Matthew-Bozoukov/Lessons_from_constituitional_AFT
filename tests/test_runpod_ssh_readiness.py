# ABOUTME: Verify a stuck SSH subprocess cannot outlive the readiness deadline.
# ABOUTME: Uses a real local child, without connecting to or renting a GPU.
import sys
import time

from src.infra import runpod


def test_readiness_bounds_a_child_that_never_exits(monkeypatch):
    monkeypatch.setattr(runpod, "ssh_argv", lambda *args: (
        [getattr(sys, "_base_executable", sys.executable), "-c", "import time; time.sleep(60)"], "unused-host"))
    started = time.monotonic()
    assert not runpod._wait_for_ssh("unused-host", timeout_s=.2)
    assert time.monotonic() - started < 3
