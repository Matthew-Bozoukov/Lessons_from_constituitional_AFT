# ABOUTME: Offline tests for opt-in RunPod ownership, process guards, and eval teardown.
# ABOUTME: All provider calls and watchdog processes are mocked; no GPUs are rented.
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from src.infra import runpod
from src.eval import run_eval


def pod():
    return {"id": "owned", "publicIp": "192.0.2.1", "portMappings": {"22": 1234},
            "env": {"LASR_POD_OWNER": runpod.POD_OWNER,
                    "LASR_POD_DEADLINE": str(runpod.time.time() + 600)}}


@pytest.fixture
def lookup(monkeypatch):
    rows = [pod()]
    monkeypatch.setattr(runpod, "active_pods", lambda: rows)
    monkeypatch.setattr(runpod.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(stdout="hostname 192.0.2.1\nport 1234\n"))
    monkeypatch.setattr(runpod.socket, "getaddrinfo", lambda *a: [(None, None, None, None, ("192.0.2.1", 0))])
    return rows


def test_alias_resolves_live_pod(lookup):
    assert runpod.pod_for_server("my-alias")["id"] == "owned"


@pytest.mark.parametrize("case", ["missing", "ambiguous", "unowned", "port", "deadline"])
def test_lookup_refuses_unsafe_targets(lookup, case):
    if case == "missing": lookup.clear()
    if case == "ambiguous": lookup.append(pod())
    if case == "unowned": lookup[0]["env"].clear()
    if case == "port": lookup[0]["portMappings"]["22"] = 99
    if case == "deadline": lookup[0]["env"]["LASR_POD_DEADLINE"] = "nan"
    with pytest.raises(ValueError):
        runpod.pod_for_server("alias")


@pytest.mark.parametrize("fail", [False, True])
def test_guard_cleanup_order(monkeypatch, fail):
    events = []
    monkeypatch.setattr(runpod, "pod_for_server", lambda s: pod())
    monkeypatch.setattr(runpod, "start_watchdog", lambda *a: SimpleNamespace(terminate=lambda: events.append("guard-stop")))
    monkeypatch.setattr(runpod, "teardown", lambda p: events.append("teardown"))
    try:
        with runpod.eval_pod("alias"):
            events.append("eval-and-publish")
            if fail: raise RuntimeError("eval failed")
    except RuntimeError:
        assert fail
    assert events == ["eval-and-publish", "teardown", "guard-stop"]


def test_failed_teardown_leaves_guard(monkeypatch):
    events = []
    monkeypatch.setattr(runpod, "pod_for_server", lambda s: pod())
    monkeypatch.setattr(runpod, "start_watchdog", lambda *a: SimpleNamespace(terminate=lambda: events.append("stopped")))
    def fail(p): raise RuntimeError("provider unavailable")
    monkeypatch.setattr(runpod, "teardown", fail)
    with pytest.raises(RuntimeError):
        with runpod.eval_pod("alias"): pass
    assert not events


@pytest.mark.parametrize("owned", [False, True])
def test_eval_entrypoint_opt_in(monkeypatch, owned):
    events = []
    @contextmanager
    def lifecycle(server):
        events.append("attach")
        try: yield
        finally: events.append("cleanup")
    monkeypatch.setattr(runpod, "eval_pod", lifecycle)
    monkeypatch.setattr(run_eval, "_run", lambda *a: events.append("eval"))
    args = ["--name", "mask", "--target", "org/model", "--server", "alias"]
    run_eval.main(args + (["--terminate-pod"] if owned else []))
    assert events == (["attach", "eval", "cleanup"] if owned else ["eval"])


def test_ownership_requires_server():
    with pytest.raises(SystemExit):
        run_eval.main(["--name", "mask", "--target", "org/model", "--terminate-pod"])


@pytest.mark.parametrize("hours", [0, -1, float("inf"), float("nan")])
def test_bad_deadlines_fail_before_provision(hours):
    with pytest.raises(ValueError): runpod.up("test", eval="org/model", max_hours=hours)


@pytest.mark.parametrize("failure", [None, "watchdog", "ssh"])
def test_up_arms_deadline_before_ssh(monkeypatch, failure):
    events = []
    monkeypatch.setattr(runpod, "plan_eval_pod", lambda *a: (["org/model"], None, None, 200))
    monkeypatch.setattr(runpod, "_bootstrap", lambda *a: "true")
    def provision(spec, **kwargs):
        assert kwargs["env"]["LASR_POD_OWNER"] == runpod.POD_OWNER
        assert float(kwargs["env"]["LASR_POD_DEADLINE"]) > runpod.time.time()
        events.append("rent")
        return "owned"
    def arm(*args, **kwargs):
        assert kwargs["parent_pid"] == 0
        events.append("arm")
        if failure == "watchdog": raise RuntimeError("spawn failed")
        return SimpleNamespace(terminate=lambda: events.append("stop"))
    def ssh(*args):
        events.append("ssh")
        if failure == "ssh": raise RuntimeError("ssh failed")
        return "192.0.2.1", 1234
    monkeypatch.setattr(runpod, "provision_runpod", provision)
    monkeypatch.setattr(runpod, "start_watchdog", arm)
    monkeypatch.setattr(runpod, "_ssh_endpoint", ssh)
    monkeypatch.setattr(runpod, "_wait_for_ssh", lambda *a: True)
    monkeypatch.setattr(runpod, "teardown", lambda p: events.append("teardown"))
    if failure:
        with pytest.raises(RuntimeError): runpod.up("test", eval="org/model")
        assert "teardown" in events
    else:
        runpod.up("test", eval="org/model")
        assert events == ["rent", "arm", "ssh"]


@pytest.mark.parametrize("parent,identity,alive,expected", [(0, "", False, "deadline"), (22, "old", True, "death")])
def test_watchdog_deadline_or_recycled_pid(monkeypatch, parent, identity, alive, expected):
    monkeypatch.setattr(runpod.time, "sleep", lambda n: None)
    times = iter([0, 100])
    monkeypatch.setattr(runpod.time, "time", lambda: next(times))
    monkeypatch.setattr(runpod, "active_pods", lambda: [{"id": "owned"}])
    monkeypatch.setattr(runpod, "_parent_alive", lambda p: alive)
    monkeypatch.setattr(runpod, "_process_identity", lambda p: "new")
    events = []
    monkeypatch.setattr(runpod, "teardown", lambda p: events.append(p))
    runpod.watchdog("owned", parent, 10, parent_identity=identity)
    assert events == ["owned"]


def test_early_release_tears_down_once_and_the_exit_path_does_not_repeat_it(monkeypatch):
    events = []
    monkeypatch.setattr(runpod, "pod_for_server", lambda s: pod())
    monkeypatch.setattr(runpod, "start_watchdog", lambda *a: SimpleNamespace(terminate=lambda: events.append("guard-stop")))
    monkeypatch.setattr(runpod, "teardown", lambda p: events.append("teardown"))
    with runpod.eval_pod("alias") as release:
        events.append("generated")
        release()          # MASK before its batch wait
        release()          # idempotent
        events.append("judged-and-published")
    assert events == ["generated", "teardown", "judged-and-published", "guard-stop"]
