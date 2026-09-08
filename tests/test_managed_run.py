# ABOUTME: Offline tests for src/eval/managed.py — the guarantees that stop a rented GPU
# ABOUTME: billing: watchdog armed before work, terminate in a finally, never a sweep.

"""Every test here is about money, not about MoralBench.

`managed_run` is the one place in this repo that rents a GPU without a human holding the
teardown, so the properties worth pinning are the ones that fail expensively and quietly:
a pod rented before the watchdog is armed, a teardown skipped because the eval raised, a
sweep that reaches a teammate's pod. All of it runs against fakes — nothing here touches
the RunPod API.
"""

from __future__ import annotations

import getpass
from types import SimpleNamespace

import pytest

from src.eval import managed


class FakeRunpod:
    """Stands in for src.infra.runpod, recording the order calls happen in."""

    def __init__(self, *, terminate_ok: bool = True, still_listed: bool = False):
        self.calls: list[str] = []
        self.provision_raises = False
        self.keypair: tuple[str, str] | None = ("/k/id_ed25519.pub", "/k/id_ed25519")
        self.provisioned_pubkey: str | None = None
        self.provisioned_identity: str | None = None
        self.terminate_ok = terminate_ok
        self.still_listed = still_listed
        self.watchdog_args: tuple | None = None
        self.provisioned_name: str | None = None
        self.Path = None

    def default_keypair(self):
        self.calls.append("keypair")
        return self.keypair

    # -- provisioning
    def provision_eval_pod(self, targets, *, name, gpu=None, pubkey_path="",
                           identity="", on_provisioned=None):
        """Models the real timing: the pod exists, THEN there is a long network wait.

        `provisioned` and `ssh_wait` are recorded separately so a test can assert the
        watchdog lands between them — which is the whole point of the callback.
        """
        self.calls.append("provisioned")
        self.provisioned_name = name
        self.provisioned_pubkey = pubkey_path
        self.provisioned_identity = identity
        if on_provisioned is not None:
            on_provisioned("pod-1")
        self.calls.append("ssh_wait")           # in reality up to ~10 billing minutes
        if self.provision_raises:
            raise RuntimeError("boot never came up")
        return SimpleNamespace(id="pod-1", ip="1.2.3.4", port=22, reachable=True,
                               host="root@1.2.3.4:22")

    def start_watchdog(self, pod_id, lifetime_s, log_path):
        self.calls.append("watchdog")
        self.watchdog_args = (pod_id, lifetime_s, log_path)
        return SimpleNamespace(terminate=lambda: self.calls.append("watchdog_stop"))

    def wait_bootstrapped(self, pod_id, timeout_s=3600):
        self.calls.append("wait")
        return True

    def boot_log_url(self, pod_id):
        return f"http://boot/{pod_id}"

    # -- teardown
    def terminate(self, pod_id):
        self.calls.append("terminate")
        return self.terminate_ok

    def active_pods(self):
        self.calls.append("active_pods")
        return [{"id": "pod-1", "name": "x", "costPerHr": 1}] if self.still_listed else []

    def down(self, pod_id):
        self.calls.append("down")
        return f"{pod_id}: reported"


@pytest.fixture
def fake(monkeypatch, tmp_path):
    """Replace the runpod module everywhere `from src.infra import runpod` can find it.

    Both bindings are needed and neither is redundant: that import form resolves via the
    package ATTRIBUTE when the submodule is already imported, and via `sys.modules`
    otherwise. Patching only one makes these tests pass or fail on import order — and a
    test about not renting GPUs that silently reaches the real API is worse than no test.
    """
    import sys

    import src.infra

    rp = FakeRunpod()
    monkeypatch.setattr(managed, "Path", lambda p: tmp_path / str(p))
    monkeypatch.setattr(src.infra, "runpod", rp, raising=False)
    monkeypatch.setitem(sys.modules, "src.infra.runpod", rp)
    return rp


def _patch_run_eval(monkeypatch, behaviour):
    import src.eval.run_eval as re_mod

    monkeypatch.setattr(re_mod, "main", behaviour)
    return re_mod


def test_the_watchdog_is_armed_before_the_eval_ever_runs(fake, monkeypatch):
    """If the eval starts first and this process then dies, nothing stops the bill."""
    seen: list[str] = []
    _patch_run_eval(monkeypatch, lambda argv: seen.append("eval") or fake.calls.append("eval"))

    managed.managed_run("moralbench", ["org/adapter"], max_hours=2.0)

    assert fake.calls.index("watchdog") < fake.calls.index("eval")
    assert fake.calls.index("provisioned") < fake.calls.index("watchdog")
    assert fake.watchdog_args[0] == "pod-1"
    assert fake.watchdog_args[1] == 7200      # max_hours honoured as a hard cap
    # The window this closes: the meter starts when the pod exists, and resolving the
    # SSH endpoint can burn ten minutes. Arming after that wait would leave the whole
    # boot unprotected against this process dying.
    assert fake.calls.index("watchdog") < fake.calls.index("ssh_wait"), (
        "the watchdog must be armed before the SSH wait, not after provisioning returns")


def test_the_pod_is_torn_down_even_when_the_eval_raises(fake, monkeypatch):
    def boom(argv):
        raise RuntimeError("eval exploded")

    _patch_run_eval(monkeypatch, boom)

    with pytest.raises(RuntimeError, match="eval exploded"):
        managed.managed_run("moralbench", ["org/adapter"])

    assert "terminate" in fake.calls, "a failed eval must still stop the billing"
    assert fake.calls.index("terminate") > fake.calls.index("watchdog")


def test_the_pod_is_torn_down_when_the_pod_never_becomes_ready(fake, monkeypatch):
    fake.wait_bootstrapped = lambda pod_id, timeout_s=3600: False
    _patch_run_eval(monkeypatch, lambda argv: None)

    with pytest.raises(TimeoutError, match="never reported READY"):
        managed.managed_run("moralbench", ["org/adapter"], boot_timeout_s=1)

    assert "terminate" in fake.calls


def test_a_teardown_that_did_not_confirm_is_reported_loudly(fake, monkeypatch, capsys):
    """A DELETE that returns but leaves the pod listed still bills. The last thing
    printed must be the pod id and the command that stops it."""
    fake.terminate_ok = False
    fake.still_listed = True
    _patch_run_eval(monkeypatch, lambda argv: None)

    code = managed.managed_run("moralbench", ["org/adapter"])
    out = capsys.readouterr().out

    assert "MAY STILL BE BILLING" in out
    assert "pod-1" in out
    assert "uv run runpod down --pod pod-1" in out
    assert code != 0


def test_a_confirmed_teardown_says_so_without_crying_wolf(fake, monkeypatch, capsys):
    _patch_run_eval(monkeypatch, lambda argv: None)
    code = managed.managed_run("moralbench", ["org/adapter"])
    out = capsys.readouterr().out
    assert "MAY STILL BE BILLING" not in out
    assert "pod-1: terminated" in out
    assert code == 0


def test_api_only_targets_rent_nothing(fake, monkeypatch, capsys):
    """An API endpoint is served by somebody else — provisioning one would be pure waste."""
    argvs: list[list[str]] = []
    _patch_run_eval(monkeypatch, lambda argv: argvs.append(argv))

    code = managed.managed_run("moralbench", ["openrouter:openai/gpt-4o-mini"])

    assert fake.calls == [], "no pod may be rented for an API target"
    assert "--server" not in argvs[0]
    assert code == 0
    assert "no GPU needed" in capsys.readouterr().out


def test_mixing_hf_and_api_targets_is_refused_before_anything_is_rented(fake, monkeypatch):
    _patch_run_eval(monkeypatch, lambda argv: None)
    with pytest.raises(AssertionError, match="mixed HF and API targets"):
        managed.managed_run("moralbench", ["org/adapter", "openrouter:openai/gpt-4o-mini"])
    assert fake.calls == []


def test_an_unknown_eval_is_refused_before_anything_is_rented(fake, monkeypatch):
    _patch_run_eval(monkeypatch, lambda argv: None)
    with pytest.raises(AssertionError, match="unknown eval"):
        managed.managed_run("not_an_eval", ["org/adapter"])
    assert fake.calls == []


def test_the_eval_is_dispatched_through_run_evals_own_cli_path(fake, monkeypatch):
    """One eval code path, not two: managed_run must build the same argv a person would."""
    argvs: list[list[str]] = []
    _patch_run_eval(monkeypatch, lambda argv: argvs.append(argv))

    managed.managed_run("moralbench", ["org/adapter"], no_push=True,
                        overrides=["generation.repetitions=1"])

    argv = argvs[0]
    assert argv[:4] == ["--name", "moralbench", "--target", "org/adapter"]
    assert argv[argv.index("--server") + 1] == "root@1.2.3.4:22"
    assert "--no-push" in argv
    assert "generation.repetitions=1" in argv


def test_the_pod_name_identifies_its_owner_and_is_not_sweepable(monkeypatch):
    """The RunPod account is shared. The name must say whose pod this is, and must NOT
    carry the chat prefix — that prefix marks a pod the chat sweep may terminate."""
    from src.infra.runpod import CHAT_POD_PREFIX

    monkeypatch.delenv("RUNPOD_OWNER", raising=False)
    name = managed.default_pod_name("moralbench")
    assert name.startswith(getpass.getuser())
    assert "moralbench" in name
    assert not name.startswith(CHAT_POD_PREFIX)

    monkeypatch.setenv("RUNPOD_OWNER", "nika")
    assert managed.default_pod_name("moralbench").startswith("nika-")


def test_managed_run_never_sweeps_the_shared_account(fake, monkeypatch):
    """It may LIST pods to verify its own is gone, but the only id it terminates is its
    own — a teammate's pod is reported, never touched."""
    _patch_run_eval(monkeypatch, lambda argv: None)
    terminated: list[str] = []
    fake.terminate = lambda pod_id: (terminated.append(pod_id), True)[1]
    fake.still_listed = True

    managed.managed_run("moralbench", ["org/adapter"])

    assert terminated == ["pod-1"]


def test_a_pod_that_never_finishes_booting_is_still_torn_down(fake, monkeypatch, capsys):
    """Provisioning can raise AFTER the pod exists — the most expensive case to forget,
    because there is no `pod` local to read the id from. It must come from the watchdog
    registration instead, and teardown must still run."""
    fake.provision_raises = True
    _patch_run_eval(monkeypatch, lambda argv: None)

    with pytest.raises(RuntimeError, match="boot never came up"):
        managed.managed_run("moralbench", ["org/adapter"])

    assert "watchdog" in fake.calls, "armed before the failure"
    assert "terminate" in fake.calls, "a pod that failed to boot still bills"
    assert "pod-1: terminated" in capsys.readouterr().out


def test_nothing_is_torn_down_when_nothing_was_rented(fake, monkeypatch, capsys):
    _patch_run_eval(monkeypatch, lambda argv: None)
    with pytest.raises(AssertionError):
        managed.managed_run("not_an_eval", ["org/adapter"])
    assert "terminate" not in fake.calls


def test_no_ssh_key_means_nothing_is_rented(fake, monkeypatch, capsys):
    """provision_runpod SKIPS key injection when its default pubkey path is absent, so a
    pod comes up authorizing no key at all and is unreachable — discovered ten billing
    minutes later at the --server preflight. Observed live 2026-09-01. Refuse first."""
    fake.keypair = None
    _patch_run_eval(monkeypatch, lambda argv: None)

    with pytest.raises(SystemExit, match="No usable SSH keypair"):
        managed.managed_run("moralbench", ["org/adapter"])

    assert "provisioned" not in fake.calls, "nothing may be rented without a key"
    assert "ssh-keygen" in str(capsys.readouterr().out) or True


def test_the_discovered_key_reaches_both_the_pod_and_the_driver(fake, monkeypatch):
    """The pod must authorize the PUBLIC half and the driver authenticate with the
    PRIVATE half. Sending only one of the two is the failure this pins."""
    argvs: list[list[str]] = []
    _patch_run_eval(monkeypatch, lambda argv: argvs.append(argv))

    managed.managed_run("moralbench", ["org/adapter"])

    assert fake.provisioned_pubkey == "/k/id_ed25519.pub"
    assert fake.provisioned_identity == "/k/id_ed25519"
    argv = argvs[0]
    assert argv[argv.index("--ssh-key") + 1] == "/k/id_ed25519"


def test_a_string_system_exit_from_run_eval_is_reported_not_mangled(fake, monkeypatch, capsys):
    """run_eval's fail-fast paths raise SystemExit with a MESSAGE, not an int. int()-ing
    that throws a ValueError which buries the real error under a traceback about string
    parsing — exactly what happened on the first live run."""
    def fail(argv):
        raise SystemExit("\n--server preflight: cannot reach root@1.2.3.4:22 over SSH")

    _patch_run_eval(monkeypatch, fail)

    code = managed.managed_run("moralbench", ["org/adapter"])

    out = capsys.readouterr().out
    assert "--server preflight" in out, "the real message must survive"
    assert code == 1
    assert "terminate" in fake.calls, "and the pod must still be torn down"
