# ABOUTME: Offline tests for the `uv run runpod` CLI: the commit it refuses to run, the
# ABOUTME: bootstrap it renders, and the ~/.ssh/config entry it rewrites rather than repeats.

import subprocess

import pytest

from src.infra import runpod as pod


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                          check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A real repo with a real origin, so the push check is exercised, not mocked."""
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    work = tmp_path / "work"
    subprocess.run(["git", "clone", "-q", str(origin), str(work)], check=True)
    git(work, "config", "user.email", "t@example.com")
    git(work, "config", "user.name", "t")
    (work / "a.txt").write_text("one\n")
    git(work, "add", "a.txt")
    git(work, "commit", "-qm", "one")
    git(work, "branch", "-M", "main")
    monkeypatch.chdir(work)
    return work


def test_a_commit_that_is_not_on_origin_is_refused(repo):
    # The whole point of the guard: a paid box must run code the team can fetch by name.
    git(repo, "push", "-q", "-u", "origin", "main")
    branch, sha = pod._commit_to_run(None)
    assert branch == "main"
    assert sha == git(repo, "rev-parse", "HEAD")

    # One commit further on, still local: the pod would clone origin and run the OLD one.
    (repo / "a.txt").write_text("two\n")
    git(repo, "commit", "-qam", "two")
    with pytest.raises(AssertionError, match="not on origin"):
        pod._commit_to_run(None)


def test_uncommitted_changes_to_tracked_files_are_refused(repo):
    git(repo, "push", "-q", "-u", "origin", "main")
    (repo / "a.txt").write_text("two\n")
    # The pod would clone HEAD and run code that is not what you are looking at.
    with pytest.raises(AssertionError, match="uncommitted changes"):
        pod._commit_to_run(None)


def test_untracked_files_do_not_block_a_pod(repo):
    # They are not in the clone either way, and scratch output is always lying around.
    git(repo, "push", "-q", "-u", "origin", "main")
    (repo / "notes.txt").write_text("scratch\n")
    assert pod._commit_to_run(None)[0] == "main"


def test_a_branch_origin_has_never_seen_is_refused_by_name(repo):
    git(repo, "push", "-q", "-u", "origin", "main")
    git(repo, "checkout", "-q", "-b", "side")
    (repo / "a.txt").write_text("side\n")
    git(repo, "commit", "-qam", "side")
    with pytest.raises(AssertionError, match="git push -u origin side"):
        pod._commit_to_run(None)


def test_bootstrap_checks_out_the_exact_sha_and_is_valid_bash():
    script = pod._bootstrap(("https://github.com/o/r.git", "main", "abc1234"))
    pod._check_bash(script)  # raises if bash cannot parse it
    # Detached at the SHA, never at the branch tip: the branch can move while a pod boots.
    assert "git checkout --detach abc1234" in script
    assert "git clone --branch main https://github.com/o/r.git /root/work" in script
    assert "uv sync" in script
    # sshd and the log server come up BEFORE the slow work, or a stall is undiagnosable.
    assert script.index("sshd") < script.index("git clone")
    assert script.index("http.server 8080") < script.index("curl -LsSf https://astral.sh/uv")


def test_a_bare_pod_clones_nothing():
    script = pod._bootstrap(None)
    pod._check_bash(script)
    assert "git clone" not in script and "uv sync" not in script
    assert "sshd" in script  # still reachable; that is what makes it useful


def test_ssh_remotes_are_rewritten_to_the_anonymous_https_form(monkeypatch):
    monkeypatch.setattr(pod, "_git", lambda *a: "git@github.com:org/repo.git")
    assert pod._clone_url() == "https://github.com/org/repo.git"
    monkeypatch.setattr(pod, "_git", lambda *a: "https://github.com/org/repo.git")
    assert pod._clone_url() == "https://github.com/org/repo.git"


def test_a_remote_host_is_an_alias_or_an_address_and_nothing_is_written(tmp_path):
    # No ssh config is read or written anywhere in this path: that file belongs to the
    # person. An alias passes through untouched, so their own options apply; an address
    # gets its port and the two options that suit a machine living for an afternoon,
    # per invocation.
    from src.infra.endpoints import vllm

    assert vllm.ssh_argv("their-own-alias") == (["ssh"], "their-own-alias")
    argv, target = vllm.ssh_argv("root@1.2.3.4:11950")
    assert target == "root@1.2.3.4"
    assert argv[:3] == ["ssh", "-p", "11950"]
    # RunPod recycles ip:port between pods, so a remembered host key turns the next
    # rental into what looks like an attack.
    assert "StrictHostKeyChecking=accept-new" in argv
    assert "UserKnownHostsFile=/dev/null" in argv

    assert not hasattr(pod, "SSH_CONFIG")
    assert not hasattr(pod, "_write_ssh_alias")


def test_the_gpu_comes_from_the_model_profile_not_the_command_line(tmp_path, monkeypatch):
    # The point of ModelProfile.gpu: a catalogue id is written once per model, and both
    # provisioning paths read it. Here, the training one.
    from src.model_profile import gpu_for

    cfg = tmp_path / "arm.yaml"
    cfg.write_text('model: "Qwen/Qwen3.6-27B"\n')
    seen = {}

    def fake_provision(spec, *, name, start_script, ports=()):
        seen.update(gpu=spec.gpu, count=spec.count)
        return "podid"

    monkeypatch.setattr(pod, "provision_runpod", fake_provision)
    monkeypatch.setattr(pod, "_commit_to_run", lambda branch: ("main", "abc1234"))
    monkeypatch.setattr(pod, "_clone_url", lambda: "https://github.com/o/r.git")
    monkeypatch.setattr(pod, "_ssh_endpoint", lambda pod_id: ("1.2.3.4", 22))
    monkeypatch.setattr(pod, "_wait_for_ssh", lambda name: True)

    pod.up(name="t", train_config=str(cfg), count=2)  # naming an arm implies the clone
    assert seen["gpu"] == gpu_for("Qwen/Qwen3.6-27B", "train") == "NVIDIA H200"
    # The COUNT is not in the profile and never reaches it: how many GPUs is a decision
    # about the run, made here.
    assert seen["count"] == 2

    pod.up(name="t", train_config=str(cfg), gpu="NVIDIA B200")
    assert seen["gpu"] == "NVIDIA B200"  # an explicit ask still wins


def test_a_bare_pod_is_the_default_and_takes_the_module_gpu(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(pod, "provision_runpod",
                        lambda spec, **kw: seen.update(gpu=spec.gpu) or "podid")
    monkeypatch.setattr(pod, "_ssh_endpoint", lambda pod_id: ("1.2.3.4", 22))
    monkeypatch.setattr(pod, "_wait_for_ssh", lambda name: True)

    pod.up(name="t")  # no config, no target, no clone
    assert seen["gpu"] == pod.GPU


def test_serving_a_target_hands_the_facts_to_serve_vllm(monkeypatch):
    # `--serve` provisions too (through serve_vllm), and the pod IS the server — so
    # every serving parameter has to come from the target and its family, not from
    # whoever typed the command.
    from src.infra.endpoints import vllm
    from src.model_profile import gpu_for

    seen = {}
    monkeypatch.setattr(pod, "serve_vllm",
                        lambda base, mods, **kw: seen.update(base=base, mods=mods, **kw) or "podid")
    monkeypatch.setattr(vllm, "resolve_target", lambda t: vllm.TargetSpec(
        hf_path=t, base_model="Qwen/Qwen3.6-27B", adapter=True, mode="think",
        model_key="arm", lora_rank=64))

    out = pod.up(name="t", serve="LASR-Callum/2026-08-31-some-adapter", max_len=65536)

    assert seen["base"] == "Qwen/Qwen3.6-27B"
    assert seen["mods"] == [("arm", "LASR-Callum/2026-08-31-some-adapter")]
    assert seen["mode"] == "think"                      # inferred from the artifact
    assert seen["gpu"] == gpu_for("Qwen/Qwen3.6-27B", "inference")
    assert seen["max_len"] == 65536
    assert seen["lora_rank"] == 64
    # ODCV scores a clean 0% without the tool-call parser: the agent cannot act, and the
    # summary looks fine. Both parsers are the family's, from ModelProfile.serving.
    assert seen["tool_call_parser"] == "qwen3_xml"
    assert seen["reasoning_parser"] == "qwen3"
    assert "--endpoint" in out and "runpod down --pod podid" in out
