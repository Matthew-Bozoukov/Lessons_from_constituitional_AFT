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


def test_the_ssh_config_entry_is_replaced_not_repeated(tmp_path, monkeypatch):
    config = tmp_path / "config"
    config.write_text("Host somewhere-else\n    HostName 10.0.0.1\n")
    monkeypatch.setattr(pod, "SSH_CONFIG", config)

    pod._write_ssh_alias("jamie-par716", "1.2.3.4", 11950, "pod1")
    pod._write_ssh_alias("jamie-par716", "5.6.7.8", 22000, "pod2")
    text = config.read_text()

    # RunPod hands out a new ip:port for every pod. Two blocks of the same name would
    # leave ssh using the first one, sending the next run to a machine that is gone.
    assert text.count("Host jamie-par716") == 1
    assert "5.6.7.8" in text and "1.2.3.4" not in text
    assert "Port 22000" in text
    assert "Host somewhere-else" in text  # the rest of the file is untouched
    assert config.stat().st_mode & 0o777 == 0o600


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
    monkeypatch.setattr(pod, "_write_ssh_alias", lambda *a: None)
    monkeypatch.setattr(pod, "_wait_for_ssh", lambda name: True)

    pod.up(name="t", train_config=str(cfg), count=2)
    assert seen["gpu"] == gpu_for("Qwen/Qwen3.6-27B", "train") == "NVIDIA H200"
    # The COUNT is not in the profile and never reaches it: how many GPUs is a decision
    # about the run, made here.
    assert seen["count"] == 2

    pod.up(name="t", train_config=str(cfg), gpu="NVIDIA B200")
    assert seen["gpu"] == "NVIDIA B200"  # an explicit ask still wins


def test_a_pod_for_no_particular_model_falls_back_to_the_module_default(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(pod, "provision_runpod",
                        lambda spec, **kw: seen.update(gpu=spec.gpu) or "podid")
    monkeypatch.setattr(pod, "_ssh_endpoint", lambda pod_id: ("1.2.3.4", 22))
    monkeypatch.setattr(pod, "_write_ssh_alias", lambda *a: None)
    monkeypatch.setattr(pod, "_wait_for_ssh", lambda name: True)

    pod.up(name="t", clone_repo=False)
    assert seen["gpu"] == pod.GPU
