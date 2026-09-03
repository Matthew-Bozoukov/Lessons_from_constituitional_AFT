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


def test_an_eval_pod_bootstrap_installs_vllm_and_serves_nothing():
    script = pod._bootstrap(None, (["Qwen/Qwen3.6-27B", "org/2026-08-31-arm"], "hf_secret"))
    pod._check_bash(script)
    assert "git clone" not in script and "uv sync" not in script
    assert "sshd" in script  # still reachable; that is what makes it useful
    # sshd and the log server come up BEFORE the slow work, or a stall is undiagnosable.
    assert script.index("sshd") < script.index("uv pip install")
    # The token is exported with xtrace OFF: /workspace/boot.log is world-readable on :8080.
    assert "set +x\nexport HF_TOKEN=hf_secret\nset -x" in script
    # It prepares a server; it never starts one. run_eval owns that.
    assert "api_server" not in script


def test_a_pod_with_neither_the_code_nor_the_weights_is_refused():
    with pytest.raises(AssertionError, match="billing for nothing"):
        pod._bootstrap(None)


def test_an_eval_pod_can_also_carry_the_repo_so_the_eval_runs_on_the_box():
    # `--eval --clone-repo`: both stacks, so the pod can be driven from here over a
    # tunnel OR on the box itself. Both halves have to survive being in one script.
    script = pod._bootstrap(("https://github.com/o/r.git", "main", "abc1234"),
                            (["Qwen/Qwen3.6-27B"], None))
    pod._check_bash(script)
    assert "git clone" in script and "uv sync" in script
    assert "uv venv /workspace/vllmenv" in script
    # One READY, naming both, so the boot log says what actually finished.
    assert script.count("echo READY") == 1
    assert "echo READY abc1234 + vllm venv at /workspace/vllmenv" in script


def test_a_pod_serves_the_vllm_this_repo_pins_not_whatever_pypi_has_today():
    # Parser names, the runtime LoRA endpoint and template handling all move between
    # vLLM versions: a pod on a different one is not the server the driver was tested
    # against. Both pod bootstraps read the SAME pin, from pyproject.
    import re
    import tomllib

    declared = tomllib.loads(open("pyproject.toml", "rb").read().decode())["project"]["dependencies"]
    spec = next(d for d in declared if re.match(r"^vllm\b", d)).split(";")[0].strip()
    assert pod._pinned_vllm() == spec
    assert "==" in spec, "the pod pin is only as exact as pyproject's own"

    eval_pod = pod._bootstrap(None, (["Qwen/Qwen3.6-27B"], None))
    chat_pod = pod.bootstrap_script(
        "Qwen/Qwen3.6-27B", [], hf_token=None, max_len=16384, lora_rank=64,
        max_num_seqs=32, mode="think", reasoning_parser="qwen3", tool_call_parser=None)
    for script in (eval_pod, chat_pod):
        assert f"'{spec}'" in script
        assert " -q vllm " not in script  # never the unpinned name


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

    pod.up(name="t", train=str(cfg), count=2)  # naming an arm implies the clone
    assert seen["gpu"] == gpu_for("Qwen/Qwen3.6-27B", "train") == "NVIDIA H200"
    # The COUNT is not in the profile and never reaches it: how many GPUs is a decision
    # about the run, made here.
    assert seen["count"] == 2

    pod.up(name="t", train=str(cfg), gpu="NVIDIA B200")
    assert seen["gpu"] == "NVIDIA B200"  # an explicit ask still wins


def test_an_eval_pod_takes_the_inference_card_and_installs_vllm_without_the_repo(monkeypatch):
    # The other half of ModelProfile.gpu. An eval pod is the cheaper card, and it holds
    # vLLM and the weights and NOTHING of this repo: run_eval owns serving over SSH.
    from src.infra.endpoints import vllm
    from src.model_profile import gpu_for

    seen = {}

    def fake_provision(spec, *, name, start_script, ports=()):
        seen.update(gpu=spec.gpu, cuda=spec.cuda, script=start_script)
        return "podid"

    monkeypatch.setattr(pod, "provision_runpod", fake_provision)
    monkeypatch.setattr(pod, "_ssh_endpoint", lambda pod_id: ("1.2.3.4", 22))
    monkeypatch.setattr(pod, "_wait_for_ssh", lambda name: True)
    monkeypatch.setattr(vllm, "resolve_target", lambda t: vllm.TargetSpec(
        hf_path=t, base_model="Qwen/Qwen3.6-27B", adapter=True, mode="think",
        model_key="arm", lora_rank=64))

    out = pod.up(name="t", eval="LASR-Callum/2026-08-31-some-adapter")

    assert seen["gpu"] == gpu_for("Qwen/Qwen3.6-27B", "inference")
    # vLLM from PyPI brings a CUDA-13 torch; an older host driver dies at _cuda_init.
    assert seen["cuda"] == "13.0"
    script = seen["script"]
    assert f"uv venv {vllm.POD_VENV}" in script
    assert "hf download Qwen/Qwen3.6-27B" in script
    assert "hf download LASR-Callum/2026-08-31-some-adapter" in script
    # The two things an eval pod must NOT do: clone the repo, or serve anything itself.
    assert "git clone" not in script and "uv sync" not in script
    assert "api_server" not in script
    assert "--server root@1.2.3.4:22" in out and "runpod down --pod podid" in out


def test_an_eval_ladder_is_one_pod_sized_for_the_biggest_arm_on_it(monkeypatch, capsys):
    # Plumbing for parallelised evals, which do not exist yet — see the note in `up`.
    # Nothing should pass several targets today (run_eval's arms are sequential and vLLM
    # takes one GPU), but the sizing rules it would need are here and tested: the card
    # fits every arm, the disk holds every base, and families disagreeing gets SAID
    # rather than silently resolved, because half the ladder would then be running on a
    # card nobody measured it on.
    from src.infra.endpoints import vllm

    bases = {"a": "Qwen/Qwen3.6-27B", "b": "Qwen/Qwen3.6-27B", "big": "Big/Model-500B"}
    seen = {}

    def fake_provision(spec, *, name, start_script, ports=()):
        seen.update(gpu=spec.gpu, disk_gb=spec.disk_gb, script=start_script)
        return "podid"

    monkeypatch.setattr(pod, "provision_runpod", fake_provision)
    monkeypatch.setattr(pod, "_ssh_endpoint", lambda pod_id: ("1.2.3.4", 22))
    monkeypatch.setattr(pod, "_wait_for_ssh", lambda name: True)
    monkeypatch.setattr(vllm, "resolve_target", lambda t: vllm.TargetSpec(
        hf_path=f"org/{t}", base_model=bases[t], adapter=True, mode="think",
        model_key=t, lora_rank=64))
    monkeypatch.setattr(pod, "gpu_for", lambda model, role: {
        "Qwen/Qwen3.6-27B": "NVIDIA H100 80GB HBM3", "Big/Model-500B": "NVIDIA H200",
    }[model])

    # Two arms over ONE base: one card, and the default disk is already right for it.
    out = pod.up(name="t", eval=("a", "b"))
    assert seen["gpu"] == "NVIDIA H100 80GB HBM3"
    assert seen["disk_gb"] == 200
    assert seen["script"].count("hf download") == 3      # one base + two adapters
    assert "--target a b --server" in out  # echoes what you typed, not the resolved paths

    # Add an arm whose family wants a bigger card: the pod takes the bigger one, says
    # why, and grows the disk for the second base.
    pod.up(name="t", eval=("a", "big"))
    assert seen["gpu"] == "NVIDIA H200"
    assert seen["disk_gb"] == 350
    warning = capsys.readouterr().out
    assert "do not agree on an inference card" in warning and "NVIDIA H200" in warning


def test_a_pod_is_for_training_or_evaluating_and_says_so(tmp_path, monkeypatch):
    # Neither shape is not a shape: a pod with no work named would rent the module
    # default card for nothing, and bill for it.
    monkeypatch.setattr(pod, "provision_runpod", lambda spec, **kw: "podid")
    with pytest.raises(AssertionError, match="--train <config> or --eval <hf_path>"):
        pod.up(name="t")
    cfg = tmp_path / "arm.yaml"
    cfg.write_text('model: "Qwen/Qwen3.6-27B"\n')
    with pytest.raises(AssertionError, match="not both"):
        pod.up(name="t", train=str(cfg), eval="LASR-Callum/2026-08-31-some-adapter")
