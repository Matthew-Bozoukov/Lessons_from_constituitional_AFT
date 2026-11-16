# ABOUTME: Offline tests for the `uv run runpod` CLI: the commit it refuses to run, the
# ABOUTME: bootstrap it renders, and the ~/.ssh/config entry it rewrites rather than repeats.

import subprocess
from types import SimpleNamespace

import pytest

from src.infra import runpod as pod


@pytest.fixture(autouse=True)
def no_real_watchdogs(monkeypatch):
    monkeypatch.setattr(pod, "start_watchdog", lambda *a, **k: None)


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
    assert "git checkout -q --detach abc1234" in script
    assert "git remote add origin https://github.com/o/r.git" in script
    assert "git fetch -q origin main" in script
    assert "uv sync" in script
    # sshd and the log server come up BEFORE the slow work, or a stall is undiagnosable.
    assert script.index("sshd") < script.index("git fetch")
    assert script.index("http.server 8080") < script.index("curl -LsSf https://astral.sh/uv")


def test_the_clone_survives_a_dir_that_already_exists():
    # `--push_env` writes /root/work/.env while the boot is still installing uv, and RunPod
    # restarts a container whose boot failed WITH its disk. `git clone` into a non-empty dir
    # fails in seconds; 2026-09-28 that made two eval pods crash-loop ~100 times each, each
    # life too short for sshd, so the pod was RUNNING with no IP and no boot log to read.
    script = pod._bootstrap(("https://github.com/o/r.git", "main", "abc1234"))
    assert "git clone" not in script
    assert "if [ ! -d .git ]; then git init -q && git remote add origin" in script
    assert "mkdir -p /root/work && cd /root/work" in script


def test_only_a_train_pod_builds_the_causal_conv1d_kernel():
    # causal-conv1d is the lock's `train` extra: a CUDA source build that needs nvcc, which
    # only KERNEL_BUILD provides. Nothing outside src/train imports it, so a pod that clones
    # the repo to DRIVE an eval on the box (`--eval --clone-repo`) syncs without it; the
    # bare sync that tried to build it is what killed both MASK pods on 2026-09-28.
    train = pod._bootstrap(("https://github.com/o/r.git", "main", "abc1234"), build_kernels=True)
    driver = pod._bootstrap(("https://github.com/o/r.git", "main", "abc1234"),
                            (["Qwen/Qwen3.6-27B"], None))
    assert "uv sync --extra train" in train and "BUILDING_CAUSAL_CONV1D" in train
    assert train.index("export CUDA_HOME=") < train.index("uv sync --extra train")
    assert "--extra train" not in driver and "CUDA_HOME" not in driver
    assert "\nuv sync\n" in driver


def test_an_eval_pod_bootstrap_installs_vllm_and_serves_nothing():
    script = pod._bootstrap(None, (["Qwen/Qwen3.6-27B", "org/2026-08-31-arm"], "hf_secret"))
    pod._check_bash(script)
    assert "git fetch" not in script and "uv sync" not in script
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
    assert "git fetch -q origin main" in script and "uv sync" in script
    assert "uv venv /workspace/vllmenv" in script
    # The serving install comes after the clone's `cd`, so it must not read the repo's uv config.
    assert script.index("uv sync") < script.index("uv pip install --no-config")
    # One READY, naming both, so the boot log says what actually finished.
    assert script.count("echo READY") == 1
    assert "echo READY abc1234 + vllm venv at /workspace/vllmenv" in script


def _vllm_pin(pyproject_path):
    import re
    import tomllib

    declared = tomllib.loads(open(pyproject_path, "rb").read().decode())["project"]["dependencies"]
    return next(d for d in declared if re.match(r"^vllm\b", d)).split(";")[0].strip()


def test_a_pod_serves_the_vllm_this_repo_pins_not_whatever_pypi_has_today():
    # Parser names, the runtime LoRA endpoint and template handling all move between
    # vLLM versions: a pod on a different one is not the server the driver was tested
    # against. The pin is the root pyproject's, and the pod's comes from the lock under it.
    from src.infra.endpoints import vllm

    spec = _vllm_pin("pyproject.toml")
    assert pod._pinned_vllm() == spec
    assert "==" in spec, "the pod pin is only as exact as the repo's own"
    name, version = spec.split("==")
    assert vllm.locked_serving_packages()[name] == version


def test_training_and_serving_share_one_exact_transformers():
    # One version wherever a model is trained or served (2026-11-16: for a day an eval pod
    # served on 5.22.2 and an on-box server on 5.14.1). Exact in the pyproject, so a re-lock
    # cannot move it; the same in the lock; and the same in what a repo-less pod installs.
    import tomllib

    from src.infra.endpoints import vllm

    declared = tomllib.loads(open("pyproject.toml", "rb").read().decode())["project"]["dependencies"]
    assert "transformers==5.14.1" in declared
    assert vllm.locked_packages()["transformers"] == "5.14.1"
    assert vllm.locked_serving_packages()["transformers"] == "5.14.1"
    # There is no second lock to drift from the first.
    assert not (vllm.REPO_LOCK.parent / "src/infra/endpoints/vllm_env").exists()


def test_the_serving_env_is_vllms_part_of_the_repo_lock_as_a_pod_would_install_it():
    from src.infra.endpoints import vllm

    serving, everything = vllm.locked_serving_packages(), vllm.locked_packages()
    assert set(vllm.SERVING_ROOTS) <= set(serving)
    assert all(everything[name] == version for name, version in serving.items())
    # Markers are read for the POD, whatever machine types `runpod up`: the linux-only GPU
    # stack is in, the trainer and the macOS/Windows-only wheels are out.
    assert {"vllm", "torch", "transformers", "tokenizers", "flashinfer-python", "nvidia-nvvm"} <= set(serving)
    assert not {"trl", "peft", "causal-conv1d", "colorama", "pywin32"} & set(serving)
    assert 150 < len(serving) < len(everything)


def _pod_scripts():
    eval_pod = pod._bootstrap(None, (["Qwen/Qwen3.6-27B"], None))
    chat_pod = pod.bootstrap_script(
        "Qwen/Qwen3.6-27B", [], hf_token=None, max_len=16384, lora_rank=64,
        max_num_seqs=32, mode="think", reasoning_parser="qwen3", tool_call_parser=None)
    return eval_pod, chat_pod


def test_a_pod_installs_every_package_at_the_version_the_serving_lock_names():
    # An eval pod has no repo and so no lock of ours: until 2026-11-16 it installed
    # `vllm==<pin> ... transformers`, and transformers 5.23.0 (released 2026-11-13) made
    # vLLM 0.26 exit at start-up on Qwen3.6. Both pod bootstraps now carry the serving part
    # of the repo's lock as exact versions and resolve nothing on the pod.
    from src.infra.endpoints import vllm

    locked = vllm.locked_serving_packages()
    for script in _pod_scripts():
        pod._check_bash(script)
        body = script.split("<<'VLLM_ENV_LOCK'\n", 1)[1].split("\nVLLM_ENV_LOCK\n", 1)[0]
        assert dict(line.split("==") for line in body.splitlines()) == locked
        install = next(ln for ln in script.splitlines() if "uv pip install" in ln)
        assert "--no-deps" in install and "-r /workspace/vllm_env.lock.txt" in install
        # On a --clone-repo pod the install runs inside the clone: without --no-config uv applies the
        # repo's own [tool.uv] constraints to the serving env and the pins are unsatisfiable; without
        # --clear a restarted container dies on the venv its previous life left (2026-11-16).
        assert "--no-config" in install
        assert f"uv venv {vllm.POD_VENV if '$VENV' not in script else '$VENV'} --python 3.12 --clear" in script
        # Nothing but that one install, and never a package named without a version.
        assert sum("pip install" in ln for ln in script.splitlines()) == 1
        assert not any(word in install.split() for word in ("transformers", "vllm", "ninja"))
    # One argument to `bash -lc`: Linux caps a single argv string at 128 KiB.
    assert all(len(script.encode()) < 64 * 1024 for script in _pod_scripts())


def test_a_served_run_records_what_the_serving_interpreter_held():
    from src.infra.endpoints import vllm

    locked, everything = vllm.locked_serving_packages(), vllm.locked_packages()
    as_locked = vllm.serving_env_record("pod:/workspace/vllmenv", dict(locked))
    assert as_locked["matches_lock"] and as_locked["differs"] == {} and as_locked["packages"] == locked
    assert as_locked["lock"] == "uv.lock"
    # A server on the box runs in the repo's own env: the rest of the lock and this repo itself
    # are installed beside the serving packages, and that is still the locked stack.
    on_box = vllm.serving_env_record("this machine", {**everything, vllm._THIS_PROJECT: "0.1.0"})
    assert on_box["matches_lock"] and on_box["differs"] == {}
    # The 2026-11-16 pod: everything as locked except a newer transformers and one stray package.
    drifted = vllm.serving_env_record("pod", {**locked, "transformers": "5.23.0", "stray": "1.0"})
    assert not drifted["matches_lock"]
    assert drifted["differs"] == {"transformers": {"installed": "5.23.0", "locked": "5.14.1"},
                                  "stray": {"installed": "1.0", "locked": None}}
    # A serving package that is missing is a difference too.
    missing = vllm.serving_env_record("pod", {n: v for n, v in locked.items() if n != "ninja"})
    assert missing["differs"] == {"ninja": {"installed": None, "locked": locked["ninja"]}}
    # Names compare as PyPI compares them, whatever case or separator the dist metadata uses.
    assert vllm._parse_installed("PyYAML==6.0.3\nhuggingface_hub==2.4.0\n\n") == {
        "pyyaml": "6.0.3", "huggingface-hub": "2.4.0"}


def test_the_serving_env_is_read_from_the_interpreter_that_runs_the_server(tmp_path):
    import sys

    from src.infra.endpoints import vllm

    where, installed = vllm.LocalExec(tmp_path).installed_packages()
    assert sys.executable in where and "pytest" in installed  # this interpreter, really queried


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

    def fake_provision(spec, *, name, start_script, ports=(), env=None):
        seen.update(gpu=spec.gpu, count=spec.count)
        return "podid"

    monkeypatch.setattr(pod, "provision_runpod", fake_provision)
    monkeypatch.setattr(pod, "_commit_to_run", lambda branch: ("main", "abc1234"))
    monkeypatch.setattr(pod, "_clone_url", lambda: "https://github.com/o/r.git")
    monkeypatch.setattr(pod, "_ssh_endpoint", lambda pod_id: ("1.2.3.4", 22))
    monkeypatch.setattr(pod, "_wait_for_ssh", lambda name: True)
    monkeypatch.setattr(pod, "wait_bootstrapped", lambda *a, **kw: True)

    pod.up(name="t", train=str(cfg), count=2)  # naming an arm implies the clone
    assert seen["gpu"] == gpu_for("Qwen/Qwen3.6-27B", "train") == "NVIDIA H200"
    # The COUNT is not in the profile and never reaches it: how many GPUs is a decision
    # about the run, made here.
    assert seen["count"] == 2

    pod.up(name="t", train=str(cfg), gpu="NVIDIA B200")
    assert seen["gpu"] == "NVIDIA B200"  # an explicit ask still wins

    # A RECIPE names no model: `--model <key>` says which, and is refused when absent.
    recipe = tmp_path / "lora.yaml"
    recipe.write_text("lora: {r: 64}\n")
    out = pod.up(name="t", train=str(recipe), model="qwen36")
    assert seen["gpu"] == "NVIDIA H200"
    assert "uv run train --config" in out and "model=qwen36" in out and "data_repo=" in out
    with pytest.raises(AssertionError, match="--model"):
        pod.up(name="t", train=str(recipe))


def test_an_eval_pod_takes_the_inference_card_and_installs_vllm_without_the_repo(monkeypatch):
    # The other half of ModelProfile.gpu. An eval pod is the cheaper card, and it holds
    # vLLM and the weights and NOTHING of this repo: run_eval owns serving over SSH.
    from src.infra.endpoints import vllm
    from src.model_profile import gpu_for

    seen = {}

    def fake_provision(spec, *, name, start_script, ports=(), env=None):
        seen.update(gpu=spec.gpu, cuda=spec.cuda, script=start_script)
        return "podid"

    monkeypatch.setattr(pod, "provision_runpod", fake_provision)
    monkeypatch.setattr(pod, "_ssh_endpoint", lambda pod_id: ("1.2.3.4", 22))
    monkeypatch.setattr(pod, "_wait_for_ssh", lambda name: True)
    monkeypatch.setattr(vllm, "resolve_target", lambda t: vllm.TargetSpec(
        hf_path=t, base_model="Qwen/Qwen3.6-27B", adapter=True, mode="think",
        model_key="arm", lora_rank=64))

    out = pod.up(name="t", eval="odcv", target="LASR-Callum/2026-08-31-some-adapter")

    assert seen["gpu"] == gpu_for("Qwen/Qwen3.6-27B", "inference", "odcv")
    # vLLM from PyPI brings a CUDA-13 torch; an older host driver dies at _cuda_init.
    assert seen["cuda"] == pod.CUDA_VERSIONS and seen["cuda"].startswith("13.0,")  # the CUDA 13 FAMILY (2026-10-30)
    script = seen["script"]
    assert f"uv venv {vllm.POD_VENV}" in script
    assert "hf download Qwen/Qwen3.6-27B" in script
    assert "hf download LASR-Callum/2026-08-31-some-adapter" in script
    # The two things an eval pod must NOT do: clone the repo, or serve anything itself.
    assert "git clone" not in script and "uv sync" not in script
    assert "api_server" not in script
    assert "--server root@1.2.3.4:22" in out and "runpod down --pod podid" in out
    assert "--name odcv --target LASR-Callum/2026-08-31-some-adapter" in out


def test_the_inference_card_is_a_model_x_eval_fact_and_the_print_says_which_entry(
        monkeypatch, capsys):
    # MASK on Qwen3.6-27B measured 3.5x faster and cheaper on an H200 (2026-09-22); ODCV
    # is unmeasured there, so it and everything else stays on the family default. The
    # eval NAME is what picks the card, and the launch print says which entry it read.
    from src.infra.endpoints import vllm

    seen = {}

    def fake_provision(spec, *, name, start_script, ports=(), env=None):
        seen.update(gpu=spec.gpu)
        return "podid"

    monkeypatch.setattr(pod, "provision_runpod", fake_provision)
    monkeypatch.setattr(pod, "_ssh_endpoint", lambda pod_id: ("1.2.3.4", 22))
    monkeypatch.setattr(pod, "_wait_for_ssh", lambda name: True)
    monkeypatch.setattr(vllm, "resolve_target", lambda t: vllm.TargetSpec(
        hf_path=t, base_model="Qwen/Qwen3.6-27B", adapter=True, mode="think",
        model_key="arm", lora_rank=64))

    pod.up(name="t", eval="mask", target="LASR-Callum/2026-08-31-some-adapter")
    assert seen["gpu"] == "NVIDIA H200"
    assert "from ModelProfile.gpu.inference[mask]" in capsys.readouterr().out

    pod.up(name="t", eval="odcv", target="LASR-Callum/2026-08-31-some-adapter")
    assert seen["gpu"] == "NVIDIA H100 80GB HBM3"
    assert "from ModelProfile.gpu.inference[default]" in capsys.readouterr().out

    # The eval is validated against the registry, and the target is required.
    with pytest.raises(AssertionError, match="no eval named 'nosuch'"):
        pod.up(name="t", eval="nosuch", target="LASR-Callum/2026-08-31-some-adapter")
    with pytest.raises(AssertionError, match="--target <hf_path>"):
        pod.up(name="t", eval="mask")


def test_the_old_eval_takes_an_hf_path_form_is_refused_with_the_new_form(monkeypatch):
    # `--eval <hf_path>` was the shape until 2026-09-22. It is caught by the slash an HF
    # id always carries, before the registry lookup would only say "no such eval".
    monkeypatch.setattr(pod, "provision_runpod", lambda spec, **kw: pytest.fail("rented"))
    with pytest.raises(AssertionError) as err:
        pod.up(name="t", eval="LASR-Callum/2026-08-31-some-adapter")
    assert ("--eval <eval> --target LASR-Callum/2026-08-31-some-adapter" in str(err.value)
            and "looks like an HF path" in str(err.value))


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

    def fake_provision(spec, *, name, start_script, ports=(), env=None):
        seen.update(gpu=spec.gpu, disk_gb=spec.disk_gb, script=start_script)
        return "podid"

    monkeypatch.setattr(pod, "provision_runpod", fake_provision)
    monkeypatch.setattr(pod, "_ssh_endpoint", lambda pod_id: ("1.2.3.4", 22))
    monkeypatch.setattr(pod, "_wait_for_ssh", lambda name: True)
    monkeypatch.setattr(vllm, "resolve_target", lambda t: vllm.TargetSpec(
        hf_path=f"org/{t}", base_model=bases[t], adapter=True, mode="think",
        model_key=t, lora_rank=64))
    monkeypatch.setattr(pod, "gpu_entry", lambda model, role, eval=None: ({
        "Qwen/Qwen3.6-27B": "NVIDIA H100 80GB HBM3", "Big/Model-500B": "NVIDIA H200",
    }[model], f"{role}[default]"))

    # Two arms over ONE base: one card, and the default disk is already right for it.
    out = pod.up(name="t", eval="odcv", target=("a", "b"))
    assert seen["gpu"] == "NVIDIA H100 80GB HBM3"
    assert seen["disk_gb"] == 200
    assert seen["script"].count("hf download") == 3      # one base + two adapters
    assert "--target a b --server" in out  # echoes what you typed, not the resolved paths

    # Add an arm whose family wants a bigger card: the pod takes the bigger one, says
    # why, and grows the disk for the second base.
    pod.up(name="t", eval="odcv", target=("a", "big"))
    assert seen["gpu"] == "NVIDIA H200"
    assert seen["disk_gb"] == 350
    warning = capsys.readouterr().out
    assert "do not agree on an inference card" in warning and "NVIDIA H200" in warning


def test_a_pod_is_for_training_or_evaluating_and_says_so(tmp_path, monkeypatch):
    # Neither shape is not a shape: a pod with no work named would rent the module
    # default card for nothing, and bill for it.
    monkeypatch.setattr(pod, "provision_runpod", lambda spec, **kw: "podid")
    with pytest.raises(AssertionError, match="--train <config> or --eval <eval> --target <hf_path>"):
        pod.up(name="t")
    cfg = tmp_path / "arm.yaml"
    cfg.write_text('model: "Qwen/Qwen3.6-27B"\n')
    with pytest.raises(AssertionError, match="not both"):
        pod.up(name="t", train=str(cfg), eval="mask",
               target="LASR-Callum/2026-08-31-some-adapter")



def _train_pod_stubs(monkeypatch, tmp_path, ready):
    """Stub a --train rental down to the boot wait; `ready` decides whether READY arrives."""
    cfg = tmp_path / "arm.yaml"
    cfg.write_text('model: "Qwen/Qwen3.6-27B"\n')
    seen = {}
    monkeypatch.setattr(pod, "provision_runpod", lambda spec, **kw: "podid")
    monkeypatch.setattr(pod, "_commit_to_run", lambda branch: ("main", "abc1234def"))
    monkeypatch.setattr(pod, "_clone_url", lambda: "https://github.com/o/r.git")
    monkeypatch.setattr(pod, "_ssh_endpoint", lambda pod_id: ("1.2.3.4", 22))
    monkeypatch.setattr(pod, "_wait_for_ssh", lambda name: True)

    def wait(pod_id, timeout_s=0, poll_s=0, marker="READY"):
        seen["marker"] = marker
        return ready
    monkeypatch.setattr(pod, "wait_bootstrapped", wait)
    return cfg, seen


def test_a_train_pod_is_handed_back_only_once_its_boot_says_ready_for_that_commit(monkeypatch, tmp_path):
    # 2026-09-28: `uv run train` launched before the boot's own `uv sync` finished started a
    # second sync without the boot's CUDA_HOME and died building causal-conv1d. `up` now
    # waits for the exact commit's READY line before returning the launch command.
    cfg, seen = _train_pod_stubs(monkeypatch, tmp_path, ready=True)
    out = pod.up(name="t", train=str(cfg))
    assert seen["marker"] == "READY abc1234def"
    assert "ready to train" in out


def test_a_train_pod_whose_boot_never_finishes_is_diagnosed_and_left_up(monkeypatch, tmp_path):
    # A slow boot is diagnosed, never torn down: the error names the step the boot reached
    # and its error lines, and the pod survives for someone to read it.
    cfg, _ = _train_pod_stubs(monkeypatch, tmp_path, ready=False)
    log = "+ echo CLONING\nCLONING\n+ uv sync\nBUILDING_CAUSAL_CONV1D\nFileNotFoundError: /usr/local/cuda/bin/nvcc\n"
    monkeypatch.setattr(pod.requests, "get", lambda *a, **kw: SimpleNamespace(text=log))
    torn = []
    monkeypatch.setattr(pod, "teardown", lambda p: torn.append(p))
    with pytest.raises(RuntimeError) as err:
        pod.up(name="t", train=str(cfg))
    msg = str(err.value)
    assert "left UP" in msg and "building causal-conv1d" in msg and "nvcc" in msg
    assert torn == []


def test_wait_bootstrapped_matches_the_given_marker_not_any_ready(monkeypatch):
    # SERVE_READY or an older commit's READY line must not satisfy a wait for READY <sha>.
    logs = iter(["+ echo SERVE_READY\nREADY 0ld5ha", "READY abc1234def\n"])
    monkeypatch.setattr(pod.requests, "get", lambda *a, **kw: SimpleNamespace(text=next(logs)))
    monkeypatch.setattr(pod.time, "sleep", lambda s: None)
    assert pod.wait_bootstrapped("p", timeout_s=60, poll_s=0, marker="READY abc1234def")


def test_a_run_pod_runs_its_command_after_ready_and_deletes_itself_only_on_success():
    """`up --run`: the pod starts the train command itself, for a driver with no SSH."""
    script = pod._bootstrap(("https://github.com/o/r.git", "main", "abc1234"), build_kernels=True,
                            then="uv run train --config configs/train/sft.yaml seed=0")
    pod._check_bash(script)
    ready = script.index("echo READY abc1234")
    assert ready < script.index("set +e") < script.index("uv run train --config") \
        < script.index('echo "RUN_EXIT $code"') < script.index("sleep infinity")
    assert "> /workspace/train.log 2>&1" in script
    assert 'if [ "$code" = 0 ]; then (sleep 600; while :; do curl -sf -X DELETE' in script
    with pytest.raises(AssertionError, match="needs the clone"):
        pod._bootstrap(None, (["org/m"], None), then="uv run train")


def test_up_run_needs_no_ssh_and_hands_the_pod_its_hf_env(tmp_path, monkeypatch):
    seen = {}

    def fake_provision(spec, *, name, start_script, ports=(), env=None):
        seen.update(script=start_script, env=dict(env or {}))
        return "podid"

    def no_ssh(*_a, **_k):
        raise AssertionError("a --run pod must not be reached over SSH")

    monkeypatch.setattr(pod, "provision_runpod", fake_provision)
    monkeypatch.setattr(pod, "_commit_to_run", lambda branch: ("nika/x", "abc1234"))
    monkeypatch.setattr(pod, "_clone_url", lambda: "https://github.com/o/r.git")
    monkeypatch.setattr(pod, "_ssh_endpoint", no_ssh)
    monkeypatch.setattr(pod, "_wait_for_ssh", no_ssh)
    monkeypatch.setattr(pod, "wait_bootstrapped", lambda *a, **kw: True)
    monkeypatch.setenv("HF_TOKEN", "hf_test_value")
    monkeypatch.setenv("HF_ORG", "org")
    recipe = tmp_path / "sft.yaml"
    recipe.write_text("lora: {r: 64}\n")
    cmd = "uv run train --config configs/train/sft.yaml model=qwen36 data_repo=org/m seed=0"
    out = pod.up(name="nika-t", train=str(recipe), model="qwen36", push_env=True, run=cmd)
    assert cmd in seen["script"] and "RUN_EXIT" in seen["script"]
    assert seen["env"]["HF_TOKEN"] == "hf_test_value" and seen["env"]["HF_ORG"] == "org"
    assert "hf_test_value" not in seen["script"], "the token never enters the logged script"
    assert "podid-8080.proxy.runpod.net/train.log" in out and "runpod down --pod podid" in out
    with pytest.raises(AssertionError, match="TRAIN pod"):
        pod.up(name="nika-t", eval="mask", target="org/2026-08-31-x", run=cmd)
