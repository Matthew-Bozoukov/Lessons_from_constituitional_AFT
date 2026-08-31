#!/usr/bin/env python3
# ABOUTME: THE way a GPU box happens: `up` rents a pod and clones this repo at the exact
# ABOUTME: commit you are on, so `ssh <name> uv run ...` runs code that is already on origin.

"""Rent a GPU pod that holds this repo at your current commit.

    uv run python scripts/infra/runpod.py up --name jamie-par716 --gpu 'H200 SXM' --count 2
    ssh jamie-par716 'cd /root/work && uv run torchrun --nproc_per_node=2 \\
        scripts/train/train_lora.py --config configs/train/lora_qwen36_t2_9284_par716.yaml'
    uv run python scripts/infra/runpod.py down --pod <id>

The pod clones over anonymous HTTPS -- this repository is public -- and checks out the
SHA you are sitting on, so it carries no credentials and no tarball, and a run's
`git_sha` is the real commit rather than `nogit`. That is why `up` REFUSES to provision
when HEAD is not on origin: what runs on a paid box must be readable by everyone on the
team at a name they can fetch. It will not push for you; pushing is a decision.

`up` writes an entry into `~/.ssh/config`, so the pod's name is usable everywhere a host
is: plain `ssh <name>`, and `--server <name>` for `uv run evals` / `uv run chat`.

Teardown is yours: this process exits after provisioning, so it cannot hold a watchdog
(`src.infra.runpod.start_watchdog` binds a pod to a LIVE parent and would kill this one
seconds after `up` returned). Run `down` when the work is finished, and `pods` to see
what is still billing.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import fire
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.infra.runpod import (  # noqa: E402
    GPU, ProvisionSpec, active_pods, call, provision_runpod, terminate,
)
from src.model_profile import gpu_for  # noqa: E402

DEFAULT_IMAGE = "runpod/pytorch:0.7.0-dev-cu1281-torch271-ubuntu2204"
WORKDIR = "/root/work"
SSH_CONFIG = Path("~/.ssh/config").expanduser()


# --------------------------------------------------------------------------------------
# what gets cloned
# --------------------------------------------------------------------------------------

def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def _clone_url() -> str:
    """The origin URL in a form a credential-less pod can clone."""
    url = _git("remote", "get-url", "origin")
    # SSH remotes (git@github.com:org/repo.git) need a key the pod does not have; the
    # HTTPS form of a public repo needs nothing at all.
    m = re.match(r"^git@([^:]+):(.+)$", url)
    return f"https://{m.group(1)}/{m.group(2)}" if m else url


def _commit_to_run(branch: str | None) -> tuple[str, str]:
    """The (branch, sha) the pod will check out — or an explanation of why it cannot.

    Three refusals, all of them the same mistake in different clothes: running code on a
    paid box that nobody else can read back.

    * uncommitted changes to tracked files — the pod would silently run the last commit,
      and the run's `git_sha` would name code that is not what you were looking at;
    * a branch that is not on origin at all;
    * a HEAD that origin has never seen.

    None of these is fixed here. `git push` is a decision about what other people will
    fetch, and a tool that pushes for you makes it silently.
    """
    branch = branch or _git("rev-parse", "--abbrev-ref", "HEAD")
    sha = _git("rev-parse", branch)

    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--"]).returncode != 0
    assert not dirty, (
        "tracked files have uncommitted changes: the pod would clone HEAD and run code "
        "that is not what you are looking at. Commit (or stash) first.\n"
        "  (untracked files are ignored — they are not in the clone either way)")

    # Read-only: this asks origin what it has, it does not change anything here or there.
    remote = subprocess.run(["git", "fetch", "--quiet", "origin", branch],
                            capture_output=True, text=True)
    assert remote.returncode == 0, (
        f"origin has no branch {branch!r} ({remote.stderr.strip()}).\n"
        f"  Push it first:  git push -u origin {branch}")
    on_origin = subprocess.run(
        ["git", "merge-base", "--is-ancestor", sha, "FETCH_HEAD"]).returncode == 0
    assert on_origin, (
        f"{sha[:8]} is not on origin/{branch} — the pod clones from origin and would run "
        "an older commit.\n"
        f"  Push it first:  git push origin {branch}")
    return branch, sha


# --------------------------------------------------------------------------------------
# the pod
# --------------------------------------------------------------------------------------

def _bootstrap(clone: tuple[str, str, str] | None) -> str:
    """Pod startup script: sshd and a log server first, then uv, the clone, and `uv sync`.

    Order is the lesson from every other bootstrap in this repo (see
    `src.infra.runpod.bootstrap_script`): SSH and the :8080 log server come up BEFORE
    anything slow, so a stall is diagnosable from a browser instead of being a pod that
    bills in silence. `sleep infinity` at the end keeps the container alive for the work
    you will drive over SSH.

    `clone` is (url, branch, sha) or None for a bare pod — uv and sshd, nothing else.
    """
    if clone:
        url, branch, sha = clone
        fetch = f"""echo CLONING
git clone --branch {branch} {url} {WORKDIR}
cd {WORKDIR}
# Detached at the exact SHA, never at the branch tip: the branch can move while the pod
# boots, and a run whose code silently differs from the commit you asked for is the
# failure this whole path exists to remove.
git checkout --detach {sha}
uv sync
echo READY {sha}
"""
    else:
        fetch = "echo READY bare pod, no repo cloned"
    return f"""mkdir -p /workspace
exec > >(tee -a /workspace/boot.log) 2>&1
set -euxo pipefail
mkdir -p ~/.ssh && [ -n "${{PUBLIC_KEY:-}}" ] && echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys 2>/dev/null || true
(apt-get update -qq && apt-get install -y -qq openssh-server git >/dev/null 2>&1; \
 mkdir -p /run/sshd && /usr/sbin/sshd -D &) || echo "sshd unavailable"
(cd /workspace && nohup python3 -m http.server 8080 </dev/null >/dev/null 2>&1 &) || true
# The HF cache belongs on the container disk, not the (unmounted) volume: a 55GB base
# model into / fills the root filesystem and the run dies somewhere unrelated.
export HF_HOME=/workspace/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled
curl -LsSf https://astral.sh/uv/install.sh | sh
# Into /usr/local/bin, not just ~/.local/bin: a non-interactive `ssh <pod> uv run ...`
# sources no profile, so a uv that lives only on the login PATH is a uv that command
# cannot find.
install -m 0755 ~/.local/bin/uv /usr/local/bin/uv
install -m 0755 ~/.local/bin/uvx /usr/local/bin/uvx 2>/dev/null || true
{fetch}
sleep infinity
"""


def _check_bash(script: str) -> None:
    """Refuse to rent a pod on a script bash cannot parse.

    The same guard as `runpod.validate_bootstrap`, minus its vLLM-specific assertions: a
    syntax error here is a container that boots, fails on line 2 and then bills at GPU
    rates doing nothing until somebody looks.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        result = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
    finally:
        os.unlink(path)
    assert result.returncode == 0, f"pod bootstrap is not valid bash:\n{result.stderr}"


def _ssh_endpoint(pod_id: str, timeout_s: int = 420) -> tuple[str, int]:
    """Poll until RunPod publishes the pod's public IP and its mapped port 22."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        info = call("GET", f"/pods/{pod_id}")
        ip = info.get("publicIp")
        port = (info.get("portMappings") or {}).get("22")
        if ip and port:
            return str(ip), int(port)
        time.sleep(10)
    raise SystemExit(
        f"pod {pod_id} published no SSH endpoint within {timeout_s}s — it is still "
        f"BILLING.\n  uv run python scripts/infra/runpod.py down --pod {pod_id}")


def _write_ssh_alias(name: str, ip: str, port: int, pod_id: str) -> None:
    """Add (or refresh) the `~/.ssh/config` entry for this pod.

    An entry rather than a printed `ssh -p ...` line, because everything downstream takes
    a HOST: `SshExec` runs `ssh <host> <cmd>`, and `--server` on evals and chat is that
    same string. Rewritten in place under a marked block, since RunPod hands out a new
    ip/port for every pod and a stale entry of the same name would silently send the next
    run to a machine that no longer exists.
    """
    start, end = f"# >>> lasr pod {name} >>>", f"# <<< lasr pod {name} <<<"
    block = "\n".join([
        start,
        f"# pod {pod_id}, written by scripts/infra/runpod.py",
        f"Host {name}",
        f"    HostName {ip}",
        f"    User root",
        f"    Port {port}",
        # Pods are ephemeral and RunPod recycles ip:port pairs, so a remembered host key
        # is a login that fails for a reason that looks like a break-in warning.
        "    StrictHostKeyChecking accept-new",
        "    UserKnownHostsFile /dev/null",
        "    ServerAliveInterval 30",
        end,
        "",
    ])
    SSH_CONFIG.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    text = SSH_CONFIG.read_text() if SSH_CONFIG.exists() else ""
    if start in text and end in text:
        head, rest = text.split(start, 1)
        text = head + block + rest.split(end, 1)[1].lstrip("\n")
    else:
        text = (text.rstrip("\n") + "\n\n" if text.strip() else "") + block
    SSH_CONFIG.write_text(text)
    SSH_CONFIG.chmod(0o600)


def _wait_for_ssh(name: str, timeout_s: int = 300) -> bool:
    """True once the pod answers SSH. Its sshd starts before the slow work, so this is quick."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                           name, "true"], capture_output=True).returncode == 0:
            return True
        time.sleep(10)
    return False


# --------------------------------------------------------------------------------------
# the CLI
# --------------------------------------------------------------------------------------

def up(name: str, train_config: str | None = None, model: str | None = None,
       gpu: str | None = None, count: int = 1, clone_repo: bool = True,
       branch: str | None = None, disk_gb: int = 200, cloud: str = "SECURE",
       image: str = DEFAULT_IMAGE, countries: str = "", push_env: bool = False) -> str:
    """Rent a pod and clone this repo into it at your current commit.

    Args:
        name: Pod name AND the `~/.ssh/config` host it is reachable at. The RunPod
            account is shared, so prefix it with who you are.
        train_config: The arm you are about to train. Its `model:` picks the GPU from
            `ModelProfile.gpu["train"]`, so the box matches the run without anyone
            retyping a catalogue id — and it is the same file you pass to the trainer.
        model: Base model id, when you want a training box without naming a config.
        gpu: RunPod catalogue id, overriding the profile. Needed only for a family with no
            profile, or to deviate deliberately: the profile records what the family was
            MEASURED to need (Qwen3.6-27B trains on H200 because an H100 80GB OOMs 7.36
            GiB short on a 1x8k step).
        count: GPUs on the pod — a decision about the RUN, not about the model, which is
            why no profile states one. `torchrun --nproc_per_node=<count>` is what uses
            them, and the command `up` prints already carries this number.
        clone_repo: Clone this repo at your current commit (the point of the script).
            `--clone_repo=False` rents a bare pod with uv and sshd and nothing else --
            for serving, or for work whose code you will put there yourself. The
            commit checks below only apply when something is being cloned.
        branch: Branch to clone. Defaults to the one you are on.
        disk_gb: Container disk. A 27B base model plus its HF cache is ~150GB.
        cloud: SECURE or COMMUNITY.
        image: Container image.
        countries: Comma-separated placement codes; "" is anywhere.
        push_env: Write HF_TOKEN and HF_ORG (nothing else) to the pod's .env, so a run
            ON the pod can push its adapter. Off by default: it is a deliberate act to
            put a credential on a rented machine.

    Returns:
        The pod id, the host name to ssh to, and the commands to run and to tear down.
    """
    assert not (train_config and model), "give --train_config or --model, not both"
    if train_config:
        model = str(OmegaConf.load(train_config).model)
    profile_gpu = gpu_for(model, "train") if model else None
    gpu = gpu or profile_gpu or GPU

    clone = None
    if clone_repo:
        branch, sha = _commit_to_run(branch)
        clone = (_clone_url(), branch, sha)
    source = (f"{model} trains here (ModelProfile.gpu)" if gpu == profile_gpu
              else f"asked for; {model} states none" if model
              else "no model named, so the module default")
    print(f">>> {count}x {gpu} ({cloud}) — {source}")
    print(f">>> cloning {clone[0]} @ {clone[1]} {clone[2][:8]}" if clone
          else ">>> bare pod: no repo cloned")

    script = _bootstrap(clone)
    _check_bash(script)
    pod_id = provision_runpod(
        ProvisionSpec(gpu=gpu, count=count, disk_gb=disk_gb, cloud=cloud, image=image,
                      cuda="", countries=countries),
        name=name,
        start_script=script,
        ports=("8080/http", "22/tcp"),
    )
    print(f">>> pod {pod_id} — BILLING NOW")
    ip, port = _ssh_endpoint(pod_id)
    _write_ssh_alias(name, ip, port, pod_id)
    reachable = _wait_for_ssh(name)

    if push_env and reachable:
        from src.infra.endpoints.vllm import SshExec

        SshExec(name, port=8000, workdir=WORKDIR).push_hf_env(Path(".env"))

    return "\n".join([
        f"pod:       {pod_id}",
        f"host:      {name}  ({ip}:{port}, written to ~/.ssh/config)",
        f"boot log:  https://{pod_id}-8080.proxy.runpod.net/boot.log",
        "ssh:       " + ("ready" if reachable else "not answering yet — watch the boot log"),
        "",
        "The boot log says READY when the clone and `uv sync` have finished. Then:"
        if clone else "Nothing is checked out on it; the boot log says READY when uv is in.",
        (f"  ssh {name} 'cd {WORKDIR} && uv run torchrun --nproc_per_node={count} "
         "scripts/train/train_lora.py --config configs/train/<arm>.yaml'") if clone
        else f"  ssh {name}",
        "",
        "IT BILLS UNTIL YOU RUN THIS:",
        f"  uv run python scripts/infra/runpod.py down --pod {pod_id}",
    ])


def status(pod: str) -> str:
    """Report the pod's state and the tail of its boot log."""
    import requests

    info = call("GET", f"/pods/{pod}")
    lines = [f"status:  {info.get('desiredStatus')}  ({info.get('name')})",
             f"gpu:     {info.get('gpuCount')}x  ${info.get('costPerHr')}/hr",
             f"created: {info.get('createdAt')}"]
    try:
        r = requests.get(f"https://{pod}-8080.proxy.runpod.net/boot.log", timeout=15)
        if r.ok:
            tail = "\n".join(r.text.strip().splitlines()[-5:])
            lines += ["--- boot.log tail ---", tail]
    except requests.RequestException:
        lines.append("boot log not reachable yet")
    return "\n".join(lines)


def pods() -> str:
    """Every pod on the SHARED account, so nothing is left billing unnoticed."""
    rows = active_pods()
    if not rows:
        return "no active pods"
    return "\n".join(
        f"{p.get('id')}  {p.get('gpuCount')}x  ${p.get('costPerHr')}/hr  "
        f"{p.get('desiredStatus')}  {p.get('name')}  (since {p.get('createdAt')})"
        for p in rows)


def down(pod: str) -> str:
    """Terminate the pod, verified against the API, and report what is still running."""
    gone = terminate(pod)
    rest = active_pods()
    return "\n".join(
        [f"{pod}: {'terminated' if gone else 'STILL LISTED — check the console'}",
         f"{len(rest)} pod(s) still active on the account"]
        + [f"  {p.get('id')}  {p.get('name')}  ${p.get('costPerHr')}/hr" for p in rest])


if __name__ == "__main__":
    fire.Fire({"up": up, "status": status, "pods": pods, "down": down})
