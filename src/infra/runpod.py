# ABOUTME: The one place this repo rents a GPU: RunPod REST client, `provision_runpod` (rent a
# ABOUTME: pod running any script) and `serve_vllm` (rent one serving a base + LoRA modules).

"""RunPod pods: rent, serve, and above all tear down.

`provision_runpod(spec, name=, start_script=)` is the only function in the repo that creates a
pod, and it knows nothing about what the pod runs. `serve_vllm` is the vLLM layer on top of it;
a training or harness driver supplies its own `start_script` instead. What to rent comes from a
`ProvisionSpec` built from a config `provision:` block, never from a constant at a call site --
so which GPU a run used is part of its record.


The serving pod boots credential-light (an HF token only when one is given, never the RunPod
key), installs vLLM into a Python 3.12 venv, pulls the base model and every adapter, pins
the thinking mode into the chat template exactly as src/infra/endpoints/vllm.pin_template
does, and serves `base` + one LoRA module per adapter on :8000, published through the
proxy at https://<pod>-8000.proxy.runpod.net/v1. :8080 serves the boot log so "still
downloading" can be told from "dead" from a browser.

Everything a pod costs is time on a billing meter, so this module also owns the three
ways one is torn down: `terminate` (verified against the API, not fire-and-forget), the
detached `watchdog` process that terminates a pod when the process that launched it is
gone or a lifetime cap passes (CLAUDE.md "Paid infrastructure": never rely on the
orchestrator surviving), and `orphans` for the startup sweep.

    python -m src.infra.runpod watchdog <pod_id> <parent_pid> <max_lifetime_s> <log>

The bottom of this file is the `uv run runpod` CLI (`up`/`status`/`pods`/`down`): the one
command that gets a person a GPU box holding this repo at this commit.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from dataclasses import dataclass, fields
from typing import Any, Callable

import requests

from src.infra.endpoints.vllm import POD_SSH_CONFIG, pin_prefix
from src.model_profile import gpu_for

REST = "https://rest.runpod.io/v1"
IMAGE = "runpod/pytorch:0.7.0-dev-cu1281-torch271-ubuntu2204"
GPU = "NVIDIA H100 80GB HBM3"
# Pods this repo's chat tool provisions carry this prefix, which is the ONLY thing the
# sweep and the watchdog will ever terminate: a teammate's pod is reported, never touched.
CHAT_POD_PREFIX = "chat-"


def _key() -> str:
    """Return the RunPod API key, or explain how to get a working one."""
    from dotenv import load_dotenv

    load_dotenv()
    key = os.environ.get("RUNPOD_API_KEY", "")
    if not key:
        raise RuntimeError("RUNPOD_API_KEY is not set. Put it in .env or export it.")
    return key


def call(method: str, path: str, **kwargs: Any) -> Any:
    """Call the RunPod REST API, turning an auth failure into an actionable message."""
    resp = requests.request(
        method,
        f"{REST}{path}",
        headers={
            "Authorization": f"Bearer {_key()}",
            "Content-Type": "application/json",
        },
        timeout=60,
        **kwargs,
    )
    if resp.status_code == 401:
        raise RuntimeError(
            "RunPod rejected the API key (401). Keys are created at "
            "https://console.runpod.io/user/settings -> API Keys, and must have WRITE "
            "permission to create pods - a read-only key 401s on every endpoint. Check the "
            "key belongs to the account holding your credit."
        )
    resp.raise_for_status()
    return resp.json() if resp.content else {}


def endpoint_url(pod_id: str) -> str:
    return f"https://{pod_id}-8000.proxy.runpod.net/v1"


def boot_log_url(pod_id: str) -> str:
    return f"https://{pod_id}-8080.proxy.runpod.net/boot.log"


# --- the serving pod ------------------------------------------------------------------------


@dataclass(frozen=True)
class ProvisionSpec:
    """What to rent. Every field a caller might reasonably vary, none of them a constant.

    Built from a config `provision:` block (`ProvisionSpec.from_config`) so the GPU choice is
    part of the scientific record of a run rather than a default buried in this module. `gpu`
    must match RunPod's catalogue id exactly -- `gpu_price` is the cheap way to check one
    before renting it.
    """
    gpu: str = GPU
    count: int = 1
    cloud: str = "SECURE"
    disk_gb: int = 200
    cuda: str = "13.0"          # "" = no constraint; only the vLLM image needs CUDA 13
    countries: str = ""         # comma-separated placement codes; "" = anywhere
    image: str = IMAGE
    max_hours: float = 6.0
    pubkey_path: str = "~/.ssh/id_ed25519.pub"

    @classmethod
    def from_config(cls, cfg: Any | None) -> "ProvisionSpec":
        """Build from a config `provision:` block (dict / OmegaConf); unknown keys are an error."""
        if not cfg:
            return cls()
        d = dict(cfg)
        unknown = set(d) - {f.name for f in fields(cls)}
        assert not unknown, (
            f"unknown provision keys {sorted(unknown)}; allowed: "
            f"{sorted(f.name for f in fields(cls))}")
        return cls(**d)


def provision_runpod(
    spec: ProvisionSpec,
    *,
    name: str,
    start_script: str,
    env: dict[str, str] | None = None,
    ports: tuple[str, ...] = ("8000/http", "8080/http", "22/tcp"),
) -> str:
    """Rent a pod and return its id. BILLS FROM THIS MOMENT until `terminate`.

    The one place in the repo that creates a RunPod pod. Everything specific to what the pod
    then DOES belongs in `start_script`, which the caller builds -- `serve_vllm` for a serving
    pod, the training driver for a training one. That split is why this function knows nothing
    about vLLM, adapters or thinking modes.

    Callers must pair this with teardown they do not rely on their own process to run
    (CLAUDE.md "Paid infrastructure"): `start_watchdog` plus a `terminate` in a `finally`.
    """
    env = dict(env or {})
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    pubkey_file = Path(spec.pubkey_path).expanduser()
    if pubkey_file.exists():
        pubkey = pubkey_file.read_text().strip()
        assert pubkey.startswith("ssh-"), f"not an ssh public key: {pubkey_file}"
        env["PUBLIC_KEY"] = pubkey
    pod = call(
        "POST",
        "/pods",
        data=json.dumps(
            {
                # The RunPod account is SHARED with teammates: the name says whose pod it is and
                # what it serves, and the prefix is what marks it as this tool's to tear down.
                "name": name,
                "imageName": spec.image,
                "gpuTypeIds": [spec.gpu],
                "gpuCount": spec.count,
                "containerDiskInGb": spec.disk_gb,
                "volumeInGb": 0,
                "ports": list(ports),
                "cloudType": spec.cloud,
                "dockerStartCmd": ["bash", "-lc", start_script],
                "env": env,
                # Both constraints are omitted entirely when unset: an empty allowedCudaVersions
                # or countryCodes is not "no preference" to the scheduler, and a CUDA pin that
                # only the vLLM image needs would leave a training pod unschedulable.
                **({"allowedCudaVersions": cuda} if (cuda := [
                    c.strip() for c in str(spec.cuda).split(",") if c.strip()]) else {}),
                **({"countryCodes": codes} if (codes := [
                    c.strip().upper() for c in str(spec.countries).split(",") if c.strip()]) else {}),
            }
        ),
    )
    return str(pod.get("id") or pod.get("podId", ""))


def bootstrap_script(
    base: str,
    mods: list[tuple[str, str]],
    *,
    hf_token: str | None,
    max_len: int,
    lora_rank: int,
    max_num_seqs: int,
    mode: str,
    reasoning_parser: str | None,
    tool_call_parser: str | None,
) -> str:
    """Pod startup script: install vLLM, pull base + each adapter, pin the mode, serve.

    Two rules this script encodes, both learned the hard way (scratch/serve_adapter_runpod.py,
    2026-08-16..19): credentials never pass through `set -x` (/workspace is served over a
    PUBLIC proxy on :8080, so anything xtrace echoes into boot.log is world-readable), and
    every command is one line — a backslash continuation inside a Python f-string is eaten
    by Python, not passed to bash, which once truncated the serve command into a foreground
    hang that billed for hours. `validate_bootstrap` checks the result before a pod exists.

    Args:
        base: HF id of the base model.
        mods: (served_name, adapter_repo) pairs; each becomes a LoRA module.
        hf_token: Exported on the pod for gated/private pulls; None = credential-free.
        mode: 'think' | 'nothink' | '' — pinned into the served template when set.
        reasoning_parser: vLLM `--reasoning-parser` (think-mode only, see plan_serving).
        tool_call_parser: vLLM `--tool-call-parser` (agentic harnesses only).
    """
    downloads = "\n".join(
        f"$VENV/bin/hf download {repo} --local-dir /workspace/adapter{i}"
        for i, (_, repo) in enumerate(mods)
    )
    parser_flags = ""
    if reasoning_parser:
        parser_flags += f" --reasoning-parser {reasoning_parser}"
    if tool_call_parser:
        parser_flags += (
            f" --enable-auto-tool-choice --tool-call-parser {tool_call_parser}"
        )
    # Pin thinking mode into the SERVED template. The prefix comes from vllm.pin_prefix -- the
    # SAME text the local path prepends -- so the two cannot drift; only the place it is applied
    # differs (here the template is read from the tokenizer at boot, on the pod).
    # Qwen3.6's stock template does NOT enable thinking by default, so a
    # client that cannot pass chat_template_kwargs (the ODCV harness cannot) gets no
    # `<think>` prefill; the model then emits a short think-wrapped answer, never closes
    # </think>, and the reasoning parser discards the lot -- 36 tokens, empty content, and a
    # transcript that looks like the model said nothing. Verified on a pod 2026-08-16:
    # base answered fine while the adapter returned empty until the flag was pinned.
    pin_block, template_flag = "", ""
    if mode:
        prefix = pin_prefix(mode)          # canonical; repr() below handles the escaping
        pin_block = (
            "$VENV/bin/python - <<'PYEOF'\n"
            "import pathlib\n"
            "from transformers import AutoTokenizer\n"
            f"tok = AutoTokenizer.from_pretrained('{base}', trust_remote_code=True)\n"
            "t = tok.chat_template\n"
            "assert t, 'no chat_template on the tokenizer'\n"
            f"pinned = {prefix!r} + t\n"
            "pathlib.Path('/workspace/chat_template.jinja').write_text(pinned)\n"
            "print('PINNED_TEMPLATE_CHARS', len(pinned))\n"
            "PYEOF"
        )
        template_flag = " --chat-template /workspace/chat_template.jinja"
    token_block = (
        f"set +x\nexport HF_TOKEN={hf_token}\nset -x"
        if hf_token
        else "echo 'no HF token: public repos only'"
    )
    lora_modules = " ".join(
        f"{served}=/workspace/adapter{i}" for i, (served, _) in enumerate(mods)
    )
    return f"""mkdir -p /workspace
exec > >(tee -a /workspace/boot.log) 2>&1
set -euxo pipefail
# SSH, so a docker host can tunnel to this endpoint instead of using the HTTPS proxy
# (docs/LOG.md 2026-08-09: the proxy times out on ODCV's long non-streaming rollouts).
# Overriding dockerStartCmd REPLACES the image entrypoint that normally installs
# PUBLIC_KEY and starts sshd, so it has to be redone here or the pod has no SSH at all.
# PUBLIC_KEY is public by definition -- nothing secret is echoed into the world-readable
# :8080 boot log by these lines.
(mkdir -p ~/.ssh && [ -n "${{PUBLIC_KEY:-}}" ] && echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys) || true
chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys 2>/dev/null || true
(apt-get update -qq && apt-get install -y -qq openssh-server >/dev/null 2>&1;  mkdir -p /run/sshd && /usr/sbin/sshd -D &) || echo "sshd unavailable"
(cd /workspace && nohup python3 -m http.server 8080 </dev/null >/dev/null 2>&1 &) || true
export HF_HOME=/workspace/hf
{token_block}
export HF_HUB_ENABLE_HF_TRANSFER=1
# flashinfer's JIT needs ninja and fails with exit 127 without it; the sampler flag is the
# belt to that braces, since a JIT failure surfaces only once the first request arrives.
export VLLM_USE_FLASHINFER_SAMPLER=0
# vLLM is installed into a PYTHON 3.12 venv, not the image's python3.10. Current
# flashinfer (pulled in by vllm) annotates with `array.array[int]`, which is 3.11+ syntax
# evaluated at import time, so on 3.10 the engine dies with
# "TypeError: 'type' object is not subscriptable" before any model loads. vLLM imports
# flashinfer.comm unconditionally from its compilation pass manager, so no env var
# disables it -- the interpreter version is the fix. (Observed 2026-08-16 on
# runpod/pytorch:0.7.0-dev-cu1281-torch271-ubuntu2204.)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=/root/.local/bin:$PATH
export VENV=/workspace/vllmenv
uv venv $VENV --python 3.12
uv pip install --python $VENV/bin/python -q vllm ninja huggingface_hub hf_transfer transformers
$VENV/bin/python -c "import sys; print('venv python', sys.version)"
$VENV/bin/hf download {base} >/dev/null
{downloads}
ls -la /workspace/adapter0
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
{pin_block}
echo SERVE_STARTING
nohup $VENV/bin/python -m vllm.entrypoints.openai.api_server --model {base} --served-model-name base --enable-lora --lora-modules {lora_modules} --max-lora-rank {lora_rank} --max-model-len {max_len} --max-num-seqs {max_num_seqs}{parser_flags}{template_flag} --host 0.0.0.0 --port 8000 </dev/null > /workspace/vllm.log 2>&1 &
for i in $(seq 1 120); do curl -sf http://localhost:8000/v1/models > /workspace/models.json && {{ echo VLLM_HEALTHY after ${{i}} polls; break; }}; sleep 15; done
cat /workspace/models.json || echo "NO MODELS RESPONSE"
echo SERVE_READY
sleep infinity
"""


def validate_bootstrap(script: str) -> None:
    """Refuse to create a billing pod on a script that is not valid bash or lost a flag."""
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        r = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
    finally:
        os.unlink(path)
    assert r.returncode == 0, f"bootstrap is not valid bash:\n{r.stderr}"
    for needle in (
        "--enable-lora",
        "--max-lora-rank",
        "--max-num-seqs",
        "--host 0.0.0.0",
        "--port 8000",
        "SERVE_READY",
        "VLLM_HEALTHY",
        "sshd",
        "authorized_keys",
    ):
        assert needle in script, f"bootstrap lost {needle!r}"
    serve_lines = [ln for ln in script.splitlines() if "api_server" in ln]
    assert len(serve_lines) == 1, (
        f"expected exactly one serve command, got {len(serve_lines)}"
    )
    assert serve_lines[0].rstrip().endswith("&"), (
        "serve command must background; it would hang"
    )


def serve_vllm(
    base: str,
    mods: list[tuple[str, str]],
    *,
    mode: str,
    pod_name: str,
    hf_token: str | None = None,
    max_len: int = 16384,
    lora_rank: int = 64,
    max_num_seqs: int = 32,
    gpu: str = GPU,
    disk_gb: int = 200,
    cloud: str = "SECURE",
    cuda: str = "13.0",
    reasoning_parser: str | None = None,
    tool_call_parser: str | None = None,
    pubkey_path: str = "~/.ssh/id_ed25519.pub",
    provision: ProvisionSpec | None = None,
) -> str:
    """Rent a pod that SERVES `base` + `mods` over vLLM; returns its id.

    The vLLM half of provisioning: it decides what the pod runs (`bootstrap_script`) and
    hands the rental itself to `provision_runpod`. A caller wanting a pod for something else
    -- training, a docker harness -- calls `provision_runpod` directly with its own script.

    Serving parameters are the CALLER's to decide, from `ModelProfile` facts: this function
    applies them, it does not choose them. Bills from this moment until `terminate`.

    Args:
        cuda: Comma-separated CUDA versions the host driver must satisfy. vLLM pulls torch
            built for CUDA 13, which needs driver >= 580; an older host dies at `_cuda_init`
            with "NVIDIA driver too old". Constrain scheduling to CUDA-13 hosts.
        max_num_seqs: Qwen3.6's hybrid Mamba arch allocates one cache block per decode
            sequence and REFUSES TO START above the block count (vLLM's default dies with
            "exceeds available Mamba cache blocks"). 32 is the verified value in
            src/model_profile.py; a boot constraint, not a throughput preference.
        lora_rank: Must be >= every adapter's r; vLLM's default of 16 rejects r=64.
        pubkey_path: Injected as PUBLIC_KEY for SSH when the file exists; skipped otherwise.
    """
    script = bootstrap_script(
        base,
        mods,
        hf_token=hf_token,
        max_len=max_len,
        lora_rank=lora_rank,
        max_num_seqs=max_num_seqs,
        mode=mode,
        reasoning_parser=reasoning_parser,
        tool_call_parser=tool_call_parser,
    )
    validate_bootstrap(script)
    return provision_runpod(
        provision or ProvisionSpec(gpu=gpu, disk_gb=disk_gb, cloud=cloud, cuda=cuda,
                                   pubkey_path=pubkey_path),
        name=pod_name,
        start_script=script,
    )


# --- observing pods -------------------------------------------------------------------------


def active_pods() -> list[dict]:
    pods = call("GET", "/pods")
    return pods if isinstance(pods, list) else pods.get("data", [])


def gpu_price(gpu: str = GPU, cloud: str = "SECURE") -> float | None:
    """$/hour for a GPU type, or None if the catalogue does not list it."""
    rows = call("GET", "/gputypes")
    rows = rows if isinstance(rows, list) else rows.get("data", [])
    field = "securePrice" if cloud.upper() == "SECURE" else "communityPrice"
    for g in rows:
        if g.get("id") == gpu:
            price = g.get(field)
            return float(price) if price else None
    return None


def boot_phase(pod_id: str) -> str:
    """The furthest marker the boot log has reached: booting → SERVE_STARTING → VLLM_HEALTHY → SERVE_READY."""
    try:
        boot = requests.get(boot_log_url(pod_id), timeout=30).text
    except requests.RequestException:
        return "booting"
    phase = "booting"
    for marker in ("SERVE_STARTING", "VLLM_HEALTHY", "SERVE_READY", "VLLM EXITED"):
        if marker in boot:
            phase = marker
    return phase


def served_models(endpoint: str, timeout: int = 30) -> list[str] | None:
    """Model ids an OpenAI-compatible endpoint lists, or None when it does not answer."""
    try:
        r = requests.get(f"{endpoint}/models", timeout=timeout)
        if not r.ok:
            return None
        return [m["id"] for m in r.json().get("data", [])]
    except (requests.RequestException, ValueError, KeyError):
        return None


def wait_serving(
    pod_id: str,
    *,
    timeout_s: int = 2700,
    interval_s: int = 30,
    on_phase: Callable[[str], None] = lambda _: None,
) -> list[str]:
    """Block until the pod's endpoint lists models; report each boot-phase change.

    Raises:
        RuntimeError: vLLM exited on the pod (the boot log says so) — no point waiting.
        TimeoutError: nothing served within `timeout_s`. The pod is still RUNNING and
            billing; the caller decides whether to tear it down.
    """
    deadline = time.time() + timeout_s
    last_phase = ""
    while time.time() < deadline:
        phase = boot_phase(pod_id)
        if phase != last_phase:
            on_phase(phase)
            last_phase = phase
        if phase == "VLLM EXITED":
            raise RuntimeError(
                f"vLLM exited on pod {pod_id}; read {boot_log_url(pod_id)}"
            )
        models = served_models(endpoint_url(pod_id), timeout=15)
        if models:
            return models
        time.sleep(interval_s)
    raise TimeoutError(
        f"pod {pod_id} served nothing within {timeout_s // 60} min; "
        f"boot log: {boot_log_url(pod_id)}"
    )


def warm_proxy(endpoint: str, model: str, n: int = 4, timeout_s: int = 300) -> None:
    """Fire tiny requests until `n` in a row return 200.

    docs/GOTCHAS.md 2026-08-19: `/v1/models` answering does NOT mean the proxy is routing;
    it can 404 for a minute or two after vLLM is healthy, and a single probe sails through
    while the first real burst does not.
    """
    deadline = time.time() + timeout_s
    ok = 0
    while ok < n and time.time() < deadline:
        try:
            r = requests.post(
                f"{endpoint}/chat/completions",
                timeout=60,
                json={
                    "model": model,
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            ok = ok + 1 if r.status_code == 200 else 0
        except requests.RequestException:
            ok = 0
        if ok < n:
            time.sleep(5)


# --- tearing down ---------------------------------------------------------------------------


def terminate(pod_id: str, attempts: int = 5) -> bool:
    """Delete a pod and VERIFY it is gone from the account; True when confirmed.

    A DELETE that returns 200 has been observed to leave a pod listed for a while; billing
    stops only when it is actually gone, so this re-lists and retries rather than trusting
    the first response.
    """
    for _ in range(attempts):
        try:
            call("DELETE", f"/pods/{pod_id}")
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return True
        time.sleep(3)
        if pod_id not in {p.get("id") for p in active_pods()}:
            return True
    return False


def orphans(pods: list[dict]) -> list[dict]:
    """The pods this tool provisioned (by name prefix) — the only ones it may sweep."""
    return [p for p in pods if str(p.get("name", "")).startswith(CHAT_POD_PREFIX)]


def start_watchdog(
    pod_id: str, max_lifetime_s: int, log_path: Path
) -> subprocess.Popen:
    """Spawn the detached watchdog for `pod_id`, bound to THIS process's lifetime.

    Own session (`start_new_session`), so a Ctrl-C, a closed terminal or a kill -9 of the
    chat process does not take the watchdog with it: it notices the parent is gone and
    terminates the pod. It reads the RunPod key from .env like everything else.
    """
    log = open(log_path, "a")
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "src.infra.runpod",
            "watchdog",
            pod_id,
            str(os.getpid()),
            str(max_lifetime_s),
            str(log_path),
        ],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd=str(Path.cwd()),
    )


def _parent_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def watchdog(
    pod_id: str, parent_pid: int, max_lifetime_s: int, interval_s: int = 30
) -> None:
    """Terminate `pod_id` when the parent process is gone, or when the lifetime cap passes.

    Idles otherwise. Exits once the pod is no longer listed (whoever removed it).
    """
    started = time.time()
    print(
        f"watchdog: pod {pod_id} parent {parent_pid} cap {max_lifetime_s}s", flush=True
    )
    while True:
        time.sleep(interval_s)
        try:
            listed = pod_id in {p.get("id") for p in active_pods()}
        except Exception as e:  # noqa: BLE001 - transient API/network trouble: keep watching
            print(f"watchdog: list failed ({type(e).__name__}); retrying", flush=True)
            continue
        if not listed:
            print("watchdog: pod gone; exiting", flush=True)
            return
        reason = None
        if not _parent_alive(parent_pid):
            reason = f"parent {parent_pid} is gone"
        elif time.time() - started > max_lifetime_s:
            reason = f"lifetime cap {max_lifetime_s}s reached"
        if reason:
            print(f"watchdog: {reason} -> terminating {pod_id}", flush=True)
            done = terminate(pod_id)
            print(f"watchdog: terminated={done}", flush=True)
            if done:
                return


# ======================================================================================
# `uv run runpod` — the CLI half: rent a box that holds this repo at this commit.
# ======================================================================================
#
# Everything above is the mechanism (rent, serve, tear down). What follows is the one
# command a person types to get a GPU: `up` provisions through `provision_runpod` like
# every other launcher and hands it a start script that clones this repository at the
# exact commit you are on, so `ssh <name> uv run ...` runs code that is already on origin.
# `scripts/infra/runpod.py` is the thin mirror of this, per the CLAUDE.md naming rule.

WORKDIR = "/root/work"
# OUR ssh config, in the repo and gitignored — never ~/.ssh/config. A tool that edits a
# person's ssh config is editing a file they own, that predates it and outlives it, and
# whose other entries it cannot reason about. This one holds nothing but pods this repo
# rented, is safe to delete, and `SshExec` passes it to ssh with -F for exactly the hosts
# it defines (src/infra/endpoints/vllm.py), so nothing depends on the reader having wired
# it into their own config. To get plain `ssh <pod>` in a terminal, add one line to
# ~/.ssh/config yourself — `Include <repo>/.pods/ssh_config` — which is a change you make,
# review and can undo.
SSH_CONFIG = POD_SSH_CONFIG


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
        f"BILLING.\n  uv run runpod down --pod {pod_id}")


def _write_ssh_alias(name: str, ip: str, port: int, pod_id: str) -> None:
    """Add (or refresh) this pod's entry in the repo's own ssh config (see SSH_CONFIG).

    An entry rather than a printed `ssh -p ...` line, because everything downstream takes
    a HOST: `SshExec` runs `ssh <host> <cmd>`, and `--server` on evals and chat is that
    same string. Rewritten in place under a marked block, since RunPod hands out a new
    ip/port for every pod and a stale entry of the same name would silently send the next
    run to a machine that no longer exists.
    """
    start, end = f"# >>> lasr pod {name} >>>", f"# <<< lasr pod {name} <<<"
    block = "\n".join([
        start,
        f"# pod {pod_id}, written by `uv run runpod up`",
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
    header = ("# Pods this repo rented (src/infra/runpod.py). Safe to delete; `uv run "
              "runpod up` rewrites it.\n# For plain `ssh <pod>`, add to ~/.ssh/config "
              f"yourself:  Include {SSH_CONFIG}\n\n")
    text = SSH_CONFIG.read_text() if SSH_CONFIG.exists() else header
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
       image: str = IMAGE, countries: str = "", push_env: bool = False) -> str:
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
        from omegaconf import OmegaConf

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
        f"host:      {name}  ({ip}:{port}, in {SSH_CONFIG})",
        f"boot log:  https://{pod_id}-8080.proxy.runpod.net/boot.log",
        "ssh:       " + ("ready" if reachable else "not answering yet — watch the boot log"),
        "",
        "The boot log says READY when the clone and `uv sync` have finished. Then:"
        if clone else "Nothing is checked out on it; the boot log says READY when uv is in.",
        (f"  ssh -F {SSH_CONFIG} {name} 'cd {WORKDIR} && uv run torchrun "
         f"--nproc_per_node={count} scripts/train/train_lora.py "
         "--config configs/train/<arm>.yaml'") if clone
        else f"  ssh -F {SSH_CONFIG} {name}",
        f"(`uv run evals --server {name}` needs no -F: it reads that file itself.",
        f" For a bare `ssh {name}`, add `Include {SSH_CONFIG}` to ~/.ssh/config.)",
        "",
        "IT BILLS UNTIL YOU RUN THIS:",
        f"  uv run runpod down --pod {pod_id}",
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


def cli() -> None:
    """Console entry (`uv run runpod up|status|pods|down`, [project.scripts])."""
    import fire

    fire.Fire({"up": up, "status": status, "pods": pods, "down": down})


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "watchdog":
        watchdog(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        raise SystemExit(
            "usage: python -m src.infra.runpod watchdog <pod_id> <parent_pid> "
            "<max_lifetime_s> <log_path>"
        )
