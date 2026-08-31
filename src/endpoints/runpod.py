# ABOUTME: RunPod REST client + the vLLM serving pod (one base, several LoRA modules, reached
# ABOUTME: over RunPod's HTTPS proxy) that `uv run chat` and the evals stand up, poll and tear down.

"""RunPod pods that serve model organisms.

A serving pod boots credential-light (an HF token only when one is given, never the RunPod
key), installs vLLM into a Python 3.12 venv, pulls the base model and every adapter, pins
the thinking mode into the chat template exactly as src/endpoints/vllm_server.pin_template
does, and serves `base` + one LoRA module per adapter on :8000, published through the
proxy at https://<pod>-8000.proxy.runpod.net/v1. :8080 serves the boot log so "still
downloading" can be told from "dead" from a browser.

Everything a pod costs is time on a billing meter, so this module also owns the three
ways one is torn down: `terminate` (verified against the API, not fire-and-forget), the
detached `watchdog` process that terminates a pod when the process that launched it is
gone or a lifetime cap passes (CLAUDE.md "Paid infrastructure": never rely on the
orchestrator surviving), and `orphans` for the startup sweep.

    python -m src.endpoints.runpod watchdog <pod_id> <parent_pid> <max_lifetime_s> <log>
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import requests

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
    # Pin thinking mode into the SERVED template, exactly as src/endpoints/vllm_server.py
    # pin_template does. Qwen3.6's stock template does NOT enable thinking by default, so a
    # client that cannot pass chat_template_kwargs (the ODCV harness cannot) gets no
    # `<think>` prefill; the model then emits a short think-wrapped answer, never closes
    # </think>, and the reasoning parser discards the lot -- 36 tokens, empty content, and a
    # transcript that looks like the model said nothing. Verified on a pod 2026-08-16:
    # base answered fine while the adapter returned empty until the flag was pinned.
    pin_block, template_flag = "", ""
    if mode:
        assert mode in ("think", "nothink"), mode
        flag = "true" if mode == "think" else "false"
        pin_block = (
            "$VENV/bin/python - <<'PYEOF'\n"
            "import pathlib\n"
            "from transformers import AutoTokenizer\n"
            f"tok = AutoTokenizer.from_pretrained('{base}', trust_remote_code=True)\n"
            "t = tok.chat_template\n"
            "assert t, 'no chat_template on the tokenizer'\n"
            f"pinned = '{{%- set enable_thinking = {flag} -%}}\\n"
            f"{{%- set preserve_thinking = {flag} -%}}\\n' + t\n"
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


def launch_pod(
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
) -> str:
    """Create a serving pod; returns its id. Bills from this moment until `terminate`.

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
    env = {"HF_HUB_ENABLE_HF_TRANSFER": "1"}
    pubkey_file = Path(pubkey_path).expanduser()
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
                "name": pod_name,
                "imageName": IMAGE,
                "gpuTypeIds": [gpu],
                "gpuCount": 1,
                "containerDiskInGb": disk_gb,
                "volumeInGb": 0,
                "ports": ["8000/http", "8080/http", "22/tcp"],
                "cloudType": cloud,
                "allowedCudaVersions": [
                    c.strip() for c in str(cuda).split(",") if c.strip()
                ],
                "dockerStartCmd": ["bash", "-lc", script],
                "env": env,
            }
        ),
    )
    return str(pod["id"])


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
            "src.endpoints.runpod",
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


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "watchdog":
        watchdog(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        raise SystemExit(
            "usage: python -m src.endpoints.runpod watchdog <pod_id> <parent_pid> "
            "<max_lifetime_s> <log_path>"
        )
