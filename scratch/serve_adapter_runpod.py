# ABOUTME: Serve a HF LoRA adapter on a RunPod GPU via vLLM, exposed over RunPod's HTTPS
# ABOUTME: proxy so evals can be driven locally with no SSH tunnel. up / status / down.

"""Stand up a vLLM endpoint for one LoRA adapter.

Ports 8000 (OpenAI API) and 8080 (boot log) are published as RunPod `/http` ports, so the
endpoint is reachable at `https://<pod>-8000.proxy.runpod.net/v1` from anywhere — the eval
loop, the judge calls and the artifacts all stay on the local machine.

Two rules this file encodes, both learned the hard way:

  - **Credentials never pass through `set -x`.** /workspace is served over a PUBLIC HTTPS
    proxy on :8080, so anything xtrace echoes into boot.log is world-readable.
  - **Every command is one line, and the script is validated before the pod is created.**
    A backslash continuation inside a Python f-string is eaten by Python, not passed to
    bash; that once truncated a serve command into a foreground hang that billed for hours.

    uv run python scratch/serve_adapter_runpod.py up --adapter LASR-Callum/... --name memself
    uv run python scratch/serve_adapter_runpod.py status --pod <id>
    uv run python scratch/serve_adapter_runpod.py down --pod <id>
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import fire
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.misalignment.internalization.scripts.runpod import call  # noqa: E402

BASE = "Qwen/Qwen3.6-27B"
IMAGE = "runpod/pytorch:0.7.0-dev-cu1281-torch271-ubuntu2204"
GPU = "NVIDIA H100 80GB HBM3"


def _bootstrap(mods: list[tuple[str, str]], hf_token: str, max_len: int,
               lora_rank: int, max_num_seqs: int, agentic: bool = False,
               mode: str = "") -> str:
    """Pod startup script: install vLLM, pull base + each adapter, serve, report readiness.

    Args:
        mods: List of (served_name, adapter_repo) pairs to serve as LoRA modules.
        agentic: Add the reasoning and tool-call parsers; see `agentic` in up().
        mode: 'think' | 'nothink' | '' — pin thinking mode into the served chat template.
    """
    downloads = "\n".join(
        f"$VENV/bin/hf download {repo} --local-dir /workspace/adapter{i}"
        for i, (_, repo) in enumerate(mods))
    # ODCV and other agentic harnesses need these; see `agentic` in up().
    agentic_flags = (" --reasoning-parser qwen3 --enable-auto-tool-choice "
                     "--tool-call-parser qwen3_xml") if agentic else ""
    # Pin thinking mode into the SERVED template, exactly as src/endpoints/vllm_server.py
    # pin_template does. Qwen3.6's stock template does NOT enable thinking by default, so a
    # client that cannot pass chat_template_kwargs (the ODCV harness cannot) gets no
    # `<think>` prefill; the model then emits a short think-wrapped answer, never closes
    # </think>, and the reasoning parser discards the lot -- 36 tokens, empty content, and a
    # transcript that looks like the model said nothing. Verified on this pod 2026-08-16:
    # base answered fine while the adapter returned empty until the flag was pinned.
    pin_block, template_flag = "", ""
    if mode:
        flag = "true" if mode == "think" else "false"
        pin_block = (
            "$VENV/bin/python - <<'PYEOF'\n"
            "import pathlib\n"
            "from transformers import AutoTokenizer\n"
            f"tok = AutoTokenizer.from_pretrained('{BASE}', trust_remote_code=True)\n"
            "t = tok.chat_template\n"
            "assert t, 'no chat_template on the tokenizer'\n"
            f"pinned = '{{%- set enable_thinking = {flag} -%}}\\n"
            f"{{%- set preserve_thinking = {flag} -%}}\\n' + t\n"
            "pathlib.Path('/workspace/chat_template.jinja').write_text(pinned)\n"
            "print('PINNED_TEMPLATE_CHARS', len(pinned))\n"
            "PYEOF"
        )
        template_flag = " --chat-template /workspace/chat_template.jinja"
    lora_modules = " ".join(
        f"{served}=/workspace/adapter{i}" for i, (served, _) in enumerate(mods))
    return f"""mkdir -p /workspace
exec > >(tee -a /workspace/boot.log) 2>&1
set -euxo pipefail
# SSH, so a vast docker host can tunnel to this endpoint instead of using the HTTPS proxy
# (which docs/LOG.md 2026-08-09 records as timing out on ODCV's long non-streaming
# rollouts). Overriding dockerStartCmd REPLACES the image entrypoint that normally installs
# PUBLIC_KEY and starts sshd, so it has to be redone here or the pod has no SSH at all.
# PUBLIC_KEY is public by definition -- nothing secret is echoed into the world-readable
# :8080 boot log by these lines.
(mkdir -p ~/.ssh && [ -n "${{PUBLIC_KEY:-}}" ] && echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys) || true
chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys 2>/dev/null || true
(apt-get update -qq && apt-get install -y -qq openssh-server >/dev/null 2>&1;  mkdir -p /run/sshd && /usr/sbin/sshd -D &) || echo "sshd unavailable"
(cd /workspace && nohup python3 -m http.server 8080 </dev/null >/dev/null 2>&1 &) || true
export HF_HOME=/workspace/hf
set +x
export HF_TOKEN={hf_token}
set -x
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
$VENV/bin/hf download {BASE} >/dev/null
{downloads}
ls -la /workspace/adapter0
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
{pin_block}
echo SERVE_STARTING
nohup $VENV/bin/python -m vllm.entrypoints.openai.api_server --model {BASE} --served-model-name base --enable-lora --lora-modules {lora_modules} --max-lora-rank {lora_rank} --max-model-len {max_len} --max-num-seqs {max_num_seqs}{agentic_flags}{template_flag} --host 0.0.0.0 --port 8000 </dev/null > /workspace/vllm.log 2>&1 &
for i in $(seq 1 120); do curl -sf http://localhost:8000/v1/models > /workspace/models.json && {{ echo VLLM_HEALTHY after ${{i}} polls; break; }}; sleep 15; done
cat /workspace/models.json || echo "NO MODELS RESPONSE"
echo SERVE_READY
sleep infinity
"""


def _validate(script: str) -> None:
    """Refuse to create a billing pod on a script that is not valid bash or lost a flag."""
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(script)
        path = f.name
    r = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
    assert r.returncode == 0, f"bootstrap is not valid bash:\n{r.stderr}"
    for needle in ("--enable-lora", "--max-lora-rank", "--max-num-seqs", "--host 0.0.0.0",
                   "--port 8000", "SERVE_READY", "VLLM_HEALTHY", "sshd", "authorized_keys"):
        assert needle in script, f"bootstrap lost {needle!r}"
    serve_lines = [ln for ln in script.splitlines() if "api_server" in ln]
    assert len(serve_lines) == 1, f"expected exactly one serve command, got {len(serve_lines)}"
    assert serve_lines[0].rstrip().endswith("&"), "serve command must background; it would hang"
    print(f">>> bootstrap validated ({len(script.splitlines())} lines, 1 serve command)")


def up(adapter: str, name: str, max_len: int = 16384, lora_rank: int = 64,
       max_num_seqs: int = 32, gpu: str = GPU, disk_gb: int = 200,
       cloud: str = "SECURE", cuda: str = "13.0", agentic: bool = False, mode: str = "",
       pubkey_path: str = "~/.ssh/id_ed25519.pub", pod_name: str = "") -> None:
    """Create the serving pod.

    Args:
        adapter: HF repo of the LoRA adapter to serve. Comma-separate to serve several on one
            pod (e.g. "repoA,repoB"), paired positionally with `name`.
        name: Served model name(s) the eval will target (e.g. "memself"), comma-separated to
            match multiple adapters.
        max_len: vLLM max_model_len.
        lora_rank: Must be >= the adapter's r; vLLM's default of 16 rejects r=64.
        max_num_seqs: Qwen3.6's hybrid Mamba arch allocates one cache block per decode
            sequence and REFUSES TO START above the block count (vLLM's default 1024 dies
            with "exceeds available Mamba cache blocks (313)"). 32 is the value verified in
            src/model_profile.py; it is a boot constraint, not a throughput preference.
        gpu: RunPod GPU type id.
        disk_gb: Container disk (base model ~55GB + cache).
        cloud: SECURE or COMMUNITY.
        agentic: Add `--reasoning-parser qwen3 --enable-auto-tool-choice
            --tool-call-parser qwen3_xml`, the verified values from src/model_profile.py.
            REQUIRED for ODCV: without them the agent cannot emit tool calls, so it never
            acts, yet the harness still reports every scenario `ok` while writing NO
            transcript. That silent failure cost a full run. Off by default so
            non-agentic evals keep the behaviour they were verified against.
        mode: 'think'|'nothink' to pin thinking mode into the served template.
        cuda: Comma-separated CUDA versions the host driver must satisfy. vLLM 0.26.0 pulls
            torch built for CUDA 13, which needs driver >= 580; an older host (e.g. 12.8) dies
            at `_cuda_init` with "NVIDIA driver too old". Constrain scheduling to CUDA-13 hosts.
    """
    load_dotenv(override=True)
    adapters = [a.strip() for a in str(adapter).split(",") if a.strip()]
    names = [n.strip() for n in str(name).split(",") if n.strip()]
    assert len(adapters) == len(names), f"got {len(adapters)} adapters but {len(names)} names"
    mods = list(zip(names, adapters))
    script = _bootstrap(mods, os.environ["HF_TOKEN"], max_len, lora_rank, max_num_seqs,
                        agentic, mode)
    _validate(script)
    pubkey = Path(pubkey_path).expanduser().read_text().strip()
    assert pubkey.startswith("ssh-"), f"not an ssh public key: {pubkey_path}"
    pod = call("POST", "/pods", data=json.dumps({
        # The RunPod account is SHARED with teammates, so a pod must be identifiable as
        # whose it is at a glance; pass pod_name to prefix it with an owner.
        "name": pod_name or f"serve-{names[0]}",
        "imageName": IMAGE,
        "gpuTypeIds": [gpu],
        "gpuCount": 1,
        "containerDiskInGb": disk_gb,
        "volumeInGb": 0,
        "ports": ["8000/http", "8080/http", "22/tcp"],
        "cloudType": cloud,
        "allowedCudaVersions": [c.strip() for c in str(cuda).split(",") if c.strip()],
        "dockerStartCmd": ["bash", "-lc", script],
        "env": {"HF_HUB_ENABLE_HF_TRANSFER": "1", "PUBLIC_KEY": pubkey},
    }))
    pid = pod["id"]
    print(f">>> pod {pid}  serving {adapters} as {names}")
    print(f"    endpoint: https://{pid}-8000.proxy.runpod.net/v1")
    print(f"    boot log: https://{pid}-8080.proxy.runpod.net/boot.log")
    print(f"    DOWN    : uv run python scratch/serve_adapter_runpod.py down --pod {pid}")


def status(pod: str) -> None:
    """Report boot phase and whether the endpoint answers.

    Args:
        pod: Pod id.
    """
    load_dotenv(override=True)
    try:
        boot = requests.get(f"https://{pod}-8080.proxy.runpod.net/boot.log", timeout=30).text
    except requests.RequestException:
        boot = ""
    phase = "booting"
    for m in ("SERVE_STARTING", "VLLM_HEALTHY", "SERVE_READY"):
        if m in boot:
            phase = m
    print(f">>> pod {pod}  phase={phase}")
    try:
        r = requests.get(f"https://{pod}-8000.proxy.runpod.net/v1/models", timeout=40)
        print("    served:", [m["id"] for m in r.json()["data"]])
    except Exception:
        print("    endpoint not answering yet")


def down(pod: str) -> None:
    """Destroy the pod and print remaining active pods as verification.

    Args:
        pod: Pod id.
    """
    load_dotenv(override=True)
    call("DELETE", f"/pods/{pod}")
    pods = call("GET", "/pods")
    pods = pods if isinstance(pods, list) else pods.get("data", [])
    print(f">>> destroyed {pod}; active pods now: {len(pods)}")
    for p in pods:
        print("   ", p["id"], p.get("desiredStatus"), p.get("name"))


if __name__ == "__main__":
    fire.Fire()
