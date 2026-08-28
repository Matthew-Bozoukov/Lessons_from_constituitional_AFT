# ABOUTME: Serve HF LoRA adapters on a RunPod GPU via vLLM over RunPod's HTTPS proxy, so evals
# ABOUTME: can be driven locally with no SSH tunnel. Thin CLI over src.endpoints.runpod: up / status / down.

"""Stand up a vLLM endpoint for one or more LoRA adapters on Qwen3.6-27B.

Ports 8000 (OpenAI API) and 8080 (boot log) are published as RunPod `/http` ports, so the
endpoint is reachable at `https://<pod>-8000.proxy.runpod.net/v1` from anywhere — the eval
loop, the judge calls and the artifacts all stay on the local machine.

The bootstrap and its hard-won rules (no credentials through `set -x`, one-line commands,
validated before a pod exists, python-3.12 venv for vLLM) moved to
src/endpoints/runpod.py on 2026-08-27 so `uv run chat` can share them; this file keeps the
CLI that ODCV runs were launched with.

    uv run python scratch/serve_adapter_runpod.py up --adapter LASR-Callum/... --name memself --mode think
    uv run python scratch/serve_adapter_runpod.py status --pod <id>
    uv run python scratch/serve_adapter_runpod.py down --pod <id>
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.endpoints import runpod  # noqa: E402

BASE = "Qwen/Qwen3.6-27B"


def up(
    adapter: str,
    name: str,
    max_len: int = 16384,
    lora_rank: int = 64,
    max_num_seqs: int = 32,
    gpu: str = runpod.GPU,
    disk_gb: int = 200,
    cloud: str = "SECURE",
    cuda: str = "13.0",
    agentic: bool = False,
    mode: str = "",
    pubkey_path: str = "~/.ssh/id_ed25519.pub",
    pod_name: str = "",
) -> None:
    """Create the serving pod.

    Args:
        adapter: HF repo of the LoRA adapter to serve. Comma-separate to serve several on one
            pod (e.g. "repoA,repoB"), paired positionally with `name`.
        name: Served model name(s) the eval will target (e.g. "memself"), comma-separated to
            match multiple adapters.
        max_len: vLLM max_model_len.
        lora_rank: Must be >= the adapter's r; vLLM's default of 16 rejects r=64.
        max_num_seqs: Qwen3.6's Mamba cache-block boot constraint; 32 is the verified value.
        gpu: RunPod GPU type id.
        disk_gb: Container disk (base model ~55GB + cache).
        cloud: SECURE or COMMUNITY.
        cuda: Comma-separated CUDA versions the host driver must satisfy (vLLM needs 13).
        agentic: Add `--reasoning-parser qwen3 --enable-auto-tool-choice
            --tool-call-parser qwen3_xml`, the verified values from src/model_profile.py.
            REQUIRED for ODCV: without them the agent cannot emit tool calls, so it never
            acts, yet the harness still reports every scenario `ok` while writing NO
            transcript. Off by default so non-agentic evals keep the behaviour they were
            verified against.
        mode: 'think'|'nothink' to pin thinking mode into the served template.
        pubkey_path: SSH public key injected into the pod (skipped when the file is absent).
        pod_name: Prefix with an owner; the RunPod account is shared with teammates.
    """
    load_dotenv(override=True)

    # fire evaluates a bare `a,b` (quoted or not -- the shell strips the quotes) as a tuple
    # when every element is a valid literal, so `--name x,y` arrives as a tuple while
    # `--adapter org/a,org/b` (slashes, dots) stays a string. str() of the tuple used to
    # leak "('x'" into the vLLM command line and fail bash validation.
    def _list(v) -> list[str]:
        parts = v if isinstance(v, (list, tuple)) else str(v).split(",")
        return [str(p).strip() for p in parts if str(p).strip()]

    adapters = _list(adapter)
    names = _list(name)
    assert len(adapters) == len(names), (
        f"got {len(adapters)} adapters but {len(names)} names"
    )
    pid = runpod.launch_pod(
        BASE,
        list(zip(names, adapters)),
        mode=mode,
        pod_name=pod_name or f"serve-{names[0]}",
        hf_token=os.environ.get("HF_TOKEN") or None,
        max_len=max_len,
        lora_rank=lora_rank,
        max_num_seqs=max_num_seqs,
        gpu=gpu,
        disk_gb=disk_gb,
        cloud=cloud,
        cuda=cuda,
        reasoning_parser="qwen3" if agentic else None,
        tool_call_parser="qwen3_xml" if agentic else None,
        pubkey_path=pubkey_path,
    )
    print(f">>> pod {pid}  serving {adapters} as {names}")
    print(f"    endpoint: {runpod.endpoint_url(pid)}")
    print(f"    boot log: {runpod.boot_log_url(pid)}")
    print(
        f"    DOWN    : uv run python scratch/serve_adapter_runpod.py down --pod {pid}"
    )


def status(pod: str) -> None:
    """Report boot phase and whether the endpoint answers.

    Args:
        pod: Pod id.
    """
    load_dotenv(override=True)
    print(f">>> pod {pod}  phase={runpod.boot_phase(pod)}")
    models = runpod.served_models(runpod.endpoint_url(pod), timeout=40)
    print("    served:", models if models is not None else "endpoint not answering yet")


def down(pod: str) -> None:
    """Destroy the pod and print remaining active pods as verification.

    Args:
        pod: Pod id.
    """
    load_dotenv(override=True)
    gone = runpod.terminate(pod)
    pods = runpod.active_pods()
    print(
        f">>> {'destroyed' if gone else 'COULD NOT CONFIRM destruction of'} {pod}; "
        f"active pods now: {len(pods)}"
    )
    for p in pods:
        print("   ", p["id"], p.get("desiredStatus"), p.get("name"))


if __name__ == "__main__":
    fire.Fire()
