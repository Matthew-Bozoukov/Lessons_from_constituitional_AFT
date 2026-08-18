# ABOUTME: Serve a LoRA arm on a throwaway RunPod pod, exposed over RunPod's HTTPS proxy so
# ABOUTME: ODCV's docker containers reach it directly. Run: uv run python scratch/serve_adapter_runpod.py up --target <hf>
"""Serve one adapter for the eval harnesses, over the RunPod HTTPS proxy.

Written 2026-08-18: the ODCV arm configs reference a `scratch/serve_adapter_runpod.py`
that did not land in the repo, so this reconstructs it against the same contract.

WHY THE PROXY AND NOT AN SSH TUNNEL. ODCV starts a docker network per scenario and the
agent containers must reach the model endpoint. Over a tunnel that means a 172.17.0.1
bridge hop that has to stay alive for the whole run; publishing :8000 as a RunPod /http
port lets the containers go straight out to the internet instead. The arm configs say the
same thing.

WHY THE ARGV IS COMPUTED LOCALLY. Every serving decision belongs to `plan_serving` --
context window, `max_num_seqs`, which reasoning/tool parsers, whether prefix caching is
safe -- and the thinking pin belongs to `pin_template`. Recomputing any of that by hand on
the pod would put a second decision-maker in the loop, which is exactly what the
facts/requirements split exists to prevent. So both are resolved here with the repo's own
functions and shipped to the pod as literals; the pod only needs vLLM.

THE FLAGS THAT MATTER. `--enable-auto-tool-choice --tool-call-parser qwen3_xml` come from
Qwen3.6's ModelProfile. Without them ODCV completes every scenario as `ok` and writes NO
transcript -- the agent cannot emit tool calls, so it never acts -- which looks like a
clean run in the summary and is only visible by checking messages_record.txt exists. That
failure cost one full run before it was diagnosed, so `--agentic true` is mandatory for
ODCV and off by default for plain generation.
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.endpoints.vllm_server import (  # noqa: E402
    native_context_window,
    pin_template,
    plan_serving,
    resolve_target,
)
from src.eval.misalignment.internalization.scripts.runpod import call  # noqa: E402
from src.huggingface import hf_download  # noqa: E402
from src.model_profile import serving_params  # noqa: E402

IMAGE = "runpod/pytorch:0.7.0-dev-cu1281-torch271-ubuntu2204"
DEFAULT_GPU = "NVIDIA H100 80GB HBM3"


def _plan(target: str, context_window: int, agentic: bool):
    """Resolve the target and translate plan_serving's decisions into a vLLM argv.

    Args:
        target: HF adapter or full-model path.
        context_window: What the eval requires; plan_serving clamps to the model's native.
        agentic: Whether to enable tool calling.

    Returns:
        (argv, pinned_template_text_or_None, served_lora_name).
    """
    spec = resolve_target(target)
    facts = dict(serving_params(spec.base_model),
                 native_context_window=native_context_window(spec.base_model))
    # `needs_tool_calls` is the requirement key that makes plan_serving emit
    # --enable-auto-tool-choice / --tool-call-parser. Passing only context_window silently
    # omits them, which is exactly the ODCV failure this file's docstring warns about:
    # every scenario completes `ok` and no transcript is ever written.
    reqs = {"context_window": context_window}
    if agentic:
        reqs["needs_tool_calls"] = True
    plan = plan_serving(facts, reqs, spec.base_model, spec.mode)
    for w in plan["warnings"]:
        print(f"!!! {w}")

    argv = ["python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", spec.base_model, "--served-model-name", "base",
            "--dtype", "bfloat16",
            "--max-model-len", str(plan["context_window"]),
            "--gpu-memory-utilization", "0.94",
            "--host", "0.0.0.0", "--port", "8000"]
    # Translation only below -- every `if` mirrors a decision plan_serving already made.
    if plan["max_num_seqs"]:
        argv += ["--max-num-seqs", str(plan["max_num_seqs"])]
    if plan["reasoning_parser"]:
        argv += ["--reasoning-parser", plan["reasoning_parser"]]
    if agentic and plan["tool_call_parser"]:
        argv += ["--enable-auto-tool-choice",
                 "--tool-call-parser", plan["tool_call_parser"]]
    if plan["prefix_caching"]:
        argv += ["--enable-prefix-caching"]

    template = None
    if spec.mode != "default":
        raw = json.load(open(hf_download(spec.base_model, "tokenizer_config.json")))
        template = pin_template(raw["chat_template"], spec.mode)
        argv += ["--chat-template", "/workspace/chat_template.jinja"]
    if spec.adapter:
        argv += ["--enable-lora",
                 "--max-lora-rank", str(max(spec.lora_rank or 32, 32)),
                 "--lora-modules", f"{spec.model_key}=/workspace/adapter"]
    print(f">>> mode={spec.mode}  base={spec.base_model}  served_lora={spec.model_key}")
    print(f">>> argv: {' '.join(argv)}")
    return argv, template, spec.model_key


def up(target: str, name: str = "serve-adapter", gpu: str = DEFAULT_GPU,
       context_window: int = 16384, agentic: bool = False, disk_gb: int = 200,
       cloud: str = "SECURE") -> str:
    """Provision a pod serving `target` on :8000 via the RunPod HTTPS proxy.

    Args:
        target: HF adapter or full-model path.
        name: Pod name; prefix it so it is distinguishable on the shared account.
        gpu: RunPod GPU type id.
        context_window: What the eval requires.
        agentic: True for ODCV (enables tool calling). False for plain generation.
        disk_gb: Container disk (base ~55GB + HF cache).
        cloud: SECURE or COMMUNITY.

    Returns:
        Pod id, the base_url to point a config at, and the teardown command.
    """
    argv, template, model_key = _plan(target, context_window, agentic)
    cmd = " ".join(shlex.quote(a) for a in argv)
    tpl = ""
    if template:
        tpl = ("cat > /workspace/chat_template.jinja <<'TEMPLATE_EOF'\n"
               + template + "\nTEMPLATE_EOF\n")

    boot = f"""mkdir -p /workspace
exec > >(tee -a /workspace/boot.log) 2>&1
set -euxo pipefail
(cd /workspace && nohup python3 -m http.server 8080 </dev/null >/dev/null 2>&1 &) || true
export HF_HOME=/workspace/hf
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
python3 -m pip install --no-cache-dir -q vllm huggingface_hub hf_transfer
{tpl}hf download {target} --local-dir /workspace/adapter
echo SERVER_STARTING
({cmd} 2>&1 | tee /workspace/vllm.log) || true
echo SERVER_EXITED
sleep infinity
"""
    payload = {
        "name": name, "imageName": IMAGE, "gpuTypeIds": [gpu], "gpuCount": 1,
        "containerDiskInGb": disk_gb, "volumeInGb": 0,
        "ports": ["8000/http", "8080/http", "22/tcp"], "cloudType": cloud,
        "dockerStartCmd": ["bash", "-lc", boot],
        "env": {"HF_HUB_ENABLE_HF_TRANSFER": "1"},
    }
    pod = call("POST", "/pods", data=json.dumps(payload))
    pid = pod.get("id") or pod.get("podId", "")
    return (f"pod:       {pid}\n"
            f"base_url:  https://{pid}-8000.proxy.runpod.net/v1\n"
            f"model:     {model_key}\n"
            f"vllm log:  https://{pid}-8080.proxy.runpod.net/vllm.log\n"
            f"boot log:  https://{pid}-8080.proxy.runpod.net/boot.log\n\n"
            f"TEAR DOWN: uv run python scratch/serve_adapter_runpod.py down --pod {pid}")


def down(pod: str) -> str:
    """Terminate the pod. Always run this; it bills by the second."""
    call("DELETE", f"/pods/{pod}")
    return f"terminated {pod}"


if __name__ == "__main__":
    fire.Fire({"up": up, "down": down})
