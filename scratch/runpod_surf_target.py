# ABOUTME: Launch/monitor/destroy a RunPod H100 serving Qwen3.6-27B + one LoRA arm as the
# ABOUTME: SURF sweep target. Run: uv run python scratch/runpod_surf_target.py up

"""Serve one LoRA arm on a throwaway RunPod H100 for a SURF red-teaming sweep.

SURF drives the target over the OpenAI-compatible endpoint, so the pod only has to serve
it — the EM loop, the query model and the judge all run on the local machine, where the
OpenRouter key lives. The base model and the adapter are both public, so the pod holds no
credential and cannot leak one.

The bootstrap is lifted from scripts/gpu/runpod_arena_hard.py, whose comments record why
each line is the way it is (log server before anything slow, vLLM in the foreground of
PID 1, `|| true` so a crash leaves a readable log instead of a restart loop). The
differences from that script are the ones this run needs: a single adapter at rank 64,
and a download filter so the adapter repo's `last-checkpoint/` optimizer state is not
pulled across the wire for nothing.

    uv run python scratch/runpod_surf_target.py up
    uv run python scratch/runpod_surf_target.py status --pod <id>
    uv run python scratch/runpod_surf_target.py down --pod <id>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.misalignment.internalization.scripts.runpod import call  # noqa: E402

DEFAULT_IMAGE = "runpod/pytorch:0.7.0-dev-cu1281-torch271-ubuntu2204"
DEFAULT_GPU = "NVIDIA H100 80GB HBM3"
BASE = "Qwen/Qwen3.6-27B"
# Every arm on ONE server: same build, same flags, same GPU, so a difference between arms
# is the adapter rather than the serving. vLLM answers for `base` too, which makes the
# untrained control free.
MODULES = {
    "t2synth": "LASR-Callum/qwen3.6-27b-lora-table2-synthdoc-r64",
    "t2only": "LASR-Callum/qwen3.6-27b-lora-table2-only-9284-r64",
    # 9,284 table2 rows + 716 synth (~7%), so the three arms form a synth-fraction ladder.
    "t2synth716": "LASR-Callum/qwen3.6-27b-lora-t2-9284-synthdoc-716-r64",
    # 80/20 mixes swapping the synth 20% for other data types, to test whether the
    # fabrication reduction is specific to synth or comes from any 20% admixture.
    "memself": "LASR-Callum/qwen3.6-27b-lora-table2-80-memself-20-r64",
    "selfreflect": "LASR-Callum/qwen3.6-27b-lora-table2-80-selfreflect-20-r64",
    # Same 9,284+716 mixture as t2synth716, but its 716 difficult-advice rows are the
    # highest-LESS-influence rows of the top three traits rather than a random draw. Its
    # protocol-matched control is untrained, so it pairs with t2synth716 only across a
    # training-protocol difference (legacy batch-1 vs dynamic batching).
    "lessswap716": "LASR-Callum/qwen3.6-27b-lora-t2-9284-synthdoc716-lessswap-r64",
}
LORA_RANK = 64


def _bootstrap(base: str, modules: dict[str, str], max_rank: int,
               reasoning_parser: bool = True, tool_calls: bool = False) -> str:
    """Return the pod startup script serving `base` with each of `modules` as a LoRA.

    Args:
        base: HF id of the base model.
        modules: served-name -> adapter HF id. All are attached to one server.
            Pass {} to serve the base with `--enable-lora` but no adapters attached: a
            PRIVATE adapter can then be scp'd up and loaded at runtime through
            /v1/load_lora_adapter, which keeps the HF token off the rented box.
        max_rank: `--max-lora-rank`; must be >= the largest adapter's r.
        tool_calls: attach `--enable-auto-tool-choice --tool-call-parser qwen3_xml`.
            REQUIRED by ODCV-Bench: without them the agent cannot emit tool calls, so
            every scenario completes as `ok` having taken no action and written no
            transcript — a clean-looking summary hiding a run that did nothing.
        reasoning_parser: attach `--reasoning-parser qwen3`. Set False to get the RAW
            completion in `content`, thinking trace included — see the flag's comment at
            the serve line for why that is sometimes the only correct choice.
    """
    # Root-level adapter files only. These repos also carry adapter/ and last-checkpoint/
    # copies, and last-checkpoint/optimizer.pt is training state vLLM has no use for.
    # POSITIONAL filenames, not `--include` patterns: `hf download` treats trailing args as
    # explicit filenames and then warns it is ignoring `--include`, so the `--include a b`
    # form silently fetches only `b` and vLLM dies ~10 min later on the missing
    # adapter_config.json. That cost one pod boot. The `test -f` lines fail immediately
    # instead of after the ~55GB base download.
    downloads = "\n".join(
        f"mkdir -p /workspace/adapters/{name}\n"
        f"hf download {repo} --local-dir /workspace/adapters/{name}"
        f" adapter_config.json adapter_model.safetensors\n"
        f"test -f /workspace/adapters/{name}/adapter_config.json\n"
        f"test -f /workspace/adapters/{name}/adapter_model.safetensors"
        for name, repo in modules.items())
    lora_args = " ".join(f"{name}=/workspace/adapters/{name}" for name in modules)
    # `--lora-modules` is omitted entirely when there are none; vLLM still accepts runtime
    # loads because --enable-lora is set. --max-loras must be >=1 regardless.
    lora_args = f"--lora-modules {lora_args}" if modules else ""
    n_loras = max(len(modules), 1)
    reasoning_arg = "--reasoning-parser qwen3 " if reasoning_parser else ""
    tool_arg = ("--enable-auto-tool-choice --tool-call-parser qwen3_xml "
                if tool_calls else "")
    return f"""mkdir -p /workspace
exec > >(tee -a /workspace/boot.log) 2>&1
set -euxo pipefail
mkdir -p ~/.ssh && [ -n "${{PUBLIC_KEY:-}}" ] && echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys 2>/dev/null || true
(apt-get update -qq && apt-get install -y -qq openssh-server >/dev/null 2>&1; \
 mkdir -p /run/sshd && /usr/sbin/sshd -D &) || echo "sshd unavailable"
(cd /workspace && nohup python3 -m http.server 8080 </dev/null >/dev/null 2>&1 &) || true
export HF_HOME=/workspace/hf
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# vllm PINNED, not ">=0.8.5". The image ships CUDA 12.8.1 (driver 12080); an unpinned
# install resolved to 0.26.0 on 2026-08-09 morning and to a CUDA-13 build the same evening,
# which dies with "The NVIDIA driver on your system is too old (found version 12080)" after
# the 55GB download. 0.26.0 is the version this sweep actually ran on. Two pods died to this.
pip install --no-cache-dir -q "vllm==0.26.0" "transformers>=4.51.3" peft huggingface_hub hf_transfer ninja
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
nvidia-smi --query-gpu=driver_version --format=csv,noheader
mkdir -p /workspace/adapters
{downloads}
# `|| true`: with no modules the directory is empty, and a failing ls under `set -e` kills
# PID 1 and puts the pod in a restart loop that shows up only as a boot log that keeps
# resetting to 0 bytes. Two pods were lost to exactly that before the pattern was spotted.
ls -lR /workspace/adapters || true
echo ADAPTERS_DOWNLOADED
# No --chat-template: the adapter was trained with thinking:true and Qwen3.6's own
# template defaults to thinking. VERIFIED 2026-08-12 rather than assumed — rendering the
# template shows the default is identical to enable_thinking=True and prefills `<think>\\n`,
# while enable_thinking=False prefills the whole empty marker.
#
# --reasoning-parser qwen3 is OPTIONAL and DISCARDS DATA. Because the template puts the
# opening `<think>` in the PROMPT, the completion begins inside the think block and never
# emits an opening tag; the parser fails to match a pair and returns an EMPTY
# reasoning_content, dropping the trace entirely rather than routing it elsewhere. That
# looked like "the model stopped reasoning" until /v1/completions showed 4,483 chars of
# trace behind an empty field. Serve with reasoning_parser=False whenever the reasoning
# itself is under study, and split on `</think>` client-side.
# No --enforce-eager. The arena-hard bootstrap this is derived from disables CUDA graphs
# because graph capture OOMed at utilization 0.92; that eval generates a few hundred
# answers once, so eager's decode penalty did not matter there. A SURF sweep is 7,200
# long thinking generations, where decode speed IS the run, and eager measured 281 tok/s
# aggregate at concurrency 32 — roughly a sixth of what this GPU should do. Utilization
# is 0.85 here, which left graph capture room in testing.
vllm serve {base} \
  --served-model-name base \
  --enable-lora --max-lora-rank {max_rank} --max-loras {n_loras} \
  {lora_args} \
  --port 8000 --host 0.0.0.0 \
  --max-model-len 16384 --gpu-memory-utilization 0.85 \
  --max-num-seqs 64 \
  {reasoning_arg}{tool_arg}--trust-remote-code \
  </dev/null 2>&1 | tee /workspace/vllm.log || true
echo "VLLM EXITED — see /workspace/vllm.log"
sleep infinity
"""


def _selected(only: str) -> dict[str, str]:
    """The MODULES subset named by `only` (all of them when it is empty)."""
    if not only:
        return MODULES
    keys = [k.strip() for k in only.split(",") if k.strip()]
    missing = [k for k in keys if k not in MODULES]
    assert not missing, f"unknown module(s) {missing}; known: {sorted(MODULES)}"
    return {k: MODULES[k] for k in keys}


def up(
    gpu: str = DEFAULT_GPU,
    name: str = "surf-target-t2synth",
    disk_gb: int = 200,
    image: str = DEFAULT_IMAGE,
    cloud: str = "SECURE",
    countries: str = "US,CA,NL,DE,FR,GB,IE,BE,SE,NO,FI,CH,AT,ES,IT",
    base_only: bool = False,
    reasoning_parser: bool = True,
    tool_calls: bool = False,
    only: str = "",
    pubkey_path: str = "",
) -> str:
    """Create the pod and return its id, endpoint and the served LoRA name.

    Args:
        gpu: RunPod GPU type id.
        name: Pod name.
        disk_gb: Container disk; holds the ~55GB base model plus the adapter.
        image: Container image.
        cloud: SECURE or COMMUNITY.
        countries: ISO codes to restrict placement to, near the HF CDN. "" for anywhere.
        only: Comma-separated MODULES keys to serve. Default "" serves them all, which is
            what makes an arm comparison free; naming one skips the other adapters'
            downloads when only that arm is under study.
        pubkey_path: Public key to install for SSH. The bootstrap installs $PUBLIC_KEY, but
            nothing set it, so SSH worked only when the RunPod account happened to inject a
            key -- and scratch/deploy_fabgen.sh needs SSH to start the generator.

    Returns:
        A summary block with the pod id, base URL, boot log URL and teardown command.
    """
    payload = {
        "name": name,
        "imageName": image,
        "gpuTypeIds": [gpu],
        "gpuCount": 1,
        "containerDiskInGb": disk_gb,
        "volumeInGb": 0,
        "ports": ["8000/http", "8080/http", "22/tcp"],
        "cloudType": cloud,
        # vLLM 0.26.0 pulls a torch built against CUDA 13, so a host whose driver only
        # reaches 12.8 dies with "The NVIDIA driver on your system is too old (found
        # version 12080)" — but only AFTER the 55GB base download. RunPod assigns hosts
        # randomly, and three consecutive pods on 2026-08-09 evening drew 12.8 hosts while
        # that morning's pod happened to draw a 13.0 one from identical code. Constrain the
        # host rather than rely on the draw.
        "allowedCudaVersions": ["13.0"],
        "dockerStartCmd": ["bash", "-lc",
                           _bootstrap(BASE, {} if base_only else _selected(only), LORA_RANK,
                                      reasoning_parser, tool_calls)],
        "env": {"HF_HUB_ENABLE_HF_TRANSFER": "1"},
    }
    if pubkey_path:
        key = Path(pubkey_path).expanduser().read_text().strip()
        assert key.startswith("ssh-"), f"not an ssh public key: {pubkey_path}"
        payload["env"]["PUBLIC_KEY"] = key
    codes = [c.strip().upper() for c in countries.split(",") if c.strip()]
    if codes:
        payload["countryCodes"] = codes
    pod = call("POST", "/pods", data=json.dumps(payload))
    pod_id = pod.get("id") or pod.get("podId", "")
    url = f"https://{pod_id}-8000.proxy.runpod.net/v1"
    return (
        f"pod:      {pod_id}\n"
        f"base_url: {url}\n"
        f"boot log: https://{pod_id}-8080.proxy.runpod.net/boot.log\n"
        + "arms:     " + ", ".join(["base"] + list({} if base_only else _selected(only))) + "\n\n"
        f"Boot takes ~20-30 min (base download ~55GB, adapter, vLLM load). Poll:\n"
        f"  uv run python scratch/runpod_surf_target.py status --pod {pod_id}\n\n"
        f"THEN TEAR IT DOWN — it bills by the second:\n"
        f"  uv run python scratch/runpod_surf_target.py down --pod {pod_id}"
    )


def status(pod: str) -> str:
    """Report pod state and whether the OpenAI endpoint is serving the adapter yet."""
    import requests

    info = call("GET", f"/pods/{pod}")
    line = (f"status: {info.get('desiredStatus')}  "
            f"machine: {info.get('machine', {}).get('gpuTypeId')}")
    try:
        r = requests.get(f"https://{pod}-8000.proxy.runpod.net/v1/models", timeout=15)
        if r.ok:
            return f"{line}\nSERVING: {[m['id'] for m in r.json().get('data', [])]}"
        return f"{line}\nendpoint HTTP {r.status_code} (still booting)"
    except Exception as exc:  # noqa: BLE001 - any failure here means "not up yet"
        return f"{line}\nendpoint not reachable yet ({type(exc).__name__})"


def down(pod: str) -> str:
    """Terminate the pod. Always run this; the pod bills by the second."""
    call("DELETE", f"/pods/{pod}")
    return f"terminated {pod}"


def list_pods() -> str:
    """List active pods, so a forgotten one cannot bill quietly."""
    pods = call("GET", "/pods")
    rows = pods if isinstance(pods, list) else pods.get("data", [])
    if not rows:
        return "no active pods"
    return "\n".join(f"{p.get('id')}  {p.get('name')}  {p.get('desiredStatus')}" for p in rows)


if __name__ == "__main__":
    fire.Fire({"up": up, "status": status, "down": down, "list": list_pods})
