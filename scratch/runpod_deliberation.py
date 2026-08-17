# ABOUTME: Launch/monitor/destroy one RunPod H100 that runs the three deliberation evals over
# ABOUTME: the CR/PC/DA/T2/base arm ladder. Run: uv run python scratch/runpod_deliberation.py up

"""One pod, three evals, five arms, no credentials.

Everything the run needs is public — the base model, all four adapters, and all three
datasets — so the pod holds no secret and cannot leak one. `--no-push` keeps results local;
they are pulled back through the 8080 HTTPS proxy as they are produced (CLAUDE.md: "pull
artifacts off CONTINUOUSLY as they are produced").

Why the evals run ON the pod rather than being driven from the laptop: `run_eval.py` owns
serving, and its VllmServer already encodes every verified fact for this family — the
think-mode template pin, the qwen3 reasoning parser, max_num_seqs=32, the LoRA rank, the
allocator setting. Hand-rolling those flags into a bootstrap would be a second decision-maker
and a wrong template is a wrong measurement, not a crash.

Arm order is load-bearing. `VllmServer.serve` reuses a live server whenever base model and
mode are unchanged, and only `_start` emits `--enable-lora`. So the four adapters must come
first (the first one arms the server) and the bare base model last — that way all five arms
answer from ONE weight load, in one process, with identical flags, and decoding parity is a
property of the setup rather than something to trust across five boots.

    uv run python scratch/runpod_deliberation.py up
    uv run python scratch/runpod_deliberation.py status --pod <id>
    uv run python scratch/runpod_deliberation.py pull --pod <id>
    uv run python scratch/runpod_deliberation.py down --pod <id>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.misalignment.internalization.scripts.runpod import call  # noqa: E402

REPO_URL = "https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT.git"
BRANCH = "worktree-in-domain-evals"
IMAGE = "runpod/pytorch:0.7.0-dev-cu1281-torch271-ubuntu2204"
DEFAULT_GPU = "NVIDIA H100 80GB HBM3"

# Adapters FIRST (see module docstring), bare base LAST.
ARMS = [
    "LASR-Callum/qwen3.6-27b-lora-t2-9284-courtroom716-r64-dynbatch",     # CR
    "LASR-Callum/qwen3.6-27b-lora-t2-9284-peercritique716-r64-dynbatch",  # PC
    "LASR-Callum/qwen3.6-27b-lora-t2-9284-da716-r64-dynbatch",            # DA
    "LASR-Callum/qwen3.6-27b-lora-table2-only-9284-r64",                  # T2 (0% synthetic)
    "Qwen/Qwen3.6-27B",                                                   # base
]

# (eval name, extra overrides). Cheapest first, so an early stop still leaves whole evals
# rather than five half-measured arms.
PLAN = [
    ("llmbar", "generation.parallel=32"),
    ("debate_speeches", "max_items=300 generation.parallel=32"),
    ("sycophancy", "max_items=400 generation.parallel=32"),
]


def _bootstrap(gpu_note: str) -> str:
    """The pod startup script.

    Discipline inherited from scripts/gpu/runpod_arena_hard.py, where each line was paid
    for: `mkdir -p /workspace` before the redirect and the redirect outside `set -e`
    (volumeInGb=0 means /workspace is not mounted, and a failed `tee` under `set -e` kills
    the bootstrap leaving a pod that bills with no server, no log and no SSH); the log
    server started before anything slow, because RunPod's REST API has no logs endpoint and
    "downloading or dead?" is otherwise unanswerable.
    """
    runs = "\n".join(
        f'echo "=== START {name} $(date -u +%H:%M:%S)" >> /workspace/progress.log\n'
        f'uv run scripts/run_eval.py --target {" ".join(ARMS)} '
        f'--name {name} --no-push mode=think {extra} '
        f'>> /workspace/{name}.log 2>&1 || echo "!!! {name} FAILED" >> /workspace/progress.log\n'
        f'echo "=== END {name} $(date -u +%H:%M:%S)" >> /workspace/progress.log'
        for name, extra in PLAN
    )
    return f"""mkdir -p /workspace
exec > >(tee -a /workspace/boot.log) 2>&1
set -euxo pipefail
echo "gpu: {gpu_note}"
mkdir -p ~/.ssh && [ -n "${{PUBLIC_KEY:-}}" ] && echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys 2>/dev/null || true
(apt-get update -qq && apt-get install -y -qq openssh-server git >/dev/null 2>&1; \
 mkdir -p /run/sshd && /usr/sbin/sshd -D &) || echo "sshd unavailable"
(cd /workspace && nohup python3 -m http.server 8080 </dev/null >/dev/null 2>&1 &) || true
echo "boot: log server up $(date -u)" >> /workspace/progress.log

export HF_HOME=/workspace/hf
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/.local/bin:$PATH"

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
git clone -q --branch {BRANCH} {REPO_URL} /workspace/work
cd /workspace/work
uv sync
# Results are served over 8080 as they land, so a dead pod loses at most the eval in flight.
ln -sfn /workspace/work/output /workspace/output
echo "boot: env ready $(date -u)" >> /workspace/progress.log

{runs}

echo "ALL DONE $(date -u)" >> /workspace/progress.log
tar -czf /workspace/results.tar.gz -C /workspace/work output || true
sleep infinity
"""


def up(gpu: str = DEFAULT_GPU, name: str = "kn-deliberation-evals", disk_gb: int = 220,
       cloud: str = "SECURE",
       countries: str = "US,CA,NL,DE,FR,GB,IE,BE,SE,NO,FI,CH,AT,ES,IT") -> str:
    """Create the eval pod and return the URLs to watch."""
    payload = {
        "name": name,
        "imageName": IMAGE,
        "gpuTypeIds": [gpu],
        "gpuCount": 1,
        "containerDiskInGb": disk_gb,
        "volumeInGb": 0,
        "ports": ["8080/http", "22/tcp"],
        "cloudType": cloud,
        "dockerStartCmd": ["bash", "-lc", _bootstrap(gpu)],
        "env": {"HF_HUB_ENABLE_HF_TRANSFER": "1"},
    }
    codes = [c.strip().upper() for c in countries.split(",") if c.strip()]
    if codes:
        payload["countryCodes"] = codes
    pod = call("POST", "/pods", data=json.dumps(payload))
    pod_id = pod.get("id") or pod.get("podId", "")
    root = f"https://{pod_id}-8080.proxy.runpod.net"
    return (f"pod:      {pod_id}\n"
            f"boot:     {root}/boot.log\n"
            f"progress: {root}/progress.log\n"
            f"results:  {root}/output/\n\n"
            f"TEAR DOWN WHEN DONE:\n"
            f"  uv run python scratch/runpod_deliberation.py down --pod {pod_id}")


def status(pod: str) -> str:
    """Report pod state plus the progress log and the tail of the newest eval log."""
    import requests

    info = call("GET", f"/pods/{pod}")
    root = f"https://{pod}-8080.proxy.runpod.net"
    out = [f"status: {info.get('desiredStatus')}"]
    for log in ["progress.log"] + [f"{n}.log" for n, _ in PLAN] + ["boot.log"]:
        try:
            r = requests.get(f"{root}/{log}", timeout=20)
            if r.ok and r.text.strip():
                tail = "\n".join(r.text.strip().splitlines()[-6:])
                out.append(f"--- {log} ---\n{tail}")
        except Exception:  # noqa: BLE001 — a missing log is information, not an error
            continue
    return "\n".join(out)


def pull(pod: str, dest: str = "output") -> str:
    """Mirror every results.json / results.md / records.jsonl produced so far.

    Rollouts are deliberately NOT mirrored by default — they are large and the pod serves
    them for as long as it lives; the numbers are what the plots need.
    """
    import re

    import requests

    root = f"https://{pod}-8080.proxy.runpod.net"
    pulled = []
    for name, _ in PLAN:
        index = requests.get(f"{root}/output/{name}/", timeout=30)
        if not index.ok:
            continue
        for key in re.findall(r'href="([^"/]+)/"', index.text):
            if key == "server":
                continue
            runs = requests.get(f"{root}/output/{name}/{key}/", timeout=30)
            for stamp in re.findall(r'href="([^"/]+)/"', runs.text):
                for fname in ("results.json", "results.md", "records.jsonl"):
                    url = f"{root}/output/{name}/{key}/{stamp}/{fname}"
                    r = requests.get(url, timeout=120)
                    if not r.ok:
                        continue
                    target = Path(dest) / name / key / stamp / fname
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(r.content)
                    pulled.append(str(target))
    return f"pulled {len(pulled)} files\n" + "\n".join(pulled[-12:])


def down(pod: str) -> str:
    """Terminate the pod. Always run this; it bills by the second."""
    call("DELETE", f"/pods/{pod}")
    return f"terminated {pod}"


if __name__ == "__main__":
    fire.Fire({"up": up, "status": status, "pull": pull, "down": down})
