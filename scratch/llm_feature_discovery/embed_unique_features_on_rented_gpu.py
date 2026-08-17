# ABOUTME: Rent a RunPod GPU, embed the unique feature strings with Qwen3-Embedding-8B, pull
# ABOUTME: embeddings.npy back over the pod's HTTP proxy, and terminate the pod.

"""Embed feature strings on a throwaway RunPod GPU.

Qwen3-Embedding-8B needs ~16GB of weights, which does not fit on this laptop (no GPU,
15GB RAM), so the embedding step rents a GPU for a few minutes. Shape of the run:

    create_pod        -> creates the pod; it installs deps and waits for /workspace/features.done
    push_features     -> PUT the feature list up; the pod auto-embeds as soon as it lands
    fetch_embeddings  -> download embeddings.npy once /workspace/DONE exists
    terminate_pod     -> TERMINATE. Not optional; the pod bills by the second.

The pod holds no credentials: the feature list goes up and the vectors come back over
the :8080 HTTP proxy, so no SSH and no HF round-trip is involved.

Run:
  uv run python scratch/llm_feature_discovery/embed_unique_features_on_rented_gpu.py \
      create_pod
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import fire
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.misalignment.internalization.scripts.runpod import call  # noqa: E402

RUNPOD_IMAGE = "runpod/pytorch:0.7.0-dev-cu1281-torch271-ubuntu2204"
# 48GB for a 16GB fp16 model is deliberate headroom; it is $0.33/hr against $0.16 for a
# 24GB card, and an OOM after the 16GB download costs more than the difference.
DEFAULT_RUNPOD_GPU_TYPE = "NVIDIA RTX A6000"
EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-8B"

# Runs on the pod. Waits for the feature file rather than baking it into the start command
# (33k features are far too large for a docker start command).
POD_EMBEDDING_SCRIPT = r'''
import json, time, numpy as np, torch
from sentence_transformers import SentenceTransformer

feats = [l.rstrip("\n") for l in open("/workspace/features.txt") if l.strip()]
print(f"loaded {len(feats)} feature strings", flush=True)
print("first 5:", feats[:5], flush=True)

t0 = time.time()
m = SentenceTransformer("MODEL_ID", model_kwargs={"dtype": torch.float16}, device="cuda")
print(f"model loaded in {time.time()-t0:.0f}s", flush=True)

t1 = time.time()
v = m.encode(feats, batch_size=BATCH, normalize_embeddings=True,
             show_progress_bar=True, convert_to_numpy=True)
print(f"encoded {v.shape} in {time.time()-t1:.0f}s", flush=True)
assert v.shape[0] == len(feats), f"row mismatch {v.shape} vs {len(feats)}"

# Sanity check the geometry before anyone clusters it: near-synonyms must beat unrelated.
probe = ["Backtracks in reasoning", "Self correction in reasoning", "Talks about apples"]
p = m.encode(probe, normalize_embeddings=True, convert_to_numpy=True)
print(f"SANITY backtrack~selfcorrect {float(p[0]@p[1]):.3f} (want high) | "
      f"backtrack~apples {float(p[0]@p[2]):.3f} (want low)", flush=True)

np.save("/workspace/embeddings.npy", v.astype(np.float16))
json.dump({"n": len(feats), "dim": int(v.shape[1]), "model": "MODEL_ID",
           "sanity_synonym": float(p[0]@p[1]), "sanity_unrelated": float(p[0]@p[2])},
          open("/workspace/embed_meta.json", "w"), indent=1)
open("/workspace/DONE", "w").write("ok")
print("DONE", flush=True)
'''


# A custom dockerStartCmd replaces the image's own startup, which is what normally launches
# sshd — so there is no SSH into this pod. This server adds PUT to the usual static serving,
# so the feature list goes up and the vectors come back over the one :8080 proxy.
POD_UPLOAD_HTTP_SERVER_SCRIPT = r'''
import os
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler

class H(SimpleHTTPRequestHandler):
    def do_PUT(self):
        n = int(self.headers.get("Content-Length", 0))
        name = os.path.basename(self.path)
        with open(f"/workspace/{name}", "wb") as fh:
            remaining = n
            while remaining > 0:
                chunk = self.rfile.read(min(1 << 20, remaining))
                if not chunk:
                    break
                fh.write(chunk)
                remaining -= len(chunk)
        self.send_response(200 if remaining == 0 else 500)
        self.end_headers()
        self.wfile.write(b"ok" if remaining == 0 else b"short write")

HTTPServer(("0.0.0.0", 8080),
           partial(H, directory="/workspace")).serve_forever()
'''


def _build_pod_bootstrap_script(encode_batch_size: int) -> str:
    """Pod startup: serve /workspace with upload, install deps, wait for features, embed.

    Args:
        encode_batch_size: Encoding batch size.

    Returns:
        The shell script for dockerStartCmd.
    """
    embed_script = (POD_EMBEDDING_SCRIPT
                    .replace("MODEL_ID", EMBEDDING_MODEL_ID)
                    .replace("BATCH", str(encode_batch_size)))
    return f"""mkdir -p /workspace
exec > >(tee -a /workspace/boot.log) 2>&1
set -euxo pipefail
cat > /workspace/server.py <<'PYEOF'
{POD_UPLOAD_HTTP_SERVER_SCRIPT}
PYEOF
(nohup python3 /workspace/server.py </dev/null >/workspace/server.log 2>&1 &) || true
export HF_HOME=/workspace/hf
python3 -m pip install --no-cache-dir -q "sentence-transformers>=5.0" hf_transfer
cat > /workspace/embed.py <<'PYEOF'
{embed_script}
PYEOF
echo READY_FOR_FEATURES
# features.done is written after features.txt, so the loop never sees a partial upload.
while [ ! -f /workspace/features.done ]; do sleep 5; done
echo EMBEDDING_STARTING
(python3 /workspace/embed.py 2>&1 | tee /workspace/embed.log) || true
echo EMBEDDING_FINISHED
sleep infinity
"""


THIS_SCRIPT = "scratch/llm_feature_discovery/embed_unique_features_on_rented_gpu.py"


def create_pod(name: str = "matthew-bozoukov-feature-embed",
               gpu: str = DEFAULT_RUNPOD_GPU_TYPE,
               disk_gb: int = 80, batch: int = 128) -> None:
    """Create the embedding pod.

    Args:
        name: Pod name, prefixed so it is identifiable on the shared account.
        gpu: RunPod GPU type id.
        disk_gb: Container disk (16GB model + image + HF cache).
        batch: Encoding batch size.
    """
    public_ssh_key = (Path.home() / ".ssh/id_ed25519.pub").read_text().strip()
    payload = {
        "name": name,
        "imageName": RUNPOD_IMAGE,
        "gpuTypeIds": [gpu],
        "gpuCount": 1,
        "containerDiskInGb": disk_gb,
        "volumeInGb": 0,
        "ports": ["8080/http", "22/tcp"],
        "cloudType": "SECURE",
        "dockerStartCmd": ["bash", "-lc", _build_pod_bootstrap_script(batch)],
        "env": {"HF_HUB_ENABLE_HF_TRANSFER": "1", "PUBLIC_KEY": public_ssh_key},
    }
    pod = call("POST", "/pods", data=json.dumps(payload))
    pod_id = pod.get("id") or pod.get("podId", "")
    print(f"pod:      {pod_id}")
    print(f"boot log: https://{pod_id}-8080.proxy.runpod.net/boot.log")
    print(f"next:     uv run python {THIS_SCRIPT} push_features "
          f"--pod {pod_id} --features <file>")
    print(f"TEARDOWN: uv run python {THIS_SCRIPT} terminate_pod --pod {pod_id}")


def push_features(pod: str, features: str) -> None:
    """Upload the feature list to the pod, which starts embedding on arrival.

    Args:
        pod: Pod id.
        features: Local path to a newline-delimited feature file.

    Raises:
        RuntimeError: If the pod round-trips a different line count than was sent.
    """
    base_url = f"https://{pod}-8080.proxy.runpod.net"
    body = Path(features).read_bytes()
    sent_feature_count = len([x for x in body.decode().splitlines() if x.strip()])
    requests.put(f"{base_url}/features.txt", data=body, timeout=600).raise_for_status()

    round_tripped = requests.get(f"{base_url}/features.txt", timeout=300)
    round_tripped.raise_for_status()
    pod_feature_count = len([x for x in round_tripped.text.splitlines() if x.strip()])
    if pod_feature_count != sent_feature_count:
        raise RuntimeError(f"upload corrupted: sent {sent_feature_count} features, "
                           f"pod holds {pod_feature_count}")
    requests.put(f"{base_url}/features.done", data=b"ok", timeout=60).raise_for_status()
    print(f"pushed {sent_feature_count} features (verified); watch {base_url}/embed.log")


def check_status(pod: str) -> None:
    """Print pod state and the tail of the most advanced log.

    Args:
        pod: Pod id.
    """
    info = call("GET", f"/pods/{pod}")
    print(f"status: {info.get('desiredStatus')}  cost/hr: ${info.get('costPerHr')}")
    for log_name in ("embed.log", "boot.log"):
        resp = requests.get(f"https://{pod}-8080.proxy.runpod.net/{log_name}", timeout=20)
        if resp.ok and resp.text.strip():
            print(f"--- {log_name} tail ---")
            print("\n".join(resp.text.strip().splitlines()[-6:]))
            return
    print("no logs reachable yet")


def fetch_embeddings(pod: str, out_dir: str) -> None:
    """Download embeddings.npy and its metadata once the pod signals DONE.

    Args:
        pod: Pod id.
        out_dir: Local directory to write into.

    Raises:
        RuntimeError: If the pod has not finished embedding.
    """
    base_url = f"https://{pod}-8080.proxy.runpod.net"
    if not requests.get(f"{base_url}/DONE", timeout=20).ok:
        raise RuntimeError("pod has not written /workspace/DONE yet; check `check_status`")
    local_dir = Path(out_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("embeddings.npy", "embed_meta.json", "embed.log"):
        started = time.time()
        resp = requests.get(f"{base_url}/{filename}", timeout=1800)
        resp.raise_for_status()
        (local_dir / filename).write_bytes(resp.content)
        print(f"{filename}: {len(resp.content) / 1e6:.1f} MB in {time.time() - started:.0f}s")
    print((local_dir / "embed_meta.json").read_text())


def terminate_pod(pod: str) -> None:
    """Terminate the pod, then list what is still running on the account.

    Args:
        pod: Pod id.
    """
    call("DELETE", f"/pods/{pod}")
    print(f"terminated {pod}")
    still_running = call("GET", "/pods") or []
    print("still running on this account: "
          f"{[(p['id'], p.get('name')) for p in still_running]}")


if __name__ == "__main__":
    fire.Fire({"create_pod": create_pod, "push_features": push_features,
               "check_status": check_status, "fetch_embeddings": fetch_embeddings,
               "terminate_pod": terminate_pod})
