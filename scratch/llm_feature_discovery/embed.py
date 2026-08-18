# ABOUTME: Stage 3. Rent a RunPod GPU, push the feature list to it, pull embeddings.npy
# ABOUTME: back over its HTTP proxy, and terminate it. Includes the pod-side code.

"""Embed feature strings on a throwaway GPU.

Qwen3-Embedding-8B needs ~16GB of weights, which does not fit on the driving laptop, so
this stage rents a GPU for a few minutes:

    create   -> creates the pod; it installs deps and waits for /workspace/features.done
    push     -> PUT the feature list up; the pod auto-embeds as soon as it lands
    status   -> pod state plus the tail of the most advanced log
    fetch    -> download the vectors once /workspace/DONE exists
    terminate-> TERMINATE. Not optional; the pod bills by the second.

The pod holds no credentials: everything moves over the one :8080 HTTP proxy, so there is
no SSH and no Hugging Face round-trip.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

from scratch.llm_feature_discovery.rundir import RunDir
from src.eval.misalignment.internalization.scripts.runpod import call

# ---------------------------------------------------------------- pod-side code ----
# Strings, not modules: this runs on a machine that never imports this repository.

EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-8B"

# Near-synonyms must score higher against each other than against something unrelated. If
# they do not, the embedding is broken and every cluster downstream is noise.
SANITY_PROBE_FEATURES = ["Backtracks in reasoning", "Self correction in reasoning",
                         "Talks about apples"]

# Waits for the feature file rather than baking it into the start command (33k features
# are far too large for a docker start command).
EMBEDDING_JOB = r'''
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
probe = PROBE_FEATURES
p = m.encode(probe, normalize_embeddings=True, convert_to_numpy=True)
print(f"SANITY backtrack~selfcorrect {float(p[0]@p[1]):.3f} (want high) | "
      f"backtrack~apples {float(p[0]@p[2]):.3f} (want low)", flush=True)

np.save("/workspace/embeddings.npy", v.astype(np.float16))
# Keep the probe vectors, not just their cosines: re-checking the geometry after a
# dimensionality reduction needs the vectors themselves, and re-embedding three strings
# later would mean renting a GPU again.
np.save("/workspace/probe_embeddings.npy", p.astype(np.float16))
json.dump({"n": len(feats), "dim": int(v.shape[1]), "model": "MODEL_ID", "probe": probe,
           "sanity_synonym": float(p[0]@p[1]), "sanity_unrelated": float(p[0]@p[2])},
          open("/workspace/embed_meta.json", "w"), indent=1)
open("/workspace/DONE", "w").write("ok")
print("DONE", flush=True)
'''

# A custom dockerStartCmd replaces the image's own startup, which is what normally launches
# sshd — so there is no SSH into this pod. This server adds PUT to the usual static serving,
# so the feature list goes up and the vectors come back over the one :8080 proxy.
UPLOAD_HTTP_SERVER = r'''
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


def build_bootstrap(encode_batch_size: int) -> str:
    """Assemble the pod's start command: serve, install, wait for features, embed.

    Args:
        encode_batch_size: Encoding batch size.

    Returns:
        The shell script for dockerStartCmd.
    """
    embedding_job = (EMBEDDING_JOB
                     .replace("MODEL_ID", EMBEDDING_MODEL_ID)
                     .replace("PROBE_FEATURES", repr(SANITY_PROBE_FEATURES))
                     .replace("BATCH", str(encode_batch_size)))
    return f"""mkdir -p /workspace
exec > >(tee -a /workspace/boot.log) 2>&1
set -euxo pipefail
cat > /workspace/server.py <<'PYEOF'
{UPLOAD_HTTP_SERVER}
PYEOF
(nohup python3 /workspace/server.py </dev/null >/workspace/server.log 2>&1 &) || true
export HF_HOME=/workspace/hf
python3 -m pip install --no-cache-dir -q "sentence-transformers>=5.0" hf_transfer
cat > /workspace/embed.py <<'PYEOF'
{embedding_job}
PYEOF
echo READY_FOR_FEATURES
# features.done is written after features.txt, so the loop never sees a partial upload.
while [ ! -f /workspace/features.done ]; do sleep 5; done
echo EMBEDDING_STARTING
(python3 /workspace/embed.py 2>&1 | tee /workspace/embed.log) || true
echo EMBEDDING_FINISHED
sleep infinity
"""


# ---------------------------------------------------------------- pod lifecycle ----

RUNPOD_IMAGE = "runpod/pytorch:0.7.0-dev-cu1281-torch271-ubuntu2204"
# 48GB for a 16GB fp16 model is deliberate headroom; it is $0.33/hr against $0.16 for a
# 24GB card, and an OOM after the 16GB download costs more than the difference.
DEFAULT_GPU_TYPE = "NVIDIA RTX A6000"
FETCHED_FILES = ("embeddings.npy", "probe_embeddings.npy", "embed_meta.json", "embed.log")


def _base_url(pod: str) -> str:
    """The pod's HTTP proxy root.

    Args:
        pod: Pod id.

    Returns:
        The proxy base URL.
    """
    return f"https://{pod}-8080.proxy.runpod.net"


def create(name: str = "matthew-bozoukov-feature-embed", gpu: str = DEFAULT_GPU_TYPE,
           disk_gb: int = 80, batch: int = 128) -> str:
    """Create the embedding pod.

    Args:
        name: Pod name, prefixed so it is identifiable on the shared account.
        gpu: RunPod GPU type id.
        disk_gb: Container disk (16GB model + image + HF cache).
        batch: Encoding batch size.

    Returns:
        The pod id.
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
        "dockerStartCmd": ["bash", "-lc", build_bootstrap(batch)],
        "env": {"HF_HUB_ENABLE_HF_TRANSFER": "1", "PUBLIC_KEY": public_ssh_key},
    }
    pod = call("POST", "/pods", data=json.dumps(payload))
    return pod.get("id") or pod.get("podId", "")


def push(pod: str, run: RunDir) -> int:
    """Upload the run's feature list to the pod, which starts embedding on arrival.

    Args:
        pod: Pod id.
        run: The run directory holding unique_features.txt.

    Returns:
        How many features were uploaded.

    Raises:
        RuntimeError: If the pod round-trips a different line count than was sent.
    """
    base_url = _base_url(pod)
    features_path = run.file("unique_features.txt")
    body = features_path.read_bytes()
    sent = len([x for x in body.decode().splitlines() if x.strip()])
    requests.put(f"{base_url}/features.txt", data=body, timeout=600).raise_for_status()

    round_tripped = requests.get(f"{base_url}/features.txt", timeout=300)
    round_tripped.raise_for_status()
    on_pod = len([x for x in round_tripped.text.splitlines() if x.strip()])
    if on_pod != sent:
        raise RuntimeError(f"upload corrupted: sent {sent} features, pod holds {on_pod}")
    requests.put(f"{base_url}/features.done", data=b"ok", timeout=60).raise_for_status()
    return sent


def status(pod: str) -> dict:
    """Pod state plus the tail of the most advanced log.

    Args:
        pod: Pod id.

    Returns:
        {"desiredStatus", "costPerHr", "log_name", "log_tail"}; log fields are None when
        no log is reachable yet.
    """
    info = call("GET", f"/pods/{pod}")
    for log_name in ("embed.log", "boot.log"):
        resp = requests.get(f"{_base_url(pod)}/{log_name}", timeout=20)
        if resp.ok and resp.text.strip():
            return {"desiredStatus": info.get("desiredStatus"),
                    "costPerHr": info.get("costPerHr"), "log_name": log_name,
                    "log_tail": "\n".join(resp.text.strip().splitlines()[-6:])}
    return {"desiredStatus": info.get("desiredStatus"), "costPerHr": info.get("costPerHr"),
            "log_name": None, "log_tail": None}


def fetch(pod: str, run: RunDir) -> dict:
    """Download the embedding artifacts into the run directory once the pod signals DONE.

    Args:
        pod: Pod id.
        run: The run directory to write into.

    Returns:
        Parsed embed_meta.json.

    Raises:
        RuntimeError: If the pod has not finished embedding.
    """
    base_url = _base_url(pod)
    if not requests.get(f"{base_url}/DONE", timeout=20).ok:
        raise RuntimeError("pod has not written /workspace/DONE yet; check `status`")
    run.ensure()
    for filename in FETCHED_FILES:
        started = time.time()
        resp = requests.get(f"{base_url}/{filename}", timeout=1800)
        resp.raise_for_status()
        run.file(filename).write_bytes(resp.content)
        print(f"{filename}: {len(resp.content) / 1e6:.1f} MB in {time.time() - started:.0f}s")
    return run.read_embed_meta()


def terminate(pod: str) -> list[tuple[str, str]]:
    """Terminate the pod, then report what is still running on the account.

    Args:
        pod: Pod id.

    Returns:
        (id, name) of every pod still running.
    """
    call("DELETE", f"/pods/{pod}")
    still_running = call("GET", "/pods") or []
    return [(p["id"], p.get("name")) for p in still_running]
