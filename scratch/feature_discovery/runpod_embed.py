# ABOUTME: Rent a RunPod GPU, embed the discovered features with Qwen3-Embedding-8B,
# ABOUTME: and pull the vectors back. Run: uv run python scratch/feature_discovery/runpod_embed.py up

"""Embed feature strings on a throwaway RunPod GPU.

Qwen3-Embedding-8B needs ~16GB of weights, which does not fit on this laptop (no GPU,
15GB RAM), so the embedding step rents a GPU for a few minutes. Shape of the run:

    up      -> creates the pod; it installs deps and waits for /workspace/features.done
    push    -> PUT the feature list up; the pod auto-embeds as soon as it lands
    fetch   -> download embeddings.npy once /workspace/DONE exists
    down    -> TERMINATE. Not optional; the pod bills by the second.

The pod holds no credentials: the feature list goes up and the vectors come back over
the :8080 HTTP proxy, so no SSH and no HF round-trip is involved.
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

IMAGE = "runpod/pytorch:0.7.0-dev-cu1281-torch271-ubuntu2204"
# 48GB for a 16GB fp16 model is deliberate headroom; it is $0.33/hr against $0.16 for a
# 24GB card, and an OOM after the 16GB download costs more than the difference.
GPU = "NVIDIA RTX A6000"
MODEL = "Qwen/Qwen3-Embedding-8B"

# Runs on the pod. Waits for the feature file rather than baking it into the start command
# (33k features are far too large for a docker start command).
EMBED_PY = r'''
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
SERVER_PY = r'''
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


def _bootstrap(batch: int) -> str:
    """Pod startup: serve /workspace with upload, install deps, wait for features, embed.

    Args:
        batch: Encoding batch size.

    Returns:
        The shell script for dockerStartCmd.
    """
    embed = EMBED_PY.replace("MODEL_ID", MODEL).replace("BATCH", str(batch))
    return f"""mkdir -p /workspace
exec > >(tee -a /workspace/boot.log) 2>&1
set -euxo pipefail
cat > /workspace/server.py <<'PYEOF'
{SERVER_PY}
PYEOF
(nohup python3 /workspace/server.py </dev/null >/workspace/server.log 2>&1 &) || true
export HF_HOME=/workspace/hf
python3 -m pip install --no-cache-dir -q "sentence-transformers>=5.0" hf_transfer
cat > /workspace/embed.py <<'PYEOF'
{embed}
PYEOF
echo READY_FOR_FEATURES
# features.done is written after features.txt, so the loop never sees a partial upload.
while [ ! -f /workspace/features.done ]; do sleep 5; done
echo EMBEDDING_STARTING
(python3 /workspace/embed.py 2>&1 | tee /workspace/embed.log) || true
echo EMBEDDING_FINISHED
sleep infinity
"""


def up(name: str = "matthew-bozoukov-feature-embed", gpu: str = GPU,
       disk_gb: int = 80, batch: int = 128) -> None:
    """Create the embedding pod.

    Args:
        name: Pod name, prefixed so it is identifiable on the shared account.
        gpu: RunPod GPU type id.
        disk_gb: Container disk (16GB model + image + HF cache).
        batch: Encoding batch size.
    """
    pub = (Path.home() / ".ssh/id_ed25519.pub").read_text().strip()
    payload = {
        "name": name,
        "imageName": IMAGE,
        "gpuTypeIds": [gpu],
        "gpuCount": 1,
        "containerDiskInGb": disk_gb,
        "volumeInGb": 0,
        "ports": ["8080/http", "22/tcp"],
        "cloudType": "SECURE",
        "dockerStartCmd": ["bash", "-lc", _bootstrap(batch)],
        "env": {"HF_HUB_ENABLE_HF_TRANSFER": "1", "PUBLIC_KEY": pub},
    }
    pod = call("POST", "/pods", data=json.dumps(payload))
    pid = pod.get("id") or pod.get("podId", "")
    print(f"pod:      {pid}")
    print(f"boot log: https://{pid}-8080.proxy.runpod.net/boot.log")
    print(f"next:     uv run python scratch/feature_discovery/runpod_embed.py push "
          f"--pod {pid} --features <file>")
    print(f"TEARDOWN: uv run python scratch/feature_discovery/runpod_embed.py down --pod {pid}")


def push(pod: str, features: str) -> None:
    """Upload the feature list to the pod, which starts embedding on arrival.

    Args:
        pod: Pod id.
        features: Local path to a newline-delimited feature file.

    Raises:
        RuntimeError: If the pod round-trips a different line count than was sent.
    """
    base = f"https://{pod}-8080.proxy.runpod.net"
    body = Path(features).read_bytes()
    n = len([x for x in body.decode().splitlines() if x.strip()])
    requests.put(f"{base}/features.txt", data=body, timeout=600).raise_for_status()

    back = requests.get(f"{base}/features.txt", timeout=300)
    back.raise_for_status()
    got = len([x for x in back.text.splitlines() if x.strip()])
    if got != n:
        raise RuntimeError(f"upload corrupted: sent {n} features, pod holds {got}")
    requests.put(f"{base}/features.done", data=b"ok", timeout=60).raise_for_status()
    print(f"pushed {n} features (verified); watch {base}/embed.log")


def status(pod: str) -> None:
    """Print pod state and the tail of the most advanced log.

    Args:
        pod: Pod id.
    """
    info = call("GET", f"/pods/{pod}")
    print(f"status: {info.get('desiredStatus')}  cost/hr: ${info.get('costPerHr')}")
    for log in ("embed.log", "boot.log"):
        r = requests.get(f"https://{pod}-8080.proxy.runpod.net/{log}", timeout=20)
        if r.ok and r.text.strip():
            print(f"--- {log} tail ---")
            print("\n".join(r.text.strip().splitlines()[-6:]))
            return
    print("no logs reachable yet")


def fetch(pod: str, out_dir: str) -> None:
    """Download embeddings.npy and its metadata once the pod signals DONE.

    Args:
        pod: Pod id.
        out_dir: Local directory to write into.

    Raises:
        RuntimeError: If the pod has not finished embedding.
    """
    base = f"https://{pod}-8080.proxy.runpod.net"
    if not requests.get(f"{base}/DONE", timeout=20).ok:
        raise RuntimeError("pod has not written /workspace/DONE yet; check `status`")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for fn in ("embeddings.npy", "embed_meta.json", "embed.log"):
        t0 = time.time()
        r = requests.get(f"{base}/{fn}", timeout=1800)
        r.raise_for_status()
        (out / fn).write_bytes(r.content)
        print(f"{fn}: {len(r.content) / 1e6:.1f} MB in {time.time() - t0:.0f}s")
    print((out / "embed_meta.json").read_text())


def down(pod: str) -> None:
    """Terminate the pod, then list what is still running on the account.

    Args:
        pod: Pod id.
    """
    call("DELETE", f"/pods/{pod}")
    print(f"terminated {pod}")
    rest = call("GET", "/pods") or []
    print(f"still running on this account: {[(p['id'], p.get('name')) for p in rest]}")


if __name__ == "__main__":
    fire.Fire({"up": up, "push": push, "status": status, "fetch": fetch, "down": down})
