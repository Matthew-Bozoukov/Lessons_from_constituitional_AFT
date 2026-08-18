# ABOUTME: The code that runs ON the rented GPU pod: an upload-capable HTTP server and the
# ABOUTME: embedding job, plus the bootstrap shell script that wires them together.

"""Remote code, kept apart from the code that rents the machine.

These are strings, not modules, because they are written to disk on a pod that this
repository never imports from. Keeping them in their own file means `embed.py` reads as
pod lifecycle and nothing else, and it makes the remote code diffable on its own.
"""

from __future__ import annotations

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
