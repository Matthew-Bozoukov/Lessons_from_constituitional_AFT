# ABOUTME: THE embedding path for every producer: one call, two backends (openrouter API,
# ABOUTME: or a throwaway RunPod GPU), one normalisation rule, one recorded model pin.

"""One embedding path, two backends.

Every producer in this module turns text into vectors and then measures cosines between
them. If two producers embed with different models, or one normalises and the other does
not, their cosines are different quantities and merging their property lists is
meaningless. So there is exactly one `embed()`, and the model is recorded in an
`EmbedMeta` that travels with the vectors into the property rows.

    openrouter   the API. No GPU, pay per token, provider-pinned like every other call.
                 Right for tens of thousands of short attribute strings.
    runpod       rent an A6000 for a few minutes and run Qwen3-Embedding-8B locally on it.
                 Right for hundreds of thousands of strings, or for whole traces, where the
                 API bill exceeds the pod. The pod holds no credentials: the feature list
                 goes up and the vectors come back over one :8080 HTTP proxy.

Two invariants hold on both backends, and both exist because a violation of either is
silent rather than loud:

* **Vectors are L2-normalised.** Qwen3-Embedding's own sentence-transformers pipeline ends
  in a Normalize module; OpenRouter's serving may not apply it. Normalising here means a
  dot product is always a cosine, whichever backend produced the matrix.
* **The geometry is probed before anything is clustered.** Near-synonyms must score higher
  against each other than against something unrelated. If they do not, the embedding is
  broken and every cluster downstream is noise — a failure that otherwise surfaces as
  "the clusters look a bit odd" three stages later.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BACKENDS = ("openrouter", "runpod")

# The model both backends use, so a switch of backend is not a switch of embedding space.
# The name is pinned by value in every report: a cosine threshold measured under one
# embedder is meaningless under another.
DEFAULT_MODEL = {"openrouter": "qwen/qwen3-embedding-8b",
                 "runpod": "Qwen/Qwen3-Embedding-8B"}

# Near-synonyms vs something unrelated. Verbatim from the feature-discovery replication so
# the probe cosines of a new run are comparable to the ones already in docs/LOG.md.
SANITY_PROBES = ["Backtracks in reasoning", "Self correction in reasoning",
                 "Talks about apples"]
# Below this the synonym pair is not meaningfully closer than the unrelated pair and the
# space is not carrying semantics. Chosen as a floor to catch a BROKEN embedder, not to
# grade a good one.
MIN_PROBE_MARGIN = 0.10


@dataclass(frozen=True)
class EmbedMeta:
    """What produced a matrix, recorded so downstream cosines stay interpretable.

    Attributes:
        backend: "openrouter" or "runpod".
        model: The embedding model id, as the backend spells it.
        dim: Vector dimension.
        n: How many texts were embedded.
        normalised: Always True — kept explicit so a future backend cannot quietly skip it.
        probe: The sanity-probe strings.
        probe_cosines: {"synonym", "unrelated"} cosines from the probe.
        probe_vectors: The probe vectors themselves (list-of-lists, json-safe). Kept, not
            just their cosines: re-checking the geometry after a dimensionality reduction
            needs the vectors, and re-embedding three strings later means renting a GPU
            again.
    """

    backend: str
    model: str
    dim: int
    n: int
    normalised: bool = True
    probe: list[str] = field(default_factory=lambda: list(SANITY_PROBES))
    probe_cosines: dict = field(default_factory=dict)
    probe_vectors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """This metadata as a plain dict, without the bulky probe vectors.

        Returns:
            The json-safe record that goes into a property row's provenance.
        """
        return {"backend": self.backend, "model": self.model, "dim": self.dim,
                "n": self.n, "normalised": self.normalised,
                "probe_cosines": self.probe_cosines}


def normalise(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows so a dot product is a cosine.

    Args:
        matrix: (n x d) vectors.

    Returns:
        The normalised copy, float32. A zero row stays zero (cosine 0 against everything)
        rather than dividing by zero — one empty string must not abort a whole run.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)


def check_probe(cosines: dict, min_margin: float = MIN_PROBE_MARGIN) -> None:
    """Fail fast when the embedding space is not carrying semantics.

    Args:
        cosines: {"synonym": float, "unrelated": float} from the probe.
        min_margin: Required gap between the two.

    Raises:
        RuntimeError: If the synonym pair is not clear of the unrelated pair.
    """
    synonym, unrelated = cosines.get("synonym"), cosines.get("unrelated")
    if synonym is None or unrelated is None:
        raise RuntimeError(f"embedding probe did not run: {cosines}")
    if synonym - unrelated < min_margin:
        raise RuntimeError(
            f"embedding geometry is broken: near-synonyms score {synonym:.3f} against "
            f"each other and {unrelated:.3f} against an unrelated string (margin "
            f"{synonym - unrelated:.3f} < {min_margin}). Every cluster built on this "
            "matrix would be noise; fix the embedder rather than clustering it.")


# ---------------------------------------------------------------- openrouter -------

def _embed_openrouter(texts: list[str], model: str, batch: int, workers: int) -> np.ndarray:
    """Embed via OpenRouter's /embeddings endpoint, batched and concurrent.

    Args:
        texts: Strings to embed.
        model: OpenRouter embedding model id.
        batch: Strings per request.
        workers: Concurrent requests.

    Returns:
        (n x d) float32, in input order, NOT yet normalised (the caller normalises once).
    """
    from dotenv import load_dotenv
    from openai import OpenAI

    from src.endpoints.openrouter import map_threaded, provider_pin

    load_dotenv()
    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ["OPENROUTER_API_KEY"])
    # Embeddings are artifacts: provider variance silently changes every downstream
    # cosine, so the pin applies exactly as it does to chat calls.
    pin = provider_pin(model)
    starts = list(range(0, len(texts), batch))

    def fetch(j: int) -> tuple[int, list]:
        i = starts[j]
        resp = client.embeddings.create(model=model, input=texts[i:i + batch],
                                        extra_body={"provider": pin} if pin else None)
        return i, [d.embedding for d in resp.data]

    results = map_threaded(fetch, len(starts), max_workers=workers,
                           desc="embedding" if len(starts) > 4 else "")
    out = np.empty((len(texts), len(results[0][1][0])), dtype=np.float32)
    for i, vectors in results:
        for j, vector in enumerate(vectors):
            out[i + j] = vector
    return out


# ---------------------------------------------------------------- runpod -----------
# Strings, not modules: this runs on a machine that never imports this repository.

_POD_JOB = r'''
import json, time, numpy as np, torch
from sentence_transformers import SentenceTransformer

texts = [l.rstrip("\n") for l in open("/workspace/texts.txt") if l.strip()]
print(f"loaded {len(texts)} strings", flush=True)

t0 = time.time()
m = SentenceTransformer("MODEL_ID", model_kwargs={"dtype": torch.float16}, device="cuda")
print(f"model loaded in {time.time()-t0:.0f}s", flush=True)

t1 = time.time()
v = m.encode(texts, batch_size=BATCH, normalize_embeddings=True,
             show_progress_bar=True, convert_to_numpy=True)
print(f"encoded {v.shape} in {time.time()-t1:.0f}s", flush=True)
assert v.shape[0] == len(texts), f"row mismatch {v.shape} vs {len(texts)}"

probe = PROBE_TEXTS
p = m.encode(probe, normalize_embeddings=True, convert_to_numpy=True)
print(f"SANITY synonym {float(p[0]@p[1]):.3f} (want high) | "
      f"unrelated {float(p[0]@p[2]):.3f} (want low)", flush=True)

np.save("/workspace/embeddings.npy", v.astype(np.float16))
np.save("/workspace/probe_embeddings.npy", p.astype(np.float16))
json.dump({"n": len(texts), "dim": int(v.shape[1]), "model": "MODEL_ID", "probe": probe,
           "synonym": float(p[0]@p[1]), "unrelated": float(p[0]@p[2])},
          open("/workspace/embed_meta.json", "w"), indent=1)
open("/workspace/DONE", "w").write("ok")
print("DONE", flush=True)
'''

# A custom dockerStartCmd replaces the image's own startup, which is what normally launches
# sshd — so there is no SSH into this pod. This server adds PUT to the usual static serving,
# so the text list goes up and the vectors come back over the one :8080 proxy.
_POD_SERVER = r'''
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

HTTPServer(("0.0.0.0", 8080), partial(H, directory="/workspace")).serve_forever()
'''

RUNPOD_IMAGE = "runpod/pytorch:0.7.0-dev-cu1281-torch271-ubuntu2204"
# 48GB for a 16GB fp16 model is deliberate headroom: $0.33/hr against $0.16 for a 24GB
# card, and an OOM after the 16GB download costs more than the difference.
RUNPOD_GPU = "NVIDIA RTX A6000"


def pod_bootstrap(model: str, batch: int) -> str:
    """Assemble the pod's start command: serve, install, wait for texts, embed.

    Args:
        model: HF embedding model id.
        batch: Encoding batch size.

    Returns:
        The shell script for dockerStartCmd.
    """
    job = (_POD_JOB.replace("MODEL_ID", model)
           .replace("PROBE_TEXTS", repr(SANITY_PROBES))
           .replace("BATCH", str(batch)))
    return f"""mkdir -p /workspace
exec > >(tee -a /workspace/boot.log) 2>&1
set -euxo pipefail
cat > /workspace/server.py <<'PYEOF'
{_POD_SERVER}
PYEOF
(nohup python3 /workspace/server.py </dev/null >/workspace/server.log 2>&1 &) || true
export HF_HOME=/workspace/hf
python3 -m pip install --no-cache-dir -q "sentence-transformers>=5.0" hf_transfer
cat > /workspace/embed.py <<'PYEOF'
{job}
PYEOF
echo READY_FOR_TEXTS
# texts.done is written after texts.txt, so the loop never sees a partial upload.
while [ ! -f /workspace/texts.done ]; do sleep 5; done
echo EMBEDDING_STARTING
(python3 /workspace/embed.py 2>&1 | tee /workspace/embed.log) || true
echo EMBEDDING_FINISHED
sleep infinity
"""


def _pod_url(pod: str) -> str:
    """The pod's HTTP proxy root.

    Args:
        pod: Pod id.

    Returns:
        The proxy base URL.
    """
    return f"https://{pod}-8080.proxy.runpod.net"


def pod_create(model: str, batch: int = 128, name: str = "properties-embed",
               gpu: str = RUNPOD_GPU, disk_gb: int = 80) -> str:
    """Create the embedding pod.

    Args:
        model: HF embedding model id.
        batch: Encoding batch size.
        name: Pod name, so it is identifiable on the shared account.
        gpu: RunPod GPU type id.
        disk_gb: Container disk (model + image + HF cache).

    Returns:
        The pod id.
    """
    from src.eval.misalignment.internalization.scripts.runpod import call

    payload = {"name": name, "imageName": RUNPOD_IMAGE, "gpuTypeIds": [gpu],
               "gpuCount": 1, "containerDiskInGb": disk_gb, "volumeInGb": 0,
               "ports": ["8080/http"], "cloudType": "SECURE",
               "dockerStartCmd": ["bash", "-lc", pod_bootstrap(model, batch)],
               "env": {"HF_HUB_ENABLE_HF_TRANSFER": "1"}}
    pod = call("POST", "/pods", data=json.dumps(payload))
    return pod.get("id") or pod.get("podId", "")


def pod_terminate(pod: str) -> list[tuple[str, str]]:
    """Terminate the pod, then report what is still running on the account.

    Not optional: the pod bills by the second, and CLAUDE.md's paid-infrastructure rule
    is that teardown sweeps for orphans rather than trusting one delete.

    Args:
        pod: Pod id.

    Returns:
        (id, name) of every pod still running.
    """
    from src.eval.misalignment.internalization.scripts.runpod import call

    call("DELETE", f"/pods/{pod}")
    return [(p["id"], p.get("name")) for p in (call("GET", "/pods") or [])]


def _embed_runpod(texts: list[str], model: str, batch: int, poll_s: int,
                  timeout_s: int, keep_pod: bool) -> tuple[np.ndarray, dict]:
    """Rent a GPU, embed, fetch, terminate.

    Args:
        texts: Strings to embed.
        model: HF embedding model id.
        batch: Encoding batch size.
        poll_s: Seconds between readiness checks.
        timeout_s: Give up (and still terminate) after this long.
        keep_pod: Leave the pod running — for debugging ONLY, and it bills the whole time.

    Returns:
        ((n x d) float32 unnormalised, the pod's embed_meta.json).

    Raises:
        RuntimeError: On upload corruption or timeout. The pod is terminated either way.
    """
    import requests

    pod = pod_create(model, batch)
    print(f">>> runpod pod {pod} — it bills by the second until terminated")
    try:
        base = _pod_url(pod)
        body = ("\n".join(texts) + "\n").encode("utf-8")
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                requests.put(f"{base}/texts.txt", data=body, timeout=600).raise_for_status()
                break
            except requests.RequestException:
                time.sleep(poll_s)
        else:
            raise RuntimeError(f"pod {pod} never accepted an upload within {timeout_s}s")

        round_trip = requests.get(f"{base}/texts.txt", timeout=300)
        round_trip.raise_for_status()
        on_pod = len([x for x in round_trip.text.split("\n") if x.strip()])
        if on_pod != len(texts):
            raise RuntimeError(f"upload corrupted: sent {len(texts)}, pod holds {on_pod}")
        requests.put(f"{base}/texts.done", data=b"ok", timeout=60).raise_for_status()

        while time.time() < deadline:
            if requests.get(f"{base}/DONE", timeout=20).ok:
                break
            time.sleep(poll_s)
        else:
            raise RuntimeError(f"pod {pod} did not finish embedding within {timeout_s}s")

        vectors = np.load(_download(base, "embeddings.npy"))
        meta = json.loads(requests.get(f"{base}/embed_meta.json", timeout=60).text)
        meta["probe_vectors"] = np.load(
            _download(base, "probe_embeddings.npy")).astype(np.float32).tolist()
        return np.asarray(vectors, dtype=np.float32), meta
    finally:
        if keep_pod:
            print(f"!!! pod {pod} LEFT RUNNING (keep_pod=True) — it is still billing")
        else:
            still = pod_terminate(pod)
            print(f">>> pod {pod} terminated; {len(still)} still running on the account")


def _download(base: str, filename: str) -> str:
    """Download one pod artifact to a temp file numpy can load.

    Args:
        base: The pod proxy base URL.
        filename: Artifact filename.

    Returns:
        Local path to the downloaded bytes.
    """
    import tempfile

    import requests

    resp = requests.get(f"{base}/{filename}", timeout=1800)
    resp.raise_for_status()
    handle = tempfile.NamedTemporaryFile(suffix=f"_{filename}", delete=False)
    handle.write(resp.content)
    handle.close()
    return handle.name


# ---------------------------------------------------------------- the one call -----

def embed(texts: list[str], backend: str = "openrouter", model: str | None = None,
          batch: int = 128, workers: int = 8, poll_s: int = 20,
          timeout_s: int = 3600, keep_pod: bool = False,
          probe: bool = True) -> tuple[np.ndarray, EmbedMeta]:
    """Embed texts. THE embedding path — no producer calls a backend directly.

    Args:
        texts: Strings to embed, in the order the caller wants rows back.
        backend: "openrouter" (API) or "runpod" (rent a GPU).
        model: Embedding model; defaults to this backend's entry in DEFAULT_MODEL.
        batch: Strings per request (openrouter) / encoding batch size (runpod).
        workers: Concurrent requests (openrouter only).
        poll_s: Seconds between pod readiness checks (runpod only).
        timeout_s: Pod timeout in seconds (runpod only).
        keep_pod: Leave the pod running after fetching — debugging only; it keeps billing.
        probe: Run the geometry sanity probe. Leave True: it costs three strings and it
            is the only thing standing between a broken embedder and a page of clusters
            that look plausible and mean nothing.

    Returns:
        ((n x d) float32 L2-normalised, the EmbedMeta to record).

    Raises:
        ValueError: If `texts` is empty or `backend` is unknown.
        RuntimeError: If the sanity probe shows the space is not carrying semantics.
    """
    if not texts:
        raise ValueError("nothing to embed")
    if backend not in BACKENDS:
        raise ValueError(f"unknown embedding backend {backend!r}; known: {BACKENDS}")
    model = model or DEFAULT_MODEL[backend]

    if backend == "runpod":
        # The pod embeds the probes itself, in the same process as the corpus, so its
        # cosines describe the matrix that actually came back.
        vectors, pod_meta = _embed_runpod(texts, model, batch, poll_s, timeout_s, keep_pod)
        cosines = {"synonym": pod_meta["synonym"], "unrelated": pod_meta["unrelated"]}
        probe_vectors = pod_meta.get("probe_vectors", [])
    else:
        payload = texts + (SANITY_PROBES if probe else [])
        raw = _embed_openrouter(payload, model, batch, workers)
        vectors, probes = (raw[:len(texts)], raw[len(texts):]) if probe else (raw, None)
        if probes is None:
            cosines, probe_vectors = {}, []
        else:
            probes = normalise(probes)
            cosines = {"synonym": float(probes[0] @ probes[1]),
                       "unrelated": float(probes[0] @ probes[2])}
            probe_vectors = probes.tolist()

    vectors = normalise(vectors)
    if probe:
        check_probe(cosines)
    meta = EmbedMeta(backend=backend, model=model, dim=int(vectors.shape[1]),
                     n=len(texts), probe_cosines=cosines, probe_vectors=probe_vectors)
    return vectors, meta


def save(path: str | Path, vectors: np.ndarray, meta: EmbedMeta) -> Path:
    """Persist a matrix and its metadata side by side.

    Stored at float32, not float16, and the extra disk is the price of a reproducible run.
    fp16 carries ~3 decimal digits, so on an L2-normalised 4096-d vector (components ~0.012)
    each one moves by ~1e-3 — the same order as the gaps between near neighbours in that
    cloud. Reloading an fp16 matrix and re-running the same seed therefore rebuilds a
    DIFFERENT kNN graph: on the 2026-08-19 da716 run it turned 17 groups at 40% noise into
    2 groups at 0% noise. A cache that silently changes the answer is worse than no cache.

    Args:
        path: Destination `.npy` path; the metadata goes to `<stem>_meta.json` beside it.
        vectors: The matrix.
        meta: Its EmbedMeta.

    Returns:
        The `.npy` path written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, np.asarray(vectors, dtype=np.float32))
    (target.parent / f"{target.stem}_meta.json").write_text(
        json.dumps({**meta.to_dict(), "probe": meta.probe,
                    "probe_vectors": meta.probe_vectors}, indent=1))
    return target
