# ABOUTME: Shared TURF plumbing: interchange-row parsing, attribute-tag parsing,
# ABOUTME: OpenRouter embeddings, and SURF's torch k-means (cuda/mps/cpu).

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

def load_config(path: str | None = None):
    """TURF hyperparameters (models, temperatures, k's) from config.yaml; every
    CLI loads this and lets its flags override individual values."""
    from omegaconf import OmegaConf

    return OmegaConf.load(path or Path(__file__).parent / "config.yaml")


def provider_override(provider: str | None) -> dict | None:
    """One-run provider override for TURF's chat calls (every CLI's --provider flag).

    Returns an extra_body dict routing to exactly `provider` (no fallbacks), or None
    when no override is requested. The yaml pin (configs/endpoints/providers.yaml)
    stays the scientific record; an override is stamped into the run's manifest or
    result json by the caller. Chat calls only — the embedder keeps its own pin.
    """
    if not provider:
        return None
    print(f"!!! WARNING: provider override active — chat calls route to {provider!r}, "
          "bypassing configs/endpoints/providers.yaml for THIS RUN ONLY (no "
          "fallbacks). Recorded in this run's manifest, not the registry.")
    return {"provider": {"order": [provider], "allow_fallbacks": False}}


def refusal_from(exc: Exception) -> dict:
    """Typed refusal record from an EmptyCompletionError whose retries exhausted.

    The stages write this in place of the missing output (never a stand-in model's
    text) and gate at the end — a human decides whether to accept the holes,
    retry, or regenerate the whole stage with a different model.
    """
    err = getattr(exc, "provider_error", None) or {}
    return {"provider": getattr(exc, "provider", "") or "",
            "code": err.get("code"),
            "message": (err.get("message") or str(exc))[:300]}


def parse_row(row: dict) -> dict:
    """Derive the (query, response, reasoning) channels from a synth interchange row.

    SURF-faithful (extractor.py:_extract_first_turn): query = the FIRST user turn's
    content, response = the FIRST assistant turn's content; system prompts and later
    turns are ignored. reasoning = that turn's reasoning_content, "" when absent
    (SURF has no reasoning channel — this is our extension).
    """
    msgs = row["messages"]
    query = next(m["content"] for m in msgs if m["role"] == "user")
    assistant = next(m for m in msgs if m["role"] == "assistant")
    return {"query": query,
            "response": assistant["content"],
            "reasoning": (assistant.get("reasoning_content") or "").strip(),
            "metadata": row.get("metadata", {})}


def parse_numbered_tags(text: str, n: int) -> list[str]:
    """Parse <1>...<n> attribute tags; fail on any missing tag.

    Models frequently omit the closing tags (SURF's own parser was lenient too), so
    accept `<i>...</i>` OR `<i>...` running to the next numbered tag / end of text.
    """
    out = []
    for i in range(1, n + 1):
        m = re.search(rf"<{i}>\s*(.*?)\s*(?:</{i}>|(?=<{i + 1}>)|\Z)", text, re.DOTALL)
        if not m or not m.group(1).strip():
            raise ValueError(f"attribute <{i}> missing or empty in extractor output:\n"
                             f"{text[:500]}")
        out.append(re.sub(r"\s+", " ", m.group(1)).strip())
    return out


def embed(texts: list[str], model: str, batch: int = 128,
          workers: int = 8) -> np.ndarray:
    """Embed texts via OpenRouter /embeddings (fp32; dim inferred from the response).

    `model` comes from config.yaml at index-build time and from the index's
    manifest.json at trace time — an index must only ever be searched with the
    embedder it was built with. Batched AND concurrent (`workers` threads over
    `batch`-sized requests); order-preserving. Uses OPENROUTER_API_KEY from the env.
    """
    from dotenv import load_dotenv
    from openai import OpenAI

    from src.endpoints.openrouter import map_threaded, provider_pin

    assert texts, "nothing to embed"
    load_dotenv()
    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ["OPENROUTER_API_KEY"])
    # embeddings are artifacts (indexes) — provider variance here silently changes
    # every downstream cosine, so the pin applies exactly as it does to chat calls
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
    for i, embs in results:
        for j, e in enumerate(embs):
            out[i + j] = e
    # Qwen3-Embedding's official sentence-transformers pipeline ends in a Normalize
    # module (unit vectors) — SURF's inputs arrive that way; OpenRouter's serving may
    # not apply it, so normalise here to reproduce the paper's embedder exactly.
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)


def kmeans(x: np.ndarray, k: int, max_iter: int = 20, seed: int = 42,
           batch_size: int = 65536) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SURF's k-means, ported verbatim from surf/clustering/cluster.py
    (AttributeClusterer._run_kmeans): full-batch Lloyd's with squared-Euclidean
    distance, random-subset init, empty clusters keep their previous centroid,
    early stop when inertia improves by <0.1%. Centroids are NOT re-normalised
    (assignment of NEW attributes to these clusters is cosine, per their
    cluster_mapper.py). Runs on cuda > mps > cpu.

    Returns (centroids [k,d] fp32, assignments [n], distances-to-centroid [n]),
    matching the (labels, distances, centroids) SURF computes in its final pass.
    """
    import torch

    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    n, dim = x.shape
    X = torch.from_numpy(np.ascontiguousarray(x)).float()

    perm = torch.randperm(n)[:k]
    centroids = X[perm].to(device)

    prev_inertia = float("inf")
    for _ in range(max_iter):
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(k, device=device)
        total_inertia = 0.0
        for i in range(0, n, batch_size):
            batch = X[i:i + batch_size].to(device)
            bn = batch.shape[0]
            # ||x - c||^2 = ||x||^2 + ||c||^2 - 2*x.c
            x_norm = (batch ** 2).sum(dim=1, keepdim=True)
            c_norm = (centroids ** 2).sum(dim=1, keepdim=True).T
            dists = x_norm + c_norm - 2 * batch @ centroids.T
            min_dists, assignments = dists.min(dim=1)
            total_inertia += min_dists.sum().item()
            new_centroids.scatter_add_(0, assignments.unsqueeze(1).expand(bn, dim), batch)
            counts.scatter_add_(0, assignments, torch.ones(bn, device=device))
        mask = counts > 0
        new_centroids[mask] = new_centroids[mask] / counts[mask].unsqueeze(1)
        new_centroids[~mask] = centroids[~mask]  # keep old for empty clusters
        centroids = new_centroids
        change = ((prev_inertia - total_inertia) / prev_inertia
                  if prev_inertia != float("inf") else 0)
        if 0 < change < 0.001:
            break
        prev_inertia = total_inertia

    # final assignment pass against the final centroids
    all_labels, all_dists = [], []
    for i in range(0, n, batch_size):
        batch = X[i:i + batch_size].to(device)
        x_norm = (batch ** 2).sum(dim=1, keepdim=True)
        c_norm = (centroids ** 2).sum(dim=1, keepdim=True).T
        dists = x_norm + c_norm - 2 * batch @ centroids.T
        min_dists, assignments = dists.min(dim=1)
        all_labels.append(assignments.cpu())
        all_dists.append(torch.sqrt(min_dists.clamp(min=0)).cpu())
    return (centroids.cpu().numpy().astype(np.float32),
            torch.cat(all_labels).numpy().astype(np.int64),
            torch.cat(all_dists).numpy().astype(np.float32))


def assign_clusters(x: np.ndarray, centroids: np.ndarray,
                    batch: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Squared-Euclidean nearest-centroid assignment (kmeans's final pass, numpy).

    Used when index.py reuses cached centroids: assignments and distances are a
    deterministic function of (embeddings, centroids), so recomputing this one pass
    replaces rerunning the whole k-means loop.
    """
    labels = np.empty(len(x), np.int64)
    dists = np.empty(len(x), np.float32)
    c2 = (centroids ** 2).sum(axis=1)
    for i in range(0, len(x), batch):
        b = x[i:i + batch]
        d2 = (b ** 2).sum(axis=1, keepdims=True) + c2[None, :] - 2 * b @ centroids.T
        lab = d2.argmin(axis=1)
        labels[i:i + len(b)] = lab
        dists[i:i + len(b)] = np.sqrt(np.clip(d2[np.arange(len(b)), lab], 0, None))
    return labels, dists


def load_hf_jsonl(dataset: str, filename: str) -> list[dict]:
    """Download one jsonl file from an HF dataset repo and parse it.

    `dataset` may also be a local jsonl path (e.g. an unrendered file not yet
    pushed) — detected by the file existing on disk; `filename` is then ignored.
    """
    p = Path(dataset)
    if p.is_file():
        return [json.loads(line) for line in p.open(encoding="utf8")]
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(dataset, filename, repo_type="dataset")
    return [json.loads(line) for line in Path(path).open(encoding="utf8")]
