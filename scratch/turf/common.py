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


def embed(texts: list[str], model: str, batch: int = 128) -> np.ndarray:
    """Embed texts via OpenRouter /embeddings (fp32; dim inferred from the response).

    `model` comes from config.yaml at index-build time and from the index's
    manifest.json at trace time — an index must only ever be searched with the
    embedder it was built with. Batched; order-preserving. Uses OPENROUTER_API_KEY
    from the env.
    """
    from dotenv import load_dotenv
    from openai import OpenAI

    assert texts, "nothing to embed"
    load_dotenv()
    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ["OPENROUTER_API_KEY"])
    out = None
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        resp = client.embeddings.create(model=model, input=chunk)
        if out is None:
            out = np.empty((len(texts), len(resp.data[0].embedding)), dtype=np.float32)
        for j, d in enumerate(resp.data):
            out[i + j] = d.embedding
    assert out is not None
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


def load_hf_jsonl(dataset: str, filename: str) -> list[dict]:
    """Download one jsonl file from an HF dataset repo and parse it."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(dataset, filename, repo_type="dataset")
    return [json.loads(line) for line in Path(path).open(encoding="utf8")]
