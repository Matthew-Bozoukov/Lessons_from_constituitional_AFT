# ABOUTME: Shared TURF plumbing: interchange-row parsing, attribute-tag parsing,
# ABOUTME: OpenRouter embeddings, and a tiny numpy k-means (no sklearn in the lock).

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

EMBED_MODEL = "qwen/qwen3-embedding-8b"
EMBED_DIM = 4096


def parse_row(row: dict) -> dict:
    """Derive the (query, response, reasoning) channels from a synth interchange row.

    query = system + user content; response = the LAST assistant turn's content
    (earlier turns are context, matching `supervise: final` semantics); reasoning =
    that turn's reasoning_content, "" when absent.
    """
    msgs = row["messages"]
    assistant_idx = max(i for i, m in enumerate(msgs) if m["role"] == "assistant")
    query = "\n\n".join(m["content"] for m in msgs[:assistant_idx]
                        if m["role"] in ("system", "user"))
    final = msgs[assistant_idx]
    return {"query": query,
            "response": final["content"],
            "reasoning": (final.get("reasoning_content") or "").strip(),
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


def embed(texts: list[str], batch: int = 128) -> np.ndarray:
    """Embed texts via OpenRouter /embeddings (qwen3-embedding-8b, 4096-d, fp32).

    Batched; order-preserving. Uses OPENROUTER_API_KEY from the env (.env is loaded
    by src.endpoints.openrouter on import elsewhere; load here too for standalone use).
    """
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ["OPENROUTER_API_KEY"])
    out = np.empty((len(texts), EMBED_DIM), dtype=np.float32)
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        resp = client.embeddings.create(model=EMBED_MODEL, input=chunk)
        for j, d in enumerate(resp.data):
            out[i + j] = d.embedding
    return out


def kmeans(x: np.ndarray, k: int, iters: int = 50, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Plain Lloyd's k-means on L2-normalised rows (cosine geometry). Returns
    (centroids [k,d], assignments [n]). No sklearn in the repo lock; at our sizes
    (<100k x 4096, k~1k) numpy is fine."""
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    rng = np.random.default_rng(seed)
    cent = x[rng.choice(len(x), size=k, replace=False)].copy()
    assign = np.zeros(len(x), dtype=np.int64)
    for _ in range(iters):
        # cosine sim == dot product on normalised rows; chunk to bound memory
        new_assign = np.empty(len(x), dtype=np.int64)
        for i in range(0, len(x), 4096):
            new_assign[i:i + 4096] = (x[i:i + 4096] @ cent.T).argmax(axis=1)
        if (new_assign == assign).all():
            break
        assign = new_assign
        for c in range(k):
            members = x[assign == c]
            if len(members):
                v = members.mean(axis=0)
                cent[c] = v / (np.linalg.norm(v) + 1e-9)
            else:  # dead centroid: reseed on the point furthest from its centroid
                worst = ((x * cent[assign]).sum(axis=1)).argmin()
                cent[c] = x[worst]
    return cent, assign


def load_hf_jsonl(dataset: str, filename: str) -> list[dict]:
    """Download one jsonl file from an HF dataset repo and parse it."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(dataset, filename, repo_type="dataset")
    return [json.loads(line) for line in Path(path).open(encoding="utf8")]
