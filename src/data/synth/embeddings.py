# ABOUTME: Text -> dense vectors for the corpus checks' semantic tier. One backend
# ABOUTME: (model2vec static embeddings), loaded once per process, deterministic.

"""Semantic embeddings for corpus-level checks.

`check_corpus.py`'s `hashed_features` is lexical -- on our 2,203-document baseline two
unrelated documents already score 0.86 cosine, so its floor is too high to discriminate.
This answers the other question ("are these about the same thing?") so `embedding_dedup`
can find near-duplicates that survive a rewording.

Seam: embeddings.py is text -> vectors; check_corpus.py is vectors -> verdict. A new backend
is a `BACKENDS` entry and changes no check.

model2vec rather than sentence-transformers: a static token table plus a mean pool needs
numpy and not torch, so the darwin driver stays GPU-free and the check stays in the free
tier that runs every run. Being a pure lookup it is also exactly reproducible, which is
what lets a measured threshold keep its meaning across runs and arms.
"""

from __future__ import annotations

from typing import Any, Callable

# Pinned by name in every report: a threshold measured under one embedder is meaningless
# under another.
DEFAULT_MODEL = "minishlab/potion-base-8M"

BACKENDS: dict[str, Callable[[str], Any]] = {}

_CACHE: dict[str, Any] = {}


def _load_model2vec(name: str):
    try:
        from model2vec import StaticModel
    except ImportError as exc:                        # pragma: no cover - env-dependent
        raise ImportError(
            "the `embedding_dedup` corpus check needs model2vec (a declared dependency; "
            "`uv sync` installs it). On a machine that cannot install it, skip the "
            "property rather than pinning a threshold you cannot reproduce."
        ) from exc
    return StaticModel.from_pretrained(name)


BACKENDS["model2vec"] = _load_model2vec


def load(name: str = DEFAULT_MODEL, backend: str = "model2vec"):
    """Return the embedding model, loading it at most once per process."""
    assert backend in BACKENDS, (
        f"unknown embedding backend {backend!r}; registered: {sorted(BACKENDS)}")
    key = f"{backend}:{name}"
    if key not in _CACHE:
        _CACHE[key] = BACKENDS[backend](name)
    return _CACHE[key]


def embed(texts: list[str], *, model: str = DEFAULT_MODEL, backend: str = "model2vec",
          batch_size: int = 256):
    """L2-normalised embedding matrix, one row per text, in the order given.

    Normalised here so a dot product is always a cosine -- the same contract
    `hashed_features` offers. An empty string embeds to a zero row (cosine 0 against
    everything) rather than raising, so one unresolved record cannot abort a check.
    """
    import numpy as np

    if not texts:
        return np.zeros((0, 1), dtype=np.float32)

    m = load(model, backend)
    out = []
    for start in range(0, len(texts), max(batch_size, 1)):
        out.append(np.asarray(m.encode(texts[start:start + batch_size]),
                              dtype=np.float32))
    X = np.vstack(out)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.where(norms > 0, norms, 1.0)).astype(np.float32)
