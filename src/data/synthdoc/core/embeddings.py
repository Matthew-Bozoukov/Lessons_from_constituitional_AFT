# ABOUTME: Embedder plugins plus a disk-cached index. Used by semantic grouping
# ABOUTME: (once per spec_id, not per run) and by embedding_dedup (per corpus).

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .hashing import stable_hash
from .registry import register, resolve

_TOKEN = re.compile(r"[a-z0-9']+")


@runtime_checkable
class Embedder(Protocol):
    """Maps texts to L2-normalized row vectors."""

    name: str

    def embed(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - protocol
        """Return a (len(texts), dim) float32 array of unit vectors."""
        ...


def _l2(mat: np.ndarray) -> np.ndarray:
    """L2-normalize rows, leaving all-zero rows at zero."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


@register("embedder", "hashing")
class HashingEmbedder:
    """Deterministic offline embedder: hashed word 1- and 2-grams, sublinear tf.

    Not semantic in the learned sense, but it is exact, free, reproducible, and
    strong at the job dedup actually needs - detecting near-duplicate surface form.
    It is the default so that no part of the pipeline requires a second API key.
    """

    name = "hashing"

    def __init__(self, dim: int = 1024, ngrams: int = 2, **_: Any) -> None:
        """Initialize.

        Args:
            dim: Output dimensionality.
            ngrams: Max word n-gram length to hash.
        """
        self.dim = int(dim)
        self.ngrams = int(ngrams)

    def _bucket(self, token: str) -> int:
        """Hash a token to a dimension index."""
        return int(hashlib.blake2b(token.encode(), digest_size=8).hexdigest(), 16) % self.dim

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts."""
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            words = _TOKEN.findall(text.lower())
            for n in range(1, self.ngrams + 1):
                for i in range(len(words) - n + 1):
                    out[row, self._bucket(" ".join(words[i : i + n]))] += 1.0
        np.log1p(out, out=out)  # sublinear tf: long documents stop dominating
        return _l2(out)


@register("embedder", "openai")
class OpenAIEmbedder:
    """Embeddings via any OpenAI-compatible /embeddings endpoint."""

    name = "openai"

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        batch_size: int = 256,
        **_: Any,
    ) -> None:
        """Initialize.

        Args:
            model: Embedding model id.
            base_url: Optional API base URL override.
            api_key_env: Environment variable holding the key.
            batch_size: Texts per request.

        Raises:
            RuntimeError: If the API key env var is unset.
        """
        import os

        from dotenv import load_dotenv

        load_dotenv()
        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(
                f"{api_key_env} is not set, which the 'openai' embedder requires. "
                "Set embedder.name=hashing to run without it."
            )
        from openai import OpenAI

        self._client = OpenAI(api_key=key, base_url=base_url)
        self.model = model
        self.batch_size = int(batch_size)
        self.name = f"openai:{model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts, chunking into API-sized requests."""
        vecs: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t[:8000] or " " for t in texts[i : i + self.batch_size]]
            resp = self._client.embeddings.create(model=self.model, input=batch)
            vecs.extend(d.embedding for d in resp.data)
        return _l2(np.asarray(vecs, dtype=np.float32))


def build_embedder(cfg: dict[str, Any] | None) -> Embedder:
    """Instantiate the embedder named by an `embedder:` config block.

    Args:
        cfg: Mapping with a "name" key plus embedder kwargs. None means hashing.

    Returns:
        An Embedder.
    """
    params = dict(cfg or {})
    name = params.pop("name", "hashing")
    return resolve("embedder", name)(**params)


class EmbeddingIndex:
    """Cached embeddings for a fixed list of texts.

    Semantic grouping needs one index per spec_id; it is built once and reused by
    every run and every sweep arm, so it is not a per-run cost.
    """

    def __init__(self, ids: list[str], vectors: np.ndarray) -> None:
        """Initialize with parallel id and vector arrays."""
        self.ids = ids
        self.vectors = vectors
        self.pos = {cid: i for i, cid in enumerate(ids)}

    @classmethod
    def build(
        cls,
        ids: list[str],
        texts: list[str],
        embedder: Embedder,
        cache_dir: Path | str | None = None,
        tag: str = "",
    ) -> EmbeddingIndex:
        """Build or load a cached index.

        Args:
            ids: Stable ids parallel to texts.
            texts: Texts to embed.
            embedder: The embedder to use on a cache miss.
            cache_dir: Where to persist the index. None disables persistence.
            tag: Extra cache-key component, e.g. the spec_id.

        Returns:
            An EmbeddingIndex.
        """
        path = None
        if cache_dir:
            key = stable_hash(
                {"tag": tag, "embedder": embedder.name, "ids": ids, "texts": texts},
                length=24,
            )
            path = Path(cache_dir) / f"emb_{tag or 'index'}_{key}.npz"
            if path.exists():
                data = np.load(path, allow_pickle=True)
                return cls(list(data["ids"]), data["vectors"])

        vectors = embedder.embed(texts)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, ids=np.array(ids, dtype=object), vectors=vectors)
        return cls(ids, vectors)

    def similarity(self, a: str, b: str) -> float:
        """Cosine similarity between two indexed ids."""
        return float(self.vectors[self.pos[a]] @ self.vectors[self.pos[b]])

    def neighbours(self, item_id: str) -> np.ndarray:
        """Return cosine similarity of one id against every indexed item."""
        return self.vectors @ self.vectors[self.pos[item_id]]
