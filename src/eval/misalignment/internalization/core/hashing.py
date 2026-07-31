# ABOUTME: Canonical hashing and seeded RNG streams. These make item_id and itemset_id
# ABOUTME: stable across runs and recipes, which every pairwise comparison relies on.

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

HASH_LEN = 16


def canonical(obj: Any) -> str:
    """Serialize to canonical JSON (sorted keys, no whitespace).

    Args:
        obj: Any JSON-serializable value. Tuples become lists; sets are rejected
            because their iteration order would make the hash non-deterministic.

    Returns:
        A deterministic JSON string.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_default)


def _default(obj: Any) -> Any:
    """Fallback serializer for objects exposing to_dict/_asdict, else repr."""
    for attr in ("to_dict", "_asdict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            return fn()
    if isinstance(obj, (set, frozenset)):
        raise TypeError("sets are unordered and must not appear in hashed payloads")
    return repr(obj)


def stable_hash(obj: Any, length: int = HASH_LEN) -> str:
    """Hash any JSON-serializable object to a short hex digest.

    Deterministic across processes and interpreter runs, unlike builtin hash().

    Args:
        obj: Value to hash.
        length: Number of hex characters to keep.

    Returns:
        Truncated sha256 hex digest.
    """
    return hashlib.sha256(canonical(obj).encode()).hexdigest()[:length]


def text_hash(text: str, length: int = HASH_LEN) -> str:
    """Hash a raw string with no JSON wrapping."""
    return hashlib.sha256(text.encode()).hexdigest()[:length]


def stream_rng(seed: int, index: int, stream: str) -> random.Random:
    """Build an independent RNG for one (seed, index, stream) triple.

    This is what keeps item generation paired across item-set versions: each
    family and axis draws from its own stream, so adding a clause or bumping a
    count for one family leaves every other family's draws bit-identical. Without
    it, any edit reshuffles the whole set and old results stop being comparable.

    Args:
        seed: Run-level seed.
        index: Item index within the stream.
        stream: Stream name, e.g. "family.application" or "transform.pressure".

    Returns:
        A seeded random.Random.
    """
    digest = hashlib.sha256(f"{seed}|{index}|{stream}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))
