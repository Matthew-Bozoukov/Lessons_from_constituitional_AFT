# ABOUTME: Canonical hashing and seeded RNG streams. These make scenario_hash and
# ABOUTME: doc_id stable across stages and across sweep arms, which everything else relies on.

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

HASH_LEN = 16


def canonical(obj: Any) -> str:
    """Serialize an object to a canonical JSON string (sorted keys, no whitespace).

    Args:
        obj: Any JSON-serializable value. Tuples become lists; sets are rejected.

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

    Deterministic across processes and Python runs (unlike builtin hash()).

    Args:
        obj: Value to hash.
        length: Number of hex characters to keep.

    Returns:
        Truncated sha256 hex digest.
    """
    return hashlib.sha256(canonical(obj).encode()).hexdigest()[:length]


def text_hash(text: str, length: int = HASH_LEN) -> str:
    """Hash a raw string (no JSON wrapping)."""
    return hashlib.sha256(text.encode()).hexdigest()[:length]


def stream_rng(seed: int, index: int, stream: str) -> random.Random:
    """Build an independent RNG for one (seed, example index, axis) triple.

    This is the mechanism behind paired sweeps: each axis draws from its own
    stream, so changing the mixture for one axis leaves every other axis's draws
    bit-identical across arms. Without it, perturbing any recipe field reshuffles
    the whole sample and comparisons stop being paired.

    Args:
        seed: Run-level seed.
        index: Example index within the run.
        stream: Axis / decision name, e.g. "doc_type" or "axis.tools".

    Returns:
        A seeded random.Random.
    """
    digest = hashlib.sha256(f"{seed}|{index}|{stream}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))
