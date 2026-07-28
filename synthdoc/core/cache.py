# ABOUTME: Content-addressed local cache keyed on (stage_idx, input_hash, prompt_hash,
# ABOUTME: model, params). This is what makes revision dose-response sweeps affordable.

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .hashing import stable_hash


class Cache:
    """A tiny on-disk JSON cache sharded by key prefix.

    Thread-safe for the concurrent-worker access pattern used by the pipeline:
    writes go to a temp file and are atomically renamed, so a partial write can
    never be read back as a hit.
    """

    def __init__(self, root: Path | str, enabled: bool = True) -> None:
        """Initialize the cache.

        Args:
            root: Directory to store cache entries in.
            enabled: When False, every get misses and puts are dropped.
        """
        self.root = Path(root)
        self.enabled = enabled
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(
        stage_idx: int,
        input_hash: str,
        prompt_hash: str,
        model: str,
        params: dict[str, Any],
    ) -> str:
        """Build the cache key for one model call.

        Args:
            stage_idx: Stage index (so the same prompt at a different stage is distinct).
            input_hash: Hash of the stage's input document content.
            prompt_hash: Hash of the rendered prompt.
            model: Model id.
            params: Sampling parameters.

        Returns:
            Hex cache key.
        """
        return stable_hash(
            {
                "stage_idx": stage_idx,
                "input_hash": input_hash,
                "prompt_hash": prompt_hash,
                "model": model,
                "params": params,
            },
            length=32,
        )

    def _path(self, key: str) -> Path:
        """Return the shard path for a key."""
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        """Return the cached payload for a key, or None on miss."""
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            with self._lock:
                self.misses += 1
            return None
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            with self._lock:
                self.misses += 1
            return None
        with self._lock:
            self.hits += 1
        return payload

    def put(self, key: str, payload: dict[str, Any]) -> None:
        """Store a payload under a key (atomic rename; failures are non-fatal)."""
        if not self.enabled:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        try:
            tmp.write_text(json.dumps(payload))
            tmp.replace(path)
        except OSError:
            tmp.unlink(missing_ok=True)

    def stats(self) -> dict[str, int]:
        """Return hit/miss counters for the run manifest."""
        return {"hits": self.hits, "misses": self.misses}
