# ABOUTME: Content-addressed on-disk cache for model and judge calls, keyed on
# ABOUTME: (scope, model, messages, params) so a re-judge costs nothing but a re-generate does.

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .hashing import stable_hash

# Cacheable call sites. `scope` selects which are cached, so a run can pay for a
# fresh generation pass while still replaying the far more expensive judge pass.
SCOPES = ("generate", "judge", "item_gen")


@dataclass
class CacheConfig:
    """Per-run cache policy.

    Attributes:
        enabled: Master switch. False makes every lookup a miss and drops writes.
        dir: Where entries live. Sharing one directory across checkpoints is the
            point - it is what makes re-judging an old checkpoint free.
        namespace: Optional key prefix, to isolate a run sharing the directory.
        scope: Which call sites to cache; a subset of SCOPES.
    """

    enabled: bool = True
    dir: str = "output/internalization_cache"
    namespace: str = ""
    scope: tuple[str, ...] = SCOPES

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> CacheConfig:
        """Build a CacheConfig from a resolved run config.

        Args:
            cfg: The resolved run config.

        Returns:
            A CacheConfig.

        Raises:
            ValueError: If `scope` names an unknown call site.
        """
        block = dict(cfg.get("cache") or {})
        scope = block.get("scope", SCOPES)
        if isinstance(scope, str):
            scope = SCOPES if scope in ("all", "*") else () if scope in ("none", "") else (scope,)
        scope = tuple(scope or ())
        unknown = sorted(set(scope) - set(SCOPES))
        if unknown:
            raise ValueError(f"cache.scope has unknown call sites {unknown}; valid: {list(SCOPES)}")
        return cls(
            enabled=bool(block.get("enabled", True)),
            dir=str(block.get("dir", "output/internalization_cache")),
            namespace=str(block.get("namespace", "")),
            scope=scope,
        )


class CallCache:
    """A sharded JSON file cache. Thread-safe; safe to share across workers."""

    def __init__(self, config: CacheConfig | None = None) -> None:
        """Initialize the cache.

        Args:
            config: Cache policy; defaults to an enabled cache in the default dir.
        """
        self.config = config or CacheConfig()
        self.root = Path(self.config.dir)
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def key(self, scope: str, model: str, messages: list[dict], params: dict[str, Any]) -> str:
        """Compute the cache key for one call.

        Args:
            scope: Call site.
            model: Model id.
            messages: Rendered prompt.
            params: Sampling params.

        Returns:
            A hex key.
        """
        return stable_hash(
            {
                "ns": self.config.namespace,
                "scope": scope,
                "model": model,
                "messages": messages,
                "params": params,
            }
        )

    def _path(self, key: str) -> Path:
        """Return the on-disk path for a key, sharded by its first two chars."""
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str, scope: str) -> dict[str, Any] | None:
        """Look up an entry.

        Args:
            key: Cache key.
            scope: Call site; a scope outside the policy always misses.

        Returns:
            The cached payload, or None on a miss.
        """
        if not self.config.enabled or scope not in self.config.scope:
            return None
        path = self._path(key)
        if not path.exists():
            with self._lock:
                self.misses += 1
            return None
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            # A half-written entry is a miss, not a crash: the call just re-runs.
            with self._lock:
                self.misses += 1
            return None
        with self._lock:
            self.hits += 1
        return payload

    def put(self, key: str, payload: dict[str, Any], scope: str) -> None:
        """Store an entry, writing atomically so a crash cannot leave a torn file.

        Args:
            key: Cache key.
            payload: JSON-serializable value.
            scope: Call site; a scope outside the policy is dropped.
        """
        if not self.config.enabled or scope not in self.config.scope:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(path)

    def stats(self) -> dict[str, int]:
        """Return hit/miss counters for the manifest."""
        with self._lock:
            return {"hits": self.hits, "misses": self.misses}
