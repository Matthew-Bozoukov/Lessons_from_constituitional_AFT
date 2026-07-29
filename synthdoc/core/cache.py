# ABOUTME: Content-addressed local cache keyed on (stage_idx, input_hash, prompt_hash,
# ABOUTME: model, params), with per-run control over what is cached, where, and how much.

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .hashing import stable_hash

# Cacheable call sites. `scope` selects which of these are cached, so a run can pay
# for fresh generations while still replaying expensive rating calls, or vice versa.
SCOPES = ("plan", "generate", "revise", "filter")


@dataclass
class CacheConfig:
    """Per-run cache policy: what is cached, where it goes, and how much is kept.

    Attributes:
        enabled: Master switch. False makes every lookup a miss and drops every write.
        dir: Where cache entries live. Sharing one directory across runs is the point -
            that is what makes a sweep's later arms cheap.
        namespace: Optional key prefix. Isolates a run from entries written by other
            runs sharing the same directory, e.g. to force a clean re-generation
            without deleting anyone else's cache.
        scope: Which call sites to cache. A subset of SCOPES.
        max_bytes: Soft ceiling on cache size; 0 means unlimited. Oldest entries are
            pruned first, on startup and periodically during a run.
        embeddings: Cache the per-spec embedding index (built once per spec, reused
            by every run and sweep arm).
        embeddings_dir: Override for the embedding index location. Defaults to
            <dir>/embeddings, so pointing `dir` at fast local disk moves both.
    """

    enabled: bool = True
    dir: str = "output/synthdoc_cache"
    namespace: str = ""
    scope: tuple[str, ...] = SCOPES
    max_bytes: int = 0
    embeddings: bool = True
    embeddings_dir: str | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> CacheConfig:
        """Build a CacheConfig from a run config.

        Accepts the `cache:` block, and falls back to the flat `cache_dir` /
        `cache_enabled` keys so older configs keep working.

        Args:
            cfg: The resolved run config.

        Returns:
            A CacheConfig.

        Raises:
            ValueError: If `scope` names an unknown call site.
        """
        block = dict(cfg.get("cache") or {})
        enabled = block.get("enabled", cfg.get("cache_enabled", True))
        directory = block.get("dir", cfg.get("cache_dir", "output/synthdoc_cache"))

        scope = block.get("scope", SCOPES)
        if isinstance(scope, str):
            if scope in ("all", "*"):
                scope = SCOPES
            elif scope in ("none", ""):
                scope = ()
            else:
                scope = (scope,)
        scope = tuple(scope or ())
        unknown = set(scope) - set(SCOPES)
        if unknown:
            raise ValueError(
                f"cache.scope has unknown call sites {sorted(unknown)}; valid: {list(SCOPES)}"
            )

        return cls(
            enabled=bool(enabled),
            dir=str(directory),
            namespace=str(block.get("namespace", "") or ""),
            scope=scope,
            max_bytes=int(block.get("max_bytes", 0) or 0),
            embeddings=bool(block.get("embeddings", True)),
            embeddings_dir=block.get("embeddings_dir"),
        )

    def embeddings_path(self) -> str | None:
        """Return the embedding index directory, or None when disabled."""
        if not self.enabled or not self.embeddings:
            return None
        return self.embeddings_dir or str(Path(self.dir) / "embeddings")


class Cache:
    """A tiny on-disk JSON cache sharded by key prefix.

    Thread-safe for the concurrent-worker access pattern used by the pipeline:
    writes go to a temp file and are atomically renamed, so a partial write can
    never be read back as a hit.
    """

    _PRUNE_EVERY = 256

    def __init__(self, cfg: CacheConfig | Path | str, enabled: bool | None = None) -> None:
        """Initialize the cache.

        Args:
            cfg: A CacheConfig, or a directory path for the simple case.
            enabled: Legacy override when `cfg` is a bare path.
        """
        if not isinstance(cfg, CacheConfig):
            cfg = CacheConfig(dir=str(cfg), enabled=True if enabled is None else bool(enabled))
        self.cfg = cfg
        self.root = Path(cfg.dir)
        self.enabled = cfg.enabled
        self._lock = threading.Lock()
        self._writes = 0
        self.hits = 0
        self.misses = 0
        self.bypassed = 0
        self.evicted = 0
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)
            self.prune()

    def caches(self, scope: str) -> bool:
        """Return True if this call site is cached under the current policy."""
        return self.enabled and scope in self.cfg.scope

    def key(
        self,
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
            Hex cache key, namespaced when the run sets cache.namespace.
        """
        return stable_hash(
            {
                "namespace": self.cfg.namespace,
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

    def get(self, key: str, scope: str = "generate") -> dict[str, Any] | None:
        """Return the cached payload for a key, or None on miss or bypass."""
        if not self.caches(scope):
            with self._lock:
                self.bypassed += 1
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

    def put(self, key: str, payload: dict[str, Any], scope: str = "generate") -> None:
        """Store a payload under a key (atomic rename; failures are non-fatal)."""
        if not self.caches(scope):
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        try:
            tmp.write_text(json.dumps(payload))
            tmp.replace(path)
        except OSError:
            tmp.unlink(missing_ok=True)
            return
        with self._lock:
            self._writes += 1
            due = self.cfg.max_bytes and self._writes % self._PRUNE_EVERY == 0
        if due:
            self.prune()

    def entries(self) -> Iterable[Path]:
        """Yield every cache entry file."""
        return self.root.glob("*/*.json")

    def size_bytes(self) -> int:
        """Return the total size of the cache on disk."""
        if not self.root.exists():
            return 0
        return sum(p.stat().st_size for p in self.entries())

    def prune(self) -> int:
        """Evict oldest entries until the cache fits under max_bytes.

        Returns:
            Number of entries removed. Always 0 when max_bytes is 0 (unlimited).
        """
        if not self.enabled or not self.cfg.max_bytes or not self.root.exists():
            return 0
        files = []
        total = 0
        for path in self.entries():
            try:
                stat = path.stat()
            except OSError:
                continue
            files.append((stat.st_mtime, stat.st_size, path))
            total += stat.st_size
        if total <= self.cfg.max_bytes:
            return 0

        files.sort()  # oldest first
        removed = 0
        for _, size, path in files:
            if total <= self.cfg.max_bytes:
                break
            try:
                path.unlink()
            except OSError:
                continue
            total -= size
            removed += 1
        with self._lock:
            self.evicted += removed
        return removed

    def stats(self) -> dict[str, Any]:
        """Return counters and policy for the run manifest."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "bypassed": self.bypassed,
            "evicted": self.evicted,
            "enabled": self.enabled,
            "dir": str(self.root),
            "scope": list(self.cfg.scope),
            "namespace": self.cfg.namespace,
            "max_bytes": self.cfg.max_bytes,
            "size_bytes": self.size_bytes() if self.enabled else 0,
        }
