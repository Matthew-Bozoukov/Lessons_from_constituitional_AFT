# ABOUTME: Plugin registry. Every extension point (chunker, grouper, doc_type, axis,
# ABOUTME: reviser, filter, exporter, embedder) is resolved by (kind, name) from here.

from __future__ import annotations

from typing import Any, Callable, TypeVar

_REGISTRY: dict[str, dict[str, Any]] = {}

T = TypeVar("T")


class RegistryError(KeyError):
    """Raised when a plugin name is not registered under a kind."""


def register(kind: str, name: str) -> Callable[[T], T]:
    """Class/function decorator that registers a plugin under (kind, name).

    Args:
        kind: Extension point, e.g. "doc_type", "reviser", "grouping".
        name: Config-facing plugin name, e.g. "trait_conflict".

    Returns:
        The decorator, which returns its argument unchanged.

    Raises:
        RegistryError: If (kind, name) is already registered by a different object.
    """

    def deco(obj: T) -> T:
        bucket = _REGISTRY.setdefault(kind, {})
        existing = bucket.get(name)
        if existing is not None and existing is not obj:
            raise RegistryError(f"{kind}/{name} already registered by {existing!r}")
        bucket[name] = obj
        return obj

    return deco


def resolve(kind: str, name: str) -> Any:
    """Look up a registered plugin.

    Args:
        kind: Extension point.
        name: Plugin name.

    Returns:
        The registered object.

    Raises:
        RegistryError: If the kind or name is unknown.
    """
    bucket = _REGISTRY.get(kind)
    if bucket is None:
        raise RegistryError(f"No plugins registered under kind {kind!r}")
    if name not in bucket:
        raise RegistryError(
            f"Unknown {kind} {name!r}. Registered: {sorted(bucket)}"
        )
    return bucket[name]


def has(kind: str, name: str) -> bool:
    """Return True if (kind, name) is registered."""
    return name in _REGISTRY.get(kind, {})


def names(kind: str) -> list[str]:
    """Return the sorted registered names for a kind."""
    return sorted(_REGISTRY.get(kind, {}))


def kinds() -> list[str]:
    """Return the sorted registered kinds."""
    return sorted(_REGISTRY)
