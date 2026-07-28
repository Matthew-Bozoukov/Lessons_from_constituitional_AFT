# ABOUTME: Config loading, defaulting, and validation. Every ablation axis is a field
# ABOUTME: here; validation fails loudly before any money is spent on a malformed run.

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .control import loader
from .core import registry
from .core.hashing import stable_hash
from .core.recipe import Recipe, RecipeError

CONTROL_CONFIGS = Path(__file__).resolve().parent / "control" / "configs"

# Applied under any user config. Keeps a minimal config runnable and makes every
# default explicit in one place rather than scattered through the code.
DEFAULTS: dict[str, Any] = {
    "run_id": None,
    "seed": 0,
    "max_workers": 16,
    "resume": True,
    "output_dir": "output/synthdoc",
    "cache_dir": "output/synthdoc_cache",
    "cache_enabled": True,
    "spec": {"id": "", "path": None, "chunker": {"granularity": "bullet"}},
    "llm": {"provider": "openrouter"},
    "embedder": {"name": "hashing"},
    "pricing": {},
    "generation": {
        "model": "anthropic/claude-sonnet-4.5",
        "temperature": 1.0,
        "max_tokens": 4096,
        "template": "v2",
        "max_parse_retries": 1,
    },
    "revision": [],
    "filters": [],
    "snapshots": {
        "backend": "local",
        "org": "",
        "repo": "synthdoc-{run_id}",
        "private": True,
        "push_every_stage": True,
        "also_local": True,
        "write_jsonl": True,
    },
    "export": {"format": "sft_chat", "mix": {}},
    "report": {"enabled": True, "plot": True},
}


class ConfigError(ValueError):
    """Raised when a run config is invalid."""


def git_sha() -> str:
    """Return the current git commit SHA, or 'nogit' if unavailable."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def timestamp() -> str:
    """Return a filesystem-safe UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _resolve_path(path: str | Path) -> Path:
    """Resolve a config path, falling back to control/configs/ for bare names."""
    p = Path(path)
    if p.exists():
        return p
    candidate = CONTROL_CONFIGS / p.name
    if candidate.exists():
        return candidate
    available = sorted(x.name for x in CONTROL_CONFIGS.glob("*.yaml"))
    raise ConfigError(f"Config not found: {path}. In control/configs/: {available}")


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load, merge, and validate a run config.

    Args:
        path: Config file path, or a bare filename inside control/configs/.
        overrides: Dotted-key overrides applied last, e.g. {"generation.model": "..."}.

    Returns:
        The resolved config as a plain dict.

    Raises:
        ConfigError: If the config is invalid.
    """
    cfg = OmegaConf.merge(
        OmegaConf.create(DEFAULTS), OmegaConf.load(_resolve_path(path))
    )
    if overrides:
        for key, value in overrides.items():
            # merge=False: an override REPLACES. Merging would leave stale keys in
            # a mixture dict, silently producing a recipe nobody wrote.
            OmegaConf.update(cfg, key, value, merge=False)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)
    validate(resolved)
    return resolved


def load_config_dict(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply dotted overrides to an already-resolved config and re-validate.

    Used by the sweep runner so every arm is a config diff off one shared base,
    rather than a separately loaded file that could drift.

    Args:
        base: A resolved config.
        overrides: Dotted-key overrides, e.g. {"generation.model": "..."}.

    Returns:
        A new resolved config.

    Raises:
        ConfigError: If the resulting config is invalid.
    """
    cfg = OmegaConf.create(base)
    for key, value in (overrides or {}).items():
        OmegaConf.update(cfg, key, value, merge=False)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)
    # A run id baked into the base would collide across arms.
    resolved["run_id"] = None
    validate(resolved)
    return resolved


def make_run_id(cfg: dict[str, Any]) -> str:
    """Return the run id, generating a timestamped one when unset.

    Args:
        cfg: Resolved config.

    Returns:
        A run id. Note doc_id depends on it, so cross-arm joins use scenario_hash,
        which is run-id independent, while cross-stage joins use doc_id.
    """
    if cfg.get("run_id"):
        return str(cfg["run_id"])
    tag = cfg.get("tag") or cfg.get("spec", {}).get("id") or "run"
    return f"{tag}_{timestamp()}_{stable_hash(cfg, 6)}"


def validate(cfg: dict[str, Any]) -> None:
    """Validate a resolved config against the registries and prompt packs.

    Checks everything that would otherwise fail deep inside a paid run: unknown
    plugin names, undeclared doc types and axes, malformed mixtures, and filters
    that need an LLM when none is configured.

    Args:
        cfg: Resolved config.

    Raises:
        ConfigError: On the first problem found.
    """
    import synthdoc.plugins  # noqa: F401  - ensures every plugin is registered

    spec = cfg.get("spec") or {}
    if not spec.get("id"):
        raise ConfigError("spec.id is required")
    granularity = (spec.get("chunker") or {}).get("granularity", "bullet")
    if not registry.has("chunker", granularity):
        raise ConfigError(
            f"Unknown chunker granularity {granularity!r}; "
            f"registered: {registry.names('chunker')}"
        )

    try:
        recipe = Recipe.from_config(cfg.get("recipe") or {})
    except RecipeError as e:
        raise ConfigError(str(e)) from e

    declared_doc_types = set(loader.declared_doc_types()) | set(registry.names("doc_type"))
    for name in recipe.doc_type:
        if name not in declared_doc_types:
            raise ConfigError(
                f"doc_type {name!r} is in the recipe but is neither declared in "
                f"control/prompts/doc_types.yaml nor registered. "
                f"Available: {sorted(declared_doc_types)}"
            )

    for strategy in recipe.grouping:
        if not registry.has("grouping", strategy):
            raise ConfigError(
                f"Unknown grouping strategy {strategy!r}; "
                f"registered: {registry.names('grouping')}"
            )

    declared_axes = set(loader.declared_axes())
    for axis, mixture in recipe.axes.items():
        if axis not in declared_axes:
            raise ConfigError(
                f"Axis {axis!r} is in the recipe but not declared in "
                f"control/prompts/axes.yaml. Declared: {sorted(declared_axes)}"
            )
        for value in mixture:
            loader.axis_fragment(axis, value)  # raises PromptError if undeclared

    for i, entry in enumerate(cfg.get("revision") or []):
        kind = entry.get("kind")
        if not kind:
            raise ConfigError(f"revision[{i}] has no kind")
        if not registry.has("reviser", kind):
            loader.entry("revision", kind)  # raises PromptError if undeclared
        if entry.get("context", "fresh") not in ("fresh", "same"):
            raise ConfigError(
                f"revision[{i}].context must be 'fresh' or 'same', "
                f"got {entry.get('context')!r}"
            )

    provider = (cfg.get("llm") or {}).get("provider", "openrouter")
    if not registry.has("llm", provider):
        raise ConfigError(
            f"Unknown llm.provider {provider!r}; registered: {registry.names('llm')}"
        )

    for i, entry in enumerate(cfg.get("filters") or []):
        kind = entry.get("kind")
        if not registry.has("filter", kind):
            raise ConfigError(
                f"filters[{i}] kind {kind!r} is not registered; "
                f"registered: {registry.names('filter')}"
            )
        if kind == "autorater":
            loader.entry("rubrics", entry.get("rubric", "v4"))

    fmt = (cfg.get("export") or {}).get("format", "sft_chat")
    if not registry.has("exporter", fmt):
        raise ConfigError(
            f"Unknown export.format {fmt!r}; registered: {registry.names('exporter')}"
        )

    snapshots = cfg.get("snapshots") or {}
    if snapshots.get("backend") == "huggingface" and not snapshots.get("org"):
        raise ConfigError("snapshots.org is required when snapshots.backend is huggingface")


def filter_score_fields(cfg: dict[str, Any]) -> list[str]:
    """Return the union of score fields every configured filter will write.

    Needed up front because the snapshot schema is declared before stage 00 runs.

    Args:
        cfg: Resolved config.

    Returns:
        Sorted score field names.
    """
    fields: set[str] = set()
    for entry in cfg.get("filters") or []:
        kind = entry.get("kind")
        if kind == "autorater":
            rubric = loader.entry("rubrics", entry.get("rubric", "v4"))
            criteria = list(rubric.get("criteria") or ["overall"])
            fields.update(
                ["autorater_overall", "autorater_std"]
                + [f"autorater_{c}" for c in criteria]
            )
        elif kind == "embedding_dedup":
            fields.add("dedup_max_sim")
        elif kind == "length":
            fields.add("length_words")
    return sorted(fields)


def to_dict(cfg: dict[str, Any] | DictConfig) -> dict[str, Any]:
    """Coerce an OmegaConf node or dict to a plain dict."""
    if isinstance(cfg, DictConfig):
        out = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(out, dict)
        return out
    return dict(cfg)
