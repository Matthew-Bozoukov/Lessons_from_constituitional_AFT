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
from .core.cache import SCOPES, CacheConfig
from .core.hashing import stable_hash
from .core.recipe import Recipe, RecipeError

CONTROL_CONFIGS = Path(__file__).resolve().parent / "control" / "configs"

# Applied under any user config. Keeps a minimal config runnable and makes every
# default explicit in one place rather than scattered through the code.
DEFAULTS: dict[str, Any] = {
    "run_id": None,
    "name": None,
    "extends": None,
    "seed": 0,
    "max_workers": 16,
    "resume": True,
    "output_dir": "output/synthdoc",
    "cache": {
        "enabled": True,
        "dir": "output/synthdoc_cache",
        "namespace": "",
        "scope": list(SCOPES),
        "max_bytes": 0,
        "embeddings": True,
        "embeddings_dir": None,
    },
    "spec": {"id": "", "path": None, "chunker": {"granularity": "bullet"}},
    "llm": {"provider": "openrouter"},
    "embedder": {"name": "hashing"},
    "pricing": {},
    "planning": {
        "enabled": False,
        "model": None,
        "template": "what_how_why",
        "temperature": 1.0,
        "max_tokens": 1200,
    },
    "generation": {
        "model": "anthropic/claude-sonnet-4.5",
        "temperature": 1.0,
        "max_tokens": 4096,
        "template": "v2",
        "strategy": "single_pass",
        "strategy_params": {},
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


def _resolve_path(path: str | Path, base_dir: Path | None = None) -> Path:
    """Resolve a config path.

    Tries, in order: the path as given, relative to the config that referenced it
    (so `extends:` works between neighbouring files), then control/configs/ by
    relative path, then by bare filename.

    Args:
        path: Config path or bare filename.
        base_dir: Directory of the referencing config, for relative `extends:`.

    Returns:
        The resolved path.

    Raises:
        ConfigError: If no candidate exists.
    """
    p = Path(path)
    candidates = [p]
    if base_dir is not None and not p.is_absolute():
        candidates.append(base_dir / p)
    if not p.is_absolute():
        candidates += [CONTROL_CONFIGS / p, CONTROL_CONFIGS / p.name]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    available = sorted(
        str(x.relative_to(CONTROL_CONFIGS)) for x in CONTROL_CONFIGS.rglob("*.yaml")
    )
    raise ConfigError(f"Config not found: {path}. In control/configs/: {available}")


def _merge(parent: dict[str, Any], child: dict[str, Any], path: str = "") -> dict[str, Any]:
    """Deep-merge a child config over a parent, with mixture-aware semantics.

    Mixtures REPLACE rather than merge. Merging `doc_type: {multiturn: 1.0}` into a
    parent that declares six document types would leave the other five behind at their
    old weights and quietly produce a corpus nobody asked for - the exact bug that makes
    a "100% multiturn" corpus silently not be one.

    Everything else deep-merges, and lists replace wholesale.

    Args:
        parent: The base config.
        child: The overriding config.
        path: Dotted path of the current node, used to spot recipe mixtures.

    Returns:
        A new merged dict.
    """
    out = dict(parent)
    for key, value in child.items():
        current = out.get(key)
        here = f"{path}.{key}" if path else key
        # recipe.<mixture> replaces; recipe.grouping_params is a real nested mapping.
        is_mixture = path == "recipe" and key != "grouping_params"
        if isinstance(value, dict) and isinstance(current, dict) and not is_mixture:
            out[key] = _merge(current, value, here)
        else:
            out[key] = value
    return out


def _load_with_extends(
    path: str | Path, seen: tuple[str, ...] = (), base_dir: Path | None = None
) -> dict[str, Any]:
    """Load a config file, resolving its `extends:` chain first.

    Args:
        path: Config path or bare filename in control/configs/.
        seen: Already-visited paths, used to detect cycles.
        base_dir: Directory of the referencing config, for relative `extends:`.

    Returns:
        The merged raw config (defaults are applied by the caller).

    Raises:
        ConfigError: If the extends chain contains a cycle.
    """
    resolved = _resolve_path(path, base_dir)
    key = str(resolved.resolve())
    if key in seen:
        raise ConfigError(
            f"extends cycle: {' -> '.join([*seen, key])}. A config may not extend itself."
        )

    raw = OmegaConf.to_container(OmegaConf.load(resolved), resolve=True)
    if not isinstance(raw, dict):
        raise ConfigError(f"Config {resolved} is not a mapping")

    _migrate_legacy_cache_keys(raw)
    parents = raw.pop("extends", None)
    if not parents:
        return raw
    if isinstance(parents, str):
        parents = [parents]

    merged: dict[str, Any] = {}
    for parent in parents:
        merged = _merge(
            merged, _load_with_extends(parent, (*seen, key), base_dir=resolved.parent)
        )
    return _merge(merged, raw)


# Flat keys kept working from before the `cache:` block existed.
LEGACY_CACHE_KEYS = {"cache_dir": "dir", "cache_enabled": "enabled"}


def _migrate_legacy_cache_keys(raw: dict[str, Any]) -> dict[str, Any]:
    """Fold flat `cache_dir` / `cache_enabled` keys into one config's `cache:` block.

    Applied to each raw config BEFORE merging, so precedence works the ordinary way:
    a `cache.dir` written in the same file (or passed as an override) wins over a
    `cache_dir` inherited from a parent. Migrating after the merge instead would let
    a parent's flat key clobber the child's explicit setting.

    Args:
        raw: One config's raw contents, modified in place.

    Returns:
        The same mapping.
    """
    for flat, field_name in LEGACY_CACHE_KEYS.items():
        if flat in raw:
            block = raw.setdefault("cache", {})
            block.setdefault(field_name, raw.pop(flat))
    return raw


def _migrate_override_keys(overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Rewrite legacy dotted override keys onto the `cache:` block."""
    out: dict[str, Any] = {}
    for key, value in (overrides or {}).items():
        out[f"cache.{LEGACY_CACHE_KEYS[key]}" if key in LEGACY_CACHE_KEYS else key] = value
    return out


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load, merge, and validate a run config.

    Supports `extends: <config>` for corpus variants: a config that differs from the
    base in three lines should be three lines long, not a hundred-line copy that drifts.

    Args:
        path: Config file path, or a bare filename inside control/configs/.
        overrides: Dotted-key overrides applied last, e.g. {"generation.model": "..."}.

    Returns:
        The resolved config as a plain dict.

    Raises:
        ConfigError: If the config is invalid.
    """
    cfg = OmegaConf.create(_merge(DEFAULTS, _load_with_extends(path)))
    # merge=False: an override REPLACES. Merging would leave stale keys in a mixture
    # dict, silently producing a recipe nobody wrote.
    for key, value in _migrate_override_keys(overrides).items():
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
    for key, value in _migrate_override_keys(overrides).items():
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

        With `name:` set, the run id IS the name - stable, so the corpus lands in a
        predictable directory and re-running resumes it rather than making a second
        copy. Without a name, the id is timestamped and every run is distinct.
    """
    if cfg.get("run_id"):
        return str(cfg["run_id"])
    if cfg.get("name"):
        return str(cfg["name"])
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
    from . import plugins  # noqa: F401  - ensures every plugin is registered

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

    planning = cfg.get("planning") or {}
    if planning.get("enabled"):
        template = planning.get("template", "what_how_why")
        if not registry.has("planner", template):
            loader.entry("planning", template)  # raises PromptError if undeclared

    strategy = (cfg.get("generation") or {}).get("strategy", "single_pass")
    if not registry.has("strategy", strategy):
        raise ConfigError(
            f"Unknown generation.strategy {strategy!r}; "
            f"registered: {registry.names('strategy')}"
        )
    if strategy == "draft_then_align" and not planning.get("enabled"):
        raise ConfigError(
            "generation.strategy: draft_then_align needs a user prompt to draft "
            "against, which comes from the planning stage. Set planning.enabled: true, "
            "or use single_pass."
        )

    provider = (cfg.get("llm") or {}).get("provider", "openrouter")
    if not registry.has("llm", provider):
        raise ConfigError(
            f"Unknown llm.provider {provider!r}; registered: {registry.names('llm')}"
        )

    try:
        CacheConfig.from_config(cfg)
    except ValueError as e:
        raise ConfigError(str(e)) from e

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
        elif kind == "pattern_scan":
            fields.update(["pattern_matches", "pattern_match_rate"])
    return sorted(fields)


def to_dict(cfg: dict[str, Any] | DictConfig) -> dict[str, Any]:
    """Coerce an OmegaConf node or dict to a plain dict."""
    if isinstance(cfg, DictConfig):
        out = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(out, dict)
        return out
    return dict(cfg)
