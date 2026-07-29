# ABOUTME: Config loading, defaulting, and validation. Every knob in the suite is a field
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
from .core.types import DIFFICULTIES, FAMILIES

CONFIGS = Path(__file__).resolve().parent / "control" / "configs"

# Applied under any user config. Keeps a minimal config runnable and makes every default
# explicit in one place rather than scattered through the code.
DEFAULTS: dict[str, Any] = {
    "run_id": None,
    "name": None,
    "extends": None,
    "seed": 0,
    "max_workers": 16,
    "output_dir": "output/constieval",
    "cache": {
        "enabled": True,
        "dir": "output/constieval_cache",
        "namespace": "",
        "scope": list(SCOPES),
    },
    "clause_set": "approved_constitution_v1",
    "itemset": {
        "id": None,
        "dir": "output/constieval/itemsets",
        # Sampled per item so scenarios for one clause do not all share a setting.
        "domains": [
            "a mid-size software company",
            "a hospital administration office",
            "a university research group",
            "a regional bank",
            "a local government department",
            "a manufacturing supplier",
            "a nonprofit delivering services",
            "a law firm",
            "a logistics operator",
            "a school district",
        ],
        # Held-out clauses are excluded from TRAINING data generation, never from evaluation -
        # evaluating them is the entire point of holding them out.
        "include_held_out": True,
        "families": {
            "application": {"enabled": True, "difficulties": list(DIFFICULTIES[:3]), "variants": 2},
            "retrieval": {"enabled": True, "variants": 1},
            "fake_clause": {"enabled": True, "per_clause": 1},
            "conflict": {"enabled": True, "pairs": 12, "variants": 1},
            "over_refusal": {"enabled": True, "variants": 2},
            "persona_drift": {"enabled": True, "n": 6},
        },
        "transforms": {
            "pressure": {
                "enabled": True,
                "wrappers": [
                    "system_override",
                    "authority_claim",
                    "persona_pressure",
                    "sunk_rapport",
                    "compelling_argument",
                ],
                "per_clause": 1,
            },
            "ood": {
                "enabled": True,
                "axes": ["domain", "language", "format", "framing"],
                "per_clause": 1,
                "max_distance": 0,
            },
        },
        "generator": {
            "provider": "openrouter",
            "model": "google/gemini-2.5-flash",
            "temperature": 1.0,
            "max_tokens": 1200,
        },
    },
    # The checkpoint under test.
    "target": {
        "provider": "vllm",
        "model": "qwen3",
        "base_url": "http://localhost:8000/v1",
        "temperature": 0.7,
        "max_tokens": 2048,
        "system": "",
        "enable_thinking": True,
        "extra_body": {},
    },
    "judge": {
        "provider": "openrouter",
        "model": "google/gemini-2.5-flash",
        "temperature": 0.0,
        "max_tokens": 900,
        "max_parse_retries": 2,
    },
    # A run whose completions were cut off mid-answer is measuring truncation, not the model.
    # Above this fraction the run is flagged loudly rather than quietly reported.
    "run": {"recipe": "baseline", "checkpoint_step": 0, "max_truncation_rate": 0.15},
    # Capability regression is ingested from whatever harness already runs it (the repo's
    # MMLU / Inspect drivers), never reimplemented here - a second implementation would
    # drift from the numbers everyone else quotes.
    "side_effects": {"capability_path": None, "require_capability": False},
    "report": {"enabled": True, "plots": True},
    "pricing": {},
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
    """Resolve a config path, trying the literal path then control/configs/.

    Args:
        path: Config path or bare filename.
        base_dir: Directory of the referencing config, so relative `extends:` works.

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
        candidates += [CONFIGS / p, CONFIGS / p.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    available = sorted(str(x.relative_to(CONFIGS)) for x in CONFIGS.rglob("*.yaml"))
    raise ConfigError(f"Config not found: {path}. In control/configs/: {available}")


def _merge(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge a child config over a parent; lists replace wholesale.

    Args:
        parent: The base config.
        child: The overriding config.

    Returns:
        A new merged dict.
    """
    out = dict(parent)
    for key, value in child.items():
        current = out.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            out[key] = _merge(current, value)
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
        base_dir: Directory of the referencing config.

    Returns:
        The merged raw config; defaults are applied by the caller.

    Raises:
        ConfigError: If the extends chain contains a cycle or a file is not a mapping.
    """
    resolved = _resolve_path(path, base_dir)
    key = str(resolved.resolve())
    if key in seen:
        raise ConfigError(f"extends cycle: {' -> '.join([*seen, key])}")

    raw = OmegaConf.to_container(OmegaConf.load(resolved), resolve=True)
    if not isinstance(raw, dict):
        raise ConfigError(f"Config {resolved} is not a mapping")

    parents = raw.pop("extends", None)
    if not parents:
        return raw
    if isinstance(parents, str):
        parents = [parents]
    merged: dict[str, Any] = {}
    for parent in parents:
        merged = _merge(merged, _load_with_extends(parent, (*seen, key), base_dir=resolved.parent))
    return _merge(merged, raw)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load, merge, and validate a run config.

    Args:
        path: Config file path, or a bare filename inside control/configs/.
        overrides: Dotted-key overrides applied last, e.g. {"target.model": "..."}.

    Returns:
        The resolved config as a plain dict.

    Raises:
        ConfigError: If the config is invalid.
    """
    cfg = OmegaConf.create(_merge(DEFAULTS, _load_with_extends(path)))
    # merge=False: an override REPLACES. Merging would leave stale keys behind in a
    # mapping and silently produce a run nobody configured.
    for key, value in (overrides or {}).items():
        OmegaConf.update(cfg, key, value, merge=False)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)
    validate(resolved)
    return resolved


def load_config_dict(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply dotted overrides to an already-resolved config and re-validate.

    Used when sweeping checkpoints, so every run is a diff off one shared base rather
    than a separately loaded file that could drift.

    Args:
        base: A resolved config.
        overrides: Dotted-key overrides.

    Returns:
        A new resolved config.

    Raises:
        ConfigError: If the resulting config is invalid.
    """
    cfg = OmegaConf.create(base)
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, merge=False)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)
    resolved["run_id"] = None  # a run id baked into the base would collide across arms
    validate(resolved)
    return resolved


def make_run_id(cfg: dict[str, Any]) -> str:
    """Return the run id, generating a timestamped one when unset.

    Args:
        cfg: Resolved config.

    Returns:
        A run id. With `name:` set the id IS the name - stable, so re-running lands in
        the same directory rather than making a second copy.
    """
    if cfg.get("run_id"):
        return str(cfg["run_id"])
    if cfg.get("name"):
        return str(cfg["name"])
    recipe = (cfg.get("run") or {}).get("recipe", "run")
    step = (cfg.get("run") or {}).get("checkpoint_step", 0)
    return f"{recipe}_step{step}_{timestamp()}_{stable_hash(cfg, 6)}"


def validate(cfg: dict[str, Any]) -> None:
    """Validate a resolved config against the registries and control plane.

    Checks everything that would otherwise fail deep inside a paid run: unknown
    providers, undeclared families, wrappers, OOD axes, and clause sets.

    Args:
        cfg: Resolved config.

    Raises:
        ConfigError: On the first problem found.
    """
    import constieval.items  # noqa: F401  - registers builders and transforms
    import constieval.judges  # noqa: F401  - registers judges

    try:
        clauses = loader.clause_set(str(cfg.get("clause_set")))
    except loader.PromptError as e:
        raise ConfigError(
            f"{e} Available clause sets: {loader.available_clause_sets()}"
        ) from e
    itemset = cfg.get("itemset") or {}
    conflict_enabled = ((itemset.get("families") or {}).get("conflict") or {}).get("enabled", True)
    if conflict_enabled and not clauses.priority_order:
        raise ConfigError(
            f"Clause set {clauses.spec_id!r} declares no priority_order, so the conflict "
            f"judge has nothing to grade a resolution against. Either add one to the clause "
            f"set, or set itemset.families.conflict.enabled: false — a source document that "
            f"states no ordering cannot support the conflict axis."
        )

    if not itemset.get("domains"):
        raise ConfigError("itemset.domains must list at least one domain")

    families = itemset.get("families") or {}
    unknown = sorted(set(families) - set(FAMILIES))
    if unknown:
        raise ConfigError(f"itemset.families has unknown families {unknown}; valid: {list(FAMILIES)}")
    for family, spec in families.items():
        if not spec.get("enabled", True):
            continue
        if not registry.has("builder", family):
            raise ConfigError(
                f"Family {family!r} is enabled but no builder is registered; "
                f"registered: {registry.names('builder')}"
            )
    app = families.get("application") or {}
    bad_difficulty = sorted(set(app.get("difficulties") or ()) - set(DIFFICULTIES))
    if bad_difficulty:
        raise ConfigError(
            f"itemset.families.application.difficulties has unknown values {bad_difficulty}; "
            f"valid: {list(DIFFICULTIES)}"
        )
    if (families.get("retrieval") or {}).get("enabled", True) and not app.get("enabled", True):
        raise ConfigError(
            "itemset.families.retrieval is enabled but application is not. Retrieval items are "
            "derived from application scenarios so that the retrieval-vs-application scatter is "
            "a within-scenario comparison."
        )

    transforms = itemset.get("transforms") or {}
    pressure = transforms.get("pressure") or {}
    if pressure.get("enabled", True):
        declared = loader.declared_wrappers()
        missing = sorted(set(pressure.get("wrappers") or ()) - set(declared))
        if missing:
            raise ConfigError(
                f"itemset.transforms.pressure.wrappers names undeclared wrappers {missing}; "
                f"declared in control/prompts/pressure.yaml: {declared}"
            )
    ood = transforms.get("ood") or {}
    if ood.get("enabled", True):
        declared = loader.declared_ood_axes()
        missing = sorted(set(ood.get("axes") or ()) - set(declared))
        if missing:
            raise ConfigError(
                f"itemset.transforms.ood.axes names undeclared axes {missing}; "
                f"declared in control/prompts/ood.yaml: {declared}"
            )
        for axis in ood.get("axes") or ():
            loader.ood_axis(axis)  # raises PromptError on a malformed distance ordering

    for section in ("target", "judge"):
        provider = (cfg.get(section) or {}).get("provider")
        if not registry.has("llm", provider):
            raise ConfigError(
                f"{section}.provider {provider!r} is not registered; "
                f"registered: {registry.names('llm')}"
            )
    gen_provider = ((itemset.get("generator")) or {}).get("provider")
    if not registry.has("llm", gen_provider):
        raise ConfigError(
            f"itemset.generator.provider {gen_provider!r} is not registered; "
            f"registered: {registry.names('llm')}"
        )

    # Every declared rubric axis must have a judge, or a family would be silently unscored.
    for axis in loader.declared_axes():
        if not registry.has("judge", axis):
            raise ConfigError(
                f"Rubric axis {axis!r} is declared in control/prompts/rubrics.yaml but no judge "
                f"is registered for it; registered: {registry.names('judge')}"
            )

    try:
        CacheConfig.from_config(cfg)
    except ValueError as e:
        raise ConfigError(str(e)) from e


def to_dict(cfg: dict[str, Any] | DictConfig) -> dict[str, Any]:
    """Coerce an OmegaConf node or dict to a plain dict."""
    if isinstance(cfg, DictConfig):
        out = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(out, dict)
        return out
    return dict(cfg)
