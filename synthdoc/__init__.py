# ABOUTME: Public API for the synthetic document generation pipeline.
# ABOUTME: Self-contained: nothing here imports the rest of the repo, and nothing needs to.

"""Synthetic document generation: model spec in, training corpus out.

Plug-and-play usage from anywhere in the repo:

    from synthdoc import load_config, run_pipeline

    cfg = load_config("synthdoc/control/configs/base.yaml", {"recipe.n": 200})
    result = run_pipeline(cfg)
    print(result.exports["main"])   # SFT chat JSONL, ready for training

Everything tunable lives in `synthdoc/control/`: run configs in `control/configs/`,
and every string a model sees in `control/prompts/`. Extending the pipeline means
registering a plugin and adding a config line - see `register`.
"""

from .config import ConfigError, load_config, load_config_dict, validate
from .core.registry import register, resolve
from .core.recipe import MixtureSampler, Recipe
from .core.types import Document, ScenarioSpec, SpecChunk, StageRecord, Turn
from .pipeline import BudgetExceeded, RunResult, build_scenarios, run_pipeline
from .snapshots import load_snapshot
from .sweep import SweepResult, load_sweep, run_sweep

__all__ = [
    "BudgetExceeded",
    "ConfigError",
    "Document",
    "MixtureSampler",
    "Recipe",
    "RunResult",
    "ScenarioSpec",
    "SpecChunk",
    "StageRecord",
    "SweepResult",
    "Turn",
    "build_scenarios",
    "load_config",
    "load_config_dict",
    "load_snapshot",
    "load_sweep",
    "register",
    "resolve",
    "run_pipeline",
    "run_sweep",
    "validate",
]
