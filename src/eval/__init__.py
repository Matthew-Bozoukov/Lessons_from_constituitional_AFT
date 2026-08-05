# ABOUTME: Evaluation of trained models (capabilities/, misalignment/, vulnerabilities/) and
# ABOUTME: the eval registry: name -> EvalSpec with a lazy runner, resolved only when selected.

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable

# Runner modules are imported only when their eval is selected, so importing src.eval
# never drags in a specific eval's dependencies (torch extras, docker glue). Every eval
# package ships a runner.py defining run(target, cfg, out_dir) -> summary dict per the
# CLAUDE.md eval-framework contract; resolve() encodes that convention once.


@dataclass(frozen=True)
class EvalSpec:
    package: str                   # eval package under src.eval; its runner.py defines run()
    config: str                    # default OmegaConf YAML under configs/eval/
    needs_docker: bool = False     # rollouts execute in containers where the driver runs
    needs_reference: bool = False  # judged against a baseline arm's artifact (reference=...)


EVALS: dict[str, EvalSpec] = {
    "mmlu": EvalSpec(
        "capabilities.mmlu",
        "configs/eval/mmlu.yaml"
    ),
    "arena_hard": EvalSpec(
        "capabilities.arena_hard",
        "configs/eval/arena_hard.yaml",
        needs_reference=True
    ),
    "lmsys": EvalSpec(
        "capabilities.lmsys",
        "configs/eval/lmsys.yaml",
        needs_reference=True
    ),
    # The STANDARDIZED baseline: upstream mini-SWE-agent, pinned, config untouched. A custom
    # scaffold gets its own registry entry — never fold one into the other.
    "swebench_mini": EvalSpec(
        "capabilities.swebench_mini",
        "configs/eval/swebench_mini_verified.yaml",
        needs_docker=True
    ),
    "internalization": EvalSpec(
        "misalignment.internalization",
        "configs/eval/internalization.yaml"
    ),
    "agentic_misalignment": EvalSpec(
        "misalignment.agentic_misalignment",
        "configs/eval/agentic_misalignment.yaml"
    ),
    "odcv": EvalSpec(
        "misalignment.odcv",
        "configs/eval/odcv_bench.yaml",
        needs_docker=True
    ),
    "psychosis": EvalSpec(
        "misalignment.psychosis",
        "configs/eval/psychosis.yaml"
    ),
}


def resolve(name: str) -> Callable:
    """Import and return the named eval's runner.py run() (the only place runners are imported)."""
    return importlib.import_module(f"src.eval.{EVALS[name].package}.runner").run
