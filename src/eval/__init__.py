# ABOUTME: Evaluation of trained models (capabilities/, misalignment/, vulnerabilities/) and
# ABOUTME: the eval registry: name -> EvalSpec with a lazy runner, resolved only when selected.

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable

# Runners are "module:function" strings, imported only when the eval is selected, so
# importing src.eval never drags in a specific eval's dependencies (torch extras, docker
# glue). Every runner implements run(target, cfg, out_dir) -> summary dict per the
# CLAUDE.md eval-framework contract.


@dataclass(frozen=True)
class EvalSpec:
    runner: str                    # "module:function", resolved lazily
    config: str                    # default OmegaConf YAML under configs/eval/
    needs_docker: bool = False     # rollouts execute in containers on the pod
    needs_reference: bool = False  # judged against a baseline arm's artifact (reference=...)


EVALS: dict[str, EvalSpec] = {
    "mmlu": EvalSpec(
        runner="src.eval.capabilities.mmlu_eval:run",
        config="configs/eval/mmlu.yaml"),
    "capability": EvalSpec(
        runner="src.eval.capabilities.capability_eval:run",
        config="configs/eval/capability.yaml",
        needs_reference=True),
    "lmsys": EvalSpec(
        runner="src.eval.capabilities.lmsys_eval:run",
        config="configs/eval/lmsys.yaml",
        needs_reference=True),
    "internalization": EvalSpec(
        runner="src.eval.misalignment.internalization.runner:run",
        config="configs/eval/internalization.yaml"),
    "agentic_misalignment": EvalSpec(
        runner="src.eval.misalignment.agentic_misalignment:run",
        config="configs/eval/agentic_misalignment.yaml"),
    "odcv": EvalSpec(
        runner="src.eval.misalignment.odcv_bench:run",
        config="configs/eval/odcv_bench.yaml",
        needs_docker=True),
    "psychosis": EvalSpec(
        runner="src.eval.misalignment.psychosis.runner:run",
        config="configs/eval/psychosis.yaml"),
}


def resolve(name: str) -> Callable:
    """Import and return the named eval's run() (the only place runners are imported)."""
    spec = EVALS[name]
    module, func = spec.runner.split(":")
    return getattr(importlib.import_module(module), func)
