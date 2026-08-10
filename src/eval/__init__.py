# ABOUTME: Evaluation of trained models (capabilities/, misalignment/, audits/) and
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
    # True when run() reaches the target purely through the OpenAI-compatible triple
    # (base_url, model_name, api_key) and so works against a public API endpoint target
    # (`<provider>:<model-id>`) as well as a vLLM-served one. Left False for evals that
    # hardcode vLLM assumptions — a served-model prefix (`vllm/`, `hosted_vllm/`), a LoRA
    # swap, a docker bridge, or a controlled chat template — which an API target can't
    # satisfy; run_eval refuses an API target for those with a clear message.
    supports_api_target: bool = False
    # run() kwargs (keyword-only params — each doubles as a --flag on run_eval.py) whose
    # value names a MODEL that must also run, first, as an ordinary arm of the same
    # invocation: e.g. lmsys's `reference` fills the HF answer cache the later arms are
    # judged against. Config default: `<kwarg>_model`. run_eval prepends declaratively —
    # it never learns what the kwarg means; required-ness is enforced by run() itself.
    arm_kwargs: tuple[str, ...] = ()


EVALS: dict[str, EvalSpec] = {
    "mmlu": EvalSpec(
        "capabilities.mmlu",
        "configs/eval/mmlu.yaml",
        supports_api_target=True,
    ),
    "arena_hard": EvalSpec(
        "capabilities.arena_hard",
        "configs/eval/arena_hard.yaml",
        supports_api_target=True,
        # takes --reference (an answers ARTIFACT, asserted in run()) — not an arm_kwarg
        # until its answer-cache migration
    ),
    "lmsys": EvalSpec(
        "capabilities.lmsys",
        "configs/eval/lmsys.yaml",
        supports_api_target=True,
        arm_kwargs=("reference",),
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
        "configs/eval/psychosis.yaml",
        supports_api_target=True,
    ),
}


def resolve(name: str) -> Callable:
    """Import and return the named eval's runner.py run() (the only place runners are imported)."""
    return importlib.import_module(f"src.eval.{EVALS[name].package}.runner").run
