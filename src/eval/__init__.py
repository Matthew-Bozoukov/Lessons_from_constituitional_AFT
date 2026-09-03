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
    # The token this eval is called in every name it produces (src/naming.py):
    # `<date>-<key>-<model>`. REQUIRED, no default: naming an eval is a decision made
    # once, when it is registered, and never again per run. Keep it short — the key and
    # the arm's style-type share one 96-character repo name.
    key: str
    needs_docker: bool = False     # rollouts execute in containers where the driver runs
    # True when run() reaches the target purely through the OpenAI-compatible triple
    # (base_url, model_name, api_key) and so works against a public API endpoint target
    # (`<provider>:<model-id>`) as well as a vLLM-served one. Left False for evals that
    # hardcode vLLM assumptions — a served-model prefix (`vllm/`, `hosted_vllm/`), a LoRA
    # swap, a docker bridge, or a controlled chat template — which an API target can't
    # satisfy; run_eval refuses an API target for those with a clear message.
    supports_api_target: bool = False
    # True when an arm may be satisfied by a PRIOR RUN of this eval instead of a model:
    # the run's published rollouts already hold that model's generations, so the arm costs
    # no GPU. Only for evals whose generations are reusable across comparisons. A
    # BEHAVIOUR eval must always generate — reusing its responses would be reusing the
    # experiment itself — so this stays False there, and run_eval refuses such a target.
    reads_answers: bool = False
    # run() kwargs (keyword-only params — each doubles as a --flag on run_eval.py) whose
    # value names a MODEL that must also run, first, as an ordinary arm of the same
    # invocation, filling the HF answer cache the later arms are judged against. Config
    # default: `<kwarg>_model`. run_eval prepends declaratively — it never learns what the
    # kwarg means; required-ness is enforced by run() itself. NO REGISTERED EVAL USES THIS
    # TODAY: lmsys did and is gone; arena_hard is the intended next user, when its
    # `--reference` stops being an answers artifact and becomes an arm.
    arm_kwargs: tuple[str, ...] = ()
    # True when the eval's package defines pool.py::pool(runs, cfg, out_dir) -> summary.
    # run_eval calls it AFTER every arm of a multi-target invocation has been published,
    # so several arms of one recipe (seed replicates) answer the question they were run
    # for — about the recipe — instead of leaving that to a plotting script nobody runs.
    # What "pooled" means is the eval's own business: ODCV puts each arm in as a
    # checkpoint so the interval carries seed-to-seed variance.
    pools: bool = False


EVALS: dict[str, EvalSpec] = {
    "mmlu": EvalSpec(
        "capabilities.mmlu",
        "configs/eval/mmlu.yaml",
        key="mmlu",
        supports_api_target=True,
    ),
    "arena_hard": EvalSpec(
        "capabilities.arena_hard",
        "configs/eval/arena_hard.yaml",
        key="ah",
        supports_api_target=True,
        # An arm's answers are an artifact, so a prior ah run is a valid target and the
        # baseline is not a special kind of thing: --reference names a model or a prior
        # run, and run_eval runs it first as an ordinary arm.
        reads_answers=True,
        arm_kwargs=("reference",),
        # Several arms against one baseline are one comparison; pool.py is the leaderboard,
        # published as `<date>-ah-vs-<baseline>`.
        pools=True,
    ),
    # The STANDARDIZED baseline: upstream mini-SWE-agent, pinned, config untouched. A custom
    # scaffold gets its own registry entry — never fold one into the other.
    "swebench_mini": EvalSpec(
        "capabilities.swebench_mini",
        "configs/eval/swebench_mini.yaml",
        key="swebench",
        needs_docker=True,
    ),
    "internalization": EvalSpec(
        "misalignment.internalization",
        "configs/eval/internalization.yaml",
        key="internal",
    ),
    "agentic_misalignment": EvalSpec(
        "misalignment.agentic_misalignment",
        "configs/eval/agentic_misalignment.yaml",
        key="am",
    ),
    "odcv": EvalSpec(
        "misalignment.odcv",
        "configs/eval/odcv.yaml",
        key="odcv",
        needs_docker=True,
        # Seed replicates are the standard ODCV shape (`--target seed0 seed1 seed2`), and
        # the recipe-level number is what they are for: pool.py.
        pools=True,
    ),
    "psychosis": EvalSpec(
        "misalignment.psychosis",
        "configs/eval/psychosis.yaml",
        key="psychosis",
        supports_api_target=True,
    ),
    # Declarative values probe (Moral Foundations Theory), as opposed to the behavioural
    # honeypots either side of it: 88 fixed A/B items scored mechanically against a
    # released human answer key, no judge and no docker. Reaches the target purely
    # through the OpenAI triple, so an API target works and is the cheapest way to
    # smoke-test the wiring before renting anything.
    "moralbench": EvalSpec(
        "misalignment.moralbench",
        "configs/eval/moralbench.yaml",
        key="moralbench",
        supports_api_target=True,
    ),
}


def resolve(name: str) -> Callable:
    """Import and return the named eval's runner.py run() (the only place runners are imported)."""
    return importlib.import_module(f"src.eval.{EVALS[name].package}.runner").run


def resolve_pool(name: str) -> Callable:
    """Import and return the named eval's pool.py pool() — see `EvalSpec.pools`."""
    assert EVALS[name].pools, f"{name} declares no pooling step"
    return importlib.import_module(f"src.eval.{EVALS[name].package}.pool").pool
