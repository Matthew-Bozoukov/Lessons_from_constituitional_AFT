# ABOUTME: run() for whistlebench_team per the eval contract: build one Inspect task per cell, point
# ABOUTME: Inspect at the served target (and the peer's model), run, then export into the layout.

"""The registry's run() for whistlebench_team.

Inspect drives the agent loop, the tools and the scorers, as upstream wrote it; this
function only decides WHAT runs and WHERE the model is. The served target reaches Inspect
through its OpenAI-compatible provider (`openai-api/vllm/<served name>`, base URL and key
from the environment run_eval prepared), or as `openrouter/<id>` for an API target. The
peer seat's model is the served BASE model on the same server by default (constant across
arms, no extra weights), or a script. Every grader is one OpenRouter model, pinned to its
provider the way every judge in this repo is.

One Inspect task per cell (storyline × docs × team), `seeds` epochs each; Inspect's logs
stay under metadata/inspect/ and export.py turns them into the published layout.
"""

from __future__ import annotations

import os
from pathlib import Path

from omegaconf import OmegaConf

from src.eval.layout import publish_layout

__all__ = ["cells", "run"]

SMOKE_CELLS = [
    ("pharma", "misconduct", "solo"),
    ("pharma", "misconduct", "ai_conceal"),
    ("pharma", "benign", "solo"),
    ("pharma", "benign", "ai_conceal"),
]


def cells(cfg) -> list[tuple[str, str, str]]:
    """(storyline, docs, team) triples the config asks for; `smoke` shrinks them to four."""
    if bool(cfg.get("smoke", False)):
        return list(SMOKE_CELLS)
    return [
        (str(s), str(d), str(t))
        for s in cfg.storylines
        for d in cfg.docs
        for t in cfg.teams
    ]


def run(target, cfg, out_dir: Path) -> dict:
    """Run whistlebench_team against one target (CLAUDE.md contract).

    Args:
        target: The ServedTarget from run_eval (base_url, model_name, api_key, spec).
        cfg: configs/eval/whistlebench_team.yaml plus CLI overrides.
        out_dir: This target's run directory; rollouts/, results/, metadata/ go under it.

    Returns:
        The summary: per-cell rates, the ladder distribution, the hit/false-alarm
        calibration per team, and run health.
    """
    cfg = OmegaConf.merge(cfg)
    rollout_dir, results_dir, metadata_dir = publish_layout(out_dir)

    # sys.path for the vendored package, then Inspect and the task builder
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import GenerateConfig, get_model

    from src.eval.misalignment.whistlebench_team import _upstream as U
    from src.eval.misalignment.whistlebench_team.export import export
    from src.eval.misalignment.whistlebench_team.task import TeamParams, build_task
    from src.infra.endpoints.openrouter import provider_pin

    gen = cfg.generation
    gen_config = GenerateConfig(
        temperature=float(gen.temperature),
        top_p=float(gen.top_p),
        max_tokens=int(gen.max_tokens),
    )

    # ── the target, as Inspect names it ──
    base_url = target.base_url  # touching it is what serves a lazy target
    model_args: dict = {}
    if target.is_api:
        provider, model_id = str(target.spec.hf_path).split(":", 1)
        assert provider == "openrouter", (
            f"{target.spec.hf_path}: only openrouter API targets are wired for Inspect here "
            "(Inspect's openrouter provider + the repo's provider pins)"
        )
        model_str = f"openrouter/{model_id}"
        try:
            model_args = {"provider": provider_pin(model_id)}
        except (
            Exception
        ) as e:  # an unpinned smoke model: Inspect routes it itself, said out loud
            print(
                f"!!! whistlebench_team: no provider pin for {model_id} ({e}); Inspect will route it"
            )
        peer_default = model_str
    else:
        os.environ["VLLM_BASE_URL"] = base_url
        os.environ["VLLM_API_KEY"] = target.api_key or "EMPTY"
        model_str = f"openai-api/vllm/{target.model_name}"
        peer_default = (
            "openai-api/vllm/base"  # the served base model, same server, no adapter
        )

    # ── the peer's model ──
    peer_mode = str(cfg.peer.model)
    scripted = peer_mode == "scripted"
    peer_model = None
    if not scripted and any(t != "solo" for _, _, t in cells(cfg)):
        if peer_mode == "base":
            peer_model = get_model(peer_default, config=gen_config, **model_args)
        else:
            sibling = target.sibling(
                peer_mode
            )  # co-served adapter: same base, same mode
            peer_model = get_model(
                f"openai-api/vllm/{sibling.model_name}", config=gen_config
            )

    # ── the judge ──
    judge_config = GenerateConfig(
        temperature=float(cfg.judge.temperature), max_tokens=int(cfg.judge.max_tokens)
    )
    grader = get_model(
        f"openrouter/{cfg.judge.model}",
        config=judge_config,
        provider=provider_pin(str(cfg.judge.model)),
    )
    U.upstream_scorers.GRADER_CONFIG = (
        judge_config  # PATCH hook: upstream's graders use this config
    )

    # ── the tasks ──
    seeds = 1 if bool(cfg.get("smoke", False)) else int(cfg.seeds)
    tasks = []
    for storyline, docs, team in cells(cfg):
        params = TeamParams(
            storyline=storyline,
            docs=docs,
            team=team,
            agent_responsibility=str(cfg.responsibility),
            boldness_prompt_detail=str(cfg.boldness),
            include_environment_and_workflow_details=bool(cfg.workflow_details),
            include_distractor_docs=bool(cfg.distractor_docs),
            include_distractor_tools=bool(cfg.distractor_tools),
            include_evaluation_deliberation=bool(cfg.get("eval_deliberation", False)),
            peer_max_calls=int(cfg.peer.max_calls),
        )
        tasks.append(
            build_task(
                params,
                grader=grader,
                grader_config=judge_config,
                peer_model=peer_model,
                scripted_peer=scripted,
                max_tool_output=int(cfg.inspect.max_tool_output),
                judge_max_chars=int(cfg.judge.max_chars),
            )
        )
    print(
        f">>> whistlebench_team: {len(tasks)} cells x {seeds} seeds | target {model_str} "
        f"(mode {target.spec.mode}) | peer {'scripted' if scripted else (peer_default if peer_mode == 'base' else peer_mode)} "
        f"| judge {cfg.judge.model}"
    )

    inspect_log_dir = metadata_dir / "inspect"
    logs = inspect_eval(
        tasks,
        model=model_str,
        model_args=model_args,
        epochs=seeds,
        log_dir=str(inspect_log_dir),
        log_format=str(cfg.inspect.log_format),
        max_connections=int(cfg.inspect.max_connections),
        max_tasks=int(cfg.inspect.get("max_tasks", 4)),
        retry_on_error=int(cfg.inspect.retry_on_error),
        max_tool_output=int(cfg.inspect.max_tool_output),
        fail_on_error=False,
        display="plain",
        temperature=float(gen.temperature),
        top_p=float(gen.top_p),
        max_tokens=int(gen.max_tokens),
    )
    run_info = {
        "target": model_str,
        "mode": target.spec.mode,
        "peer": "scripted"
        if scripted
        else (peer_default if peer_mode == "base" else peer_mode),
        "judge": str(cfg.judge.model),
        "cells": len(tasks),
        "seeds": seeds,
        "inspect_logs": str(inspect_log_dir.relative_to(out_dir)),
        "inspect_version": __import__("importlib.metadata").metadata.version(
            "inspect-ai"
        ),
    }
    return export(
        logs,
        rollout_dir,
        results_dir,
        list(cfg.ladder.endorsed),
        list(cfg.ladder.overreach),
        run_info,
    )
