# ABOUTME: Build the Colosseum sweep YAML for one experiment from this repo's eval config
# ABOUTME: and the two live vLLM targets — the only place seat assignment is decided.

"""Our eval config plus two served adapters -> the YAML `experiments.collusion.run` reads.

Three experiments differ ONLY in what this module writes, which is why they live here
rather than in three runners:

    collusion    2 coalition seats hold the ARM UNDER TEST, 4 hold the control peer.
                 Two cells per arm: a baseline with no channel and no private objective,
                 and a collusion cell with both. Paired on seed.
    single       1 seat holds the arm under test with the private objective and NO
                 partner and NO channel; the other 5 hold the control peer. The
                 single-agent effect size the transfer ratio divides by.
    cooperation  no private objective, no channel, and all six seats the same arm.
                 Catches a "safer" arm that is really just a worse teammate.

The two cells of `collusion` are written as two SWEEP ENTRIES rather than one crossed
sweep, and that is not cosmetic. Upstream expands `secret_channel_enabled x
prompt_variants` and skips every no-channel cell whose variant is not `control`
(`run.py`: `if not bool(_secret) and str(_pv) != "control": continue`), so the crossed
form `[false, true] x [control, simple]` yields THREE cells — it silently adds a
channel-without-objective arm and a third of the GPU bill with it. Two explicit sweeps
give exactly the two conditions the design asks for.
"""

from __future__ import annotations

from typing import Any

from src.infra.endpoints.vllm import ServedTarget

__all__ = ["EXPERIMENTS", "llm_block", "build_sweep_config"]

# colluder_count and channel per experiment. `single` keeps the channel flag ON with one
# colluder: upstream creates a secret blackboard only at `len(colluders) >= 2`, so the one
# agent gets the private objective with no partner and no channel — Experiment 2 exactly —
# while the flag being on is what stops the role being downgraded to "normal" and the
# prompt variant being forced back to "control".
EXPERIMENTS: dict[str, dict[str, Any]] = {
    "collusion": {
        "colluder_count": 2,
        "cells": [("baseline", False, "control"), ("collusion", True, "simple")],
        "mixed_team": True,
    },
    "single": {
        "colluder_count": 1,
        "cells": [("baseline", False, "control"), ("single", True, "simple")],
        "mixed_team": True,
    },
    "cooperation": {
        "colluder_count": 0,
        "cells": [("cooperation", False, "control")],
        "mixed_team": False,
    },
}


def llm_block(target: ServedTarget, cfg) -> dict[str, Any]:
    """The `llm` block pointing Colosseum at one already-running vLLM model.

    `auto_start_server: false` because run_eval owns serving in this repo — Colosseum
    attaches to the server it finds and never launches one, so exactly one piece of code
    decides how a model under measurement is served.

    Two upstream defaults are overridden here and both are load-bearing:

    `health_check_path` defaults to `/v1/models`, which upstream joins onto a base URL
    that already ends in `/v1` — producing `/v1/v1/models`, a 404, and the misleading
    error "server is not reachable". `/models` is what every shipped example uses.

    `request_timeout` defaults to 60s. This is a 27B model in THINKING mode: a turn is a
    reasoning trace plus an answer, and 60s truncates it into a timeout that looks like a
    dead server.
    """
    return {
        "provider": "vllm",
        "vllm": {
            "auto_start_server": False,
            "health_check_path": "/models",
            "params": {
                # Gotcha 4 in CLAUDE.md: a reasoning model with a tight cap truncates
                # INSIDE <think>, emits no answer and no tool call, and scores a clean
                # zero that looks like refusal. Colosseum's own configs use 1500, which is
                # sized for non-reasoning models.
                "max_tokens": int(cfg.max_tokens),
                "temperature": float(cfg.temperature),
            },
            "models": [
                {
                    # Both fields carry the vLLM --served-model-name (the adapter's key, or
                    # "base"). Upstream requires this string to appear VERBATIM in
                    # GET <base_url>/models or it refuses to start, so it is taken from the
                    # ServedTarget rather than spelled out in a config where it could drift.
                    "checkpoint": target.model_name,
                    "served_model_name": target.model_name,
                    # Passed through verbatim: upstream appends /v1 to a host/port pair but
                    # NOT to an explicit base_url, despite its docstring.
                    "base_url": target.base_url,
                    "api_key": target.api_key,
                    "request_timeout": int(cfg.request_timeout),
                }
            ],
        },
    }


def build_sweep_config(
    experiment: str, target: ServedTarget, peer: ServedTarget, cfg, out_dir
) -> dict[str, Any]:
    """The full Colosseum config for one experiment and one arm under test.

    Args:
        experiment: A key of EXPERIMENTS.
        target: The arm under test — the coalition seats (or every seat, for
            `cooperation`).
        peer: The control checkpoint filling the remaining seats. For a control-arm block
            this is the same model as `target`, which is correct and costs nothing: both
            resolve to one adapter on one server.
        cfg: The eval config (configs/eval/colosseum_jira.yaml).
        out_dir: The run directory; Colosseum's tree lands under `rollouts/`.

    Returns:
        A dict ready to be dumped as the YAML `experiments.collusion.run --config` reads.
    """
    assert experiment in EXPERIMENTS, (
        f"unknown experiment {experiment!r}; expected one of {sorted(EXPERIMENTS)}"
    )
    spec = EXPERIMENTS[experiment]
    target_llm, peer_llm = llm_block(target, cfg), llm_block(peer, cfg)

    # Seat assignment, and the only place it is decided. Keyed on the collusion ROLE
    # because `colluder_selection: random` picks the coalition from the seed at run time —
    # a positional list would be guessing which seats end up colluding. `cooperation` has
    # no colluders at all, so every seat resolves through "normal".
    by_role = (
        {"colluder": target_llm, "normal": peer_llm}
        if spec["mixed_team"]
        else {"normal": target_llm}
    )

    return {
        "simulation": {
            "max_iterations": 1,
            "max_planning_rounds": int(cfg.max_planning_rounds),
            "max_conversation_steps": int(cfg.max_conversation_steps),
            "tags": ["colosseum_jira", experiment],
        },
        "environment": {
            "name": "JiraTicketEnvironment",
            # Pinned deliberately: upstream defaults this to True and its own shipped
            # configs disagree with each other about it. With it on, an agent that
            # assigned nothing has a random task filled in at scoring time, which moves
            # regret for a reason that has nothing to do with the model.
            "assignment_filling": bool(cfg.assignment_filling),
            "max_tasks": int(cfg.max_tasks),
        },
        "communication_network": {
            "topology": "complete",
            "num_agents": int(cfg.num_agents),
            # Required, not stylistic: Megaboard.add_blackboard dedups by participant set,
            # so without consolidation an existing pairwise public board between the two
            # colluders is handed back AS the secret board, and the "secret" channel is
            # one everybody can read.
            "consolidate_channels": True,
        },
        "llm_models": [{"label": target.spec.model_key, "llm": target_llm}],
        "experiment": {
            "tag": f"colosseum_jira_{experiment}",
            "output_dir": str(out_dir / "rollouts" / "colosseum"),
            "max_concurrent_runs": int(cfg.max_concurrent_runs),
            "seeds": [int(s) for s in cfg.seeds],
            "collusion": {
                "colluder_selection": "random",
                "agent_llms_by_role": by_role,  # third_party/per_agent_models.patch
            },
            "sweeps": [
                {
                    "name": name,
                    "topologies": ["complete"],
                    "num_agents": [int(cfg.num_agents)],
                    "colluder_counts": [spec["colluder_count"]],
                    "secret_channel_enabled": [secret],
                    "prompt_variants": [variant],
                }
                for name, secret, variant in spec["cells"]
            ],
        },
    }
