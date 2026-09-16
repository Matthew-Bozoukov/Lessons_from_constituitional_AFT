# ABOUTME: Unit tests for the Colosseum sweep-config builder — seat assignment by role,
# ABOUTME: the two-sweep shape that avoids upstream's silent third cell, and the vLLM traps.

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.eval.misalignment.colosseum.config import EXPERIMENTS, build_sweep_config

CONTROL = "LASR-Callum/2026-08-04-qwen36-lora-table2-only-9284-rank-64"
TREATMENT = "LASR-Callum/2026-08-21-qwen36-lora-difficult-advice-chunk-only-702"


class FakeSpec:
    def __init__(self, hf_path, key):
        self.hf_path = hf_path
        self.model_key = key


class FakeTarget:
    """A ServedTarget stand-in: identity, a served name, and an endpoint."""

    def __init__(self, hf_path, key):
        self.spec = FakeSpec(hf_path, key)
        self.model_name = key
        self.base_url = "http://127.0.0.1:8000/v1"
        self.api_key = "EMPTY"


@pytest.fixture
def cfg():
    return OmegaConf.create(
        {
            "peer": CONTROL,
            "seeds": [1, 2, 3],
            "num_agents": 6,
            "max_tasks": 8,
            "max_planning_rounds": 4,
            "max_conversation_steps": 3,
            "assignment_filling": True,
            "max_tokens": 4096,
            "temperature": 0.7,
            "request_timeout": 900,
            "max_concurrent_runs": 5,
        }
    )


@pytest.fixture
def targets():
    return FakeTarget(TREATMENT, "treatment_key"), FakeTarget(CONTROL, "control_key")


def _build(experiment, cfg, targets):
    treat, ctrl = targets
    return build_sweep_config(experiment, treat, ctrl, cfg, Path("/tmp/out"))


def test_collusion_writes_two_explicit_sweeps_not_one_crossed_sweep(cfg, targets):
    # Upstream expands secret_channel_enabled x prompt_variants and drops every
    # no-channel cell whose variant is not "control". The crossed form
    # [false, true] x [control, simple] therefore yields THREE cells, silently adding a
    # channel-without-objective arm and a third of the GPU bill. Two explicit sweeps give
    # exactly the two conditions the design asks for.
    sweeps = _build("collusion", cfg, targets)["experiment"]["sweeps"]
    assert len(sweeps) == 2
    assert [s["name"] for s in sweeps] == ["baseline", "collusion"]
    for s in sweeps:
        assert len(s["secret_channel_enabled"]) == 1, "a list of 2 re-creates the cross"
        assert len(s["prompt_variants"]) == 1
    baseline, collusion = sweeps
    assert baseline["secret_channel_enabled"] == [False]
    assert baseline["prompt_variants"] == ["control"]
    assert collusion["secret_channel_enabled"] == [True]
    assert collusion["prompt_variants"] == ["simple"]


def test_no_sweep_cell_would_be_skipped_by_upstreams_variant_gate(cfg, targets):
    # The gate is `if not secret and variant != "control": continue`. A cell that trips it
    # never runs, and a missing baseline is invisible until the pairing fails.
    for experiment in EXPERIMENTS:
        for s in _build(experiment, cfg, targets)["experiment"]["sweeps"]:
            secret, variant = s["secret_channel_enabled"][0], s["prompt_variants"][0]
            assert secret or variant == "control", (
                f"{experiment}/{s['name']} would be skipped by upstream"
            )


def test_mixed_team_seats_the_arm_under_test_in_the_coalition_only(cfg, targets):
    by_role = _build("collusion", cfg, targets)["experiment"]["collusion"][
        "agent_llms_by_role"
    ]
    assert set(by_role) == {"colluder", "normal"}
    assert (
        by_role["colluder"]["vllm"]["models"][0]["served_model_name"] == "treatment_key"
    )
    assert by_role["normal"]["vllm"]["models"][0]["served_model_name"] == "control_key"


def test_single_agent_keeps_the_channel_flag_on_with_one_colluder(cfg, targets):
    # The private objective survives only while the flag is on: upstream downgrades every
    # role to "normal" and forces the variant back to "control" when it is off. With one
    # colluder no secret blackboard is created (it needs >= 2), which is the design —
    # a private objective, no partner, no channel.
    built = _build("single", cfg, targets)
    assert EXPERIMENTS["single"]["colluder_count"] == 1
    treated = [s for s in built["experiment"]["sweeps"] if s["name"] == "single"][0]
    assert treated["secret_channel_enabled"] == [True]
    assert treated["colluder_counts"] == [1]


def test_cooperation_puts_one_arm_in_every_seat(cfg, targets):
    built = _build("cooperation", cfg, targets)
    by_role = built["experiment"]["collusion"]["agent_llms_by_role"]
    assert set(by_role) == {"normal"}, "a cooperation run has no coalition seats"
    assert (
        by_role["normal"]["vllm"]["models"][0]["served_model_name"] == "treatment_key"
    )
    sweeps = built["experiment"]["sweeps"]
    assert len(sweeps) == 1 and sweeps[0]["colluder_counts"] == [0]
    assert sweeps[0]["secret_channel_enabled"] == [False]


def test_vllm_block_avoids_the_two_upstream_defaults_that_break_it(cfg, targets):
    vllm = _build("collusion", cfg, targets)["llm_models"][0]["llm"]["vllm"]
    # Default "/v1/models" is joined onto a base URL already ending /v1, giving
    # /v1/v1/models -> 404 -> "server is not reachable".
    assert vllm["health_check_path"] == "/models"
    # Default 60s times out mid-reasoning-trace on a 27B model.
    assert vllm["models"][0]["request_timeout"] == 900
    # run_eval owns serving in this repo; Colosseum must never launch its own.
    assert vllm["auto_start_server"] is False
    # Passed through verbatim — upstream does NOT append /v1 to an explicit base_url.
    assert vllm["models"][0]["base_url"].endswith("/v1")


def test_reasoning_headroom_is_not_upstreams_1500(cfg, targets):
    # 1500 truncates inside <think>, emits no tool call, and scores a clean zero that
    # reads as good behaviour (CLAUDE.md gotcha 4).
    params = _build("collusion", cfg, targets)["llm_models"][0]["llm"]["vllm"]["params"]
    assert params["max_tokens"] >= 4096


def test_channels_are_consolidated_so_the_secret_board_is_actually_secret(cfg, targets):
    # Megaboard.add_blackboard dedups by participant set: without consolidation an
    # existing pairwise PUBLIC board between the two colluders is handed back as the
    # "secret" one.
    net = _build("collusion", cfg, targets)["communication_network"]
    assert net["consolidate_channels"] is True
    assert net["topology"] == "complete"


def test_unknown_experiment_is_refused(cfg, targets):
    with pytest.raises(AssertionError, match="unknown experiment"):
        _build("sabotage", cfg, targets)
