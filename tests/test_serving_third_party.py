# ABOUTME: Offline tests for serving a family we evaluate but do not train: Qwen3-32B's serving-only facts,
# ABOUTME: declared thinking modes for adapters trained elsewhere, and the explicit YaRN rope scaling an eval may declare.

import pytest

from src.infra.endpoints.vllm import THIRD_PARTY_MODES, _spec_from_files, plan_serving
from src.model_profile import model_profile, serving_params

ADAPTER = {"base_model_name_or_path": "Qwen/Qwen3-32B", "r": 64}
YARN = {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 32768}


def test_qwen3_32b_serves_with_hermes_but_stays_untrainable():
    facts = serving_params("Qwen/Qwen3-32B")
    assert (
        facts["tool_call_parser"] == "hermes" and facts["reasoning_parser"] == "qwen3"
    )
    assert facts["supports_prefix_caching"] is True
    # Qwen3.6 keeps its own facts (no substring collision), and the training lookup still
    # refuses Qwen3: a serving-only entry does not make a family trainable.
    assert serving_params("Qwen/Qwen3.6-27B")["tool_call_parser"] == "qwen3_xml"
    with pytest.raises((Exception, SystemExit)):
        model_profile("Qwen/Qwen3-32B")


def test_declared_third_party_mode_replaces_the_missing_stamp():
    spec = _spec_from_files(
        "chloeli/qwen-3-32b-philosophy-spec-msm-aft-cot", ADAPTER, None
    )
    assert spec.mode == "think" and spec.adapter and spec.lora_rank == 64
    assert spec.base_model == "Qwen/Qwen3-32B"
    for repo, (mode, reason) in THIRD_PARTY_MODES.items():
        assert mode in ("think", "nothink") and reason, repo
    # Every organism one Hospital episode seats must share a mode (ServedTarget.sibling).
    assert len({mode for mode, _ in THIRD_PARTY_MODES.values()}) == 1
    # Not listed: still the hard error, and it names the table.
    with pytest.raises(RuntimeError, match="THIRD_PARTY_MODES"):
        _spec_from_files("someone/unlisted-lora", ADAPTER, None)


def test_rope_scaling_is_explicit_yarn_and_bounds_the_window():
    facts = dict(serving_params("Qwen/Qwen3-32B"), native_context_window=40960)
    plan = plan_serving(
        facts,
        {"context_window": 65536, "needs_tool_calls": True, "rope_scaling": YARN},
        "Qwen/Qwen3-32B",
        "think",
    )
    assert plan["hf_overrides"] == {"rope_scaling": YARN}
    assert plan["context_window"] == 65536 and plan["tool_call_parser"] == "hermes"
    assert any("YaRN" in w for w in plan["warnings"])
    # Without it, a window above native is refused exactly as before.
    with pytest.raises(SystemExit, match="native window"):
        plan_serving(facts, {"context_window": 65536}, "m", "think")
    # Beyond what the declared factor gives: refused.
    with pytest.raises(SystemExit, match="exceeds"):
        plan_serving(
            facts, {"context_window": 98304, "rope_scaling": YARN}, "m", "think"
        )
    # YaRN only, and exactly its three keys.
    with pytest.raises(SystemExit, match="rope_scaling"):
        plan_serving(
            facts,
            {"context_window": 65536, "rope_scaling": dict(YARN, rope_type="linear")},
            "m",
            "think",
        )
    with pytest.raises(SystemExit, match="rope_scaling"):
        plan_serving(
            facts,
            {"context_window": 65536, "rope_scaling": dict(YARN, beta_fast=32)},
            "m",
            "think",
        )
    # Declared scaling cannot start above the trained window.
    with pytest.raises(SystemExit, match="original_max_position_embeddings"):
        plan_serving(
            facts,
            {
                "context_window": 65536,
                "rope_scaling": dict(YARN, original_max_position_embeddings=50000),
            },
            "m",
            "think",
        )
    # Nothing declared: nothing emitted.
    assert (
        plan_serving(facts, {"context_window": 32768}, "m", "think")["hf_overrides"]
        is None
    )
