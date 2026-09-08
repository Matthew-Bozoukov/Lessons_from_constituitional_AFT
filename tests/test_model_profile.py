# ABOUTME: configs/models/<key>.yaml is the ModelProfile registry: every file loads, a key
# ABOUTME: resolves from an id or a path, stubs are named and served but never trained, stamps round-trip.

from __future__ import annotations

import pytest

from src.model_profile import (
    DEFAULT_SERVING,
    PROFILES_DIR,
    ModelProfile,
    find_profile,
    gpu_for,
    model_key,
    model_keys,
    model_profile,
    profiles,
    serving_params,
)


def test_every_profile_file_loads_and_its_stem_is_its_key():
    stems = {p.stem for p in PROFILES_DIR.glob("*.yaml")}
    assert stems and stems == model_keys()
    for p in profiles():
        assert p.model and p.match and p.family, p.key


def test_a_key_an_id_a_path_and_a_served_name_are_one_model():
    assert (model_key("qwen36") == model_key("Qwen/Qwen3.6-27B") == model_key("/root/qwen36")
            == model_key("qwen3_6-27b") == "qwen36")
    assert model_key("Qwen/Qwen3-32B") == "qwen3"
    assert model_key("Qwen/Qwen3-0.6B") == "qwen306b"
    assert model_key("openai/gpt-oss-120b") == "gptoss120b"
    with pytest.raises(ValueError, match="configs/models"):
        model_key("mistralai/Mistral-7B")
    assert find_profile("mistralai/Mistral-7B") is None


def test_only_a_verified_family_may_be_trained_but_a_stub_is_still_named_and_served():
    p = model_profile("qwen36")
    assert p.verified and p.lora_target_modules and p.render_kwargs == {"preserve_thinking": True}
    assert p.model_class == "image_text_to_text" and p.load_in_4bit is False
    with pytest.raises(ValueError, match="no verified thinking profile"):
        model_profile("Qwen/Qwen3-32B")
    assert model_key("Qwen/Qwen3-32B") == "qwen3"
    assert serving_params("Qwen/Qwen3-32B") is DEFAULT_SERVING
    assert gpu_for("Qwen/Qwen3-32B", "train") is None


def test_gpu_and_serving_facts_come_from_the_file():
    assert gpu_for("qwen36", "train") == "NVIDIA H200"
    assert gpu_for("Qwen/Qwen3.6-27B", "inference") == "NVIDIA H100 80GB HBM3"
    facts = serving_params("Qwen/Qwen3.6-27B")
    assert facts["tool_call_parser"] == "qwen3_xml" and facts["reasoning_parser"] == "qwen3"
    assert model_profile("qwen36").train_memory["H200"]["max_padded_tokens"] == 8000


def test_a_stamp_round_trips_verbatim():
    """A train run stamps `profile.to_dict()` into its train_config.yaml; a rerun rebuilds
    the very profile it rendered and masked with, whatever configs/models/ says by then."""
    p = model_profile("qwen36")
    d = p.to_dict()
    assert d["key"] == "qwen36" and "template" in d and "train" in d and "serving" in d
    assert ModelProfile.from_dict(d) == p


def test_a_template_block_states_all_five_literals_or_no_block_at_all():
    with pytest.raises(AssertionError, match="missing"):
        ModelProfile.from_dict({"model": "x/y", "template": {"prefill": "<think>\n"}}, key="x")
    stub = ModelProfile.from_dict({"model": "x/y"}, key="x")
    assert not stub.verified and stub.serving is None and stub.match == "x"
    assert "template" not in stub.to_dict()
    with pytest.raises(AssertionError, match="model profile needs a key"):
        ModelProfile.from_dict({"model": "x/y"})
