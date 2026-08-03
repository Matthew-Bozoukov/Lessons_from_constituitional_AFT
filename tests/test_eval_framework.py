# ABOUTME: Offline tests for the eval framework's pure logic: target resolution, thinking-mode
# ABOUTME: template pinning, registry shape, and dataset-card field enforcement.

import pytest

from src.endpoints.vllm_server import TargetSpec, _spec_from_files, pin_template
from src.eval import EVALS, EvalSpec
from src.eval.publish import REQUIRED_FIELDS, card_markdown

ADAPTER_CONFIG = {"base_model_name_or_path": "Qwen/Qwen3-32B", "r": 16}


def test_full_model_uses_template_default_mode():
    spec = _spec_from_files("Qwen/Qwen3-32B", None, None)
    assert spec == TargetSpec(hf_path="Qwen/Qwen3-32B", base_model="Qwen/Qwen3-32B",
                              adapter=False, mode="default", model_key="Qwen3-32B",
                              lora_rank=None)


def test_adapter_mode_comes_from_training_meta():
    think = _spec_from_files("org/arm-lora", ADAPTER_CONFIG, {"thinking": True})
    nothink = _spec_from_files("org/arm-lora", ADAPTER_CONFIG, {"thinking": False})
    assert (think.mode, nothink.mode) == ("think", "nothink")
    assert think.base_model == "Qwen/Qwen3-32B" and think.lora_rank == 16


def test_adapter_without_stamp_is_a_hard_error():
    with pytest.raises(RuntimeError, match="training_meta.json"):
        _spec_from_files("org/legacy-lora", ADAPTER_CONFIG, None)
    with pytest.raises(AssertionError, match="thinking"):
        _spec_from_files("org/bad-lora", ADAPTER_CONFIG, {"mode": "think"})


def test_pin_template_shadows_request_kwargs():
    template = "{%- if enable_thinking %}T{% else %}N{% endif %}"
    pinned = pin_template(template, "nothink")
    # The pin is a top-level set BEFORE the original template, so it wins over any
    # enable_thinking a client passes per request.
    assert pinned.startswith("{%- set enable_thinking = false -%}\n")
    assert pin_template(template, "think").startswith("{%- set enable_thinking = true -%}\n")
    with pytest.raises(AssertionError):
        pin_template(template, "default")


def test_registry_specs_are_wellformed():
    assert EVALS, "registry is empty"
    for name, spec in EVALS.items():
        assert isinstance(spec, EvalSpec), name
        module, _, func = spec.runner.partition(":")
        assert module.startswith("src.eval.") and func, (name, spec.runner)
        assert spec.config.startswith("configs/eval/"), (name, spec.config)


def test_card_markdown_enforces_required_fields():
    fields = {f: "x" for f in REQUIRED_FIELDS}
    text = card_markdown(fields)
    assert all(f"| `{f}` |" in text for f in REQUIRED_FIELDS)
    with pytest.raises(AssertionError, match="constitution"):
        card_markdown({f: "x" for f in REQUIRED_FIELDS if f != "constitution"})
