# ABOUTME: MASK's judge and agentic misalignment's classifier are served by the ONE provider
# ABOUTME: configs/endpoints/providers.yaml pins for their model; the pin must reach each vendored harness's request.

import asyncio
import importlib
import importlib.util
import json
from types import SimpleNamespace

import pytest
import yaml

from src.infra.endpoints.openrouter import provider_pin


def _mask_evaluate(monkeypatch, name, provider_env):
    import dotenv
    from src.eval.misalignment.mask import runner
    monkeypatch.setenv("MASK_JUDGE_API_KEY", "x")
    if provider_env is None:
        monkeypatch.delenv("MASK_JUDGE_PROVIDER", raising=False)
    else:
        monkeypatch.setenv("MASK_JUDGE_PROVIDER", provider_env)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)
    monkeypatch.syspath_prepend(str(runner._HARNESS))
    spec = importlib.util.spec_from_file_location(name, runner._HARNESS / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_neither_eval_config_names_a_provider_of_its_own():
    # one registry, no per-config overrides: the eval configs choose the judge MODEL only
    for path in ("configs/eval/mask.yaml", "configs/eval/agentic_misalignment.yaml"):
        cfg = yaml.safe_load(open(path, encoding="utf-8"))
        assert not [k for k in cfg if "provider" in k], path
    mask = yaml.safe_load(open("configs/eval/mask.yaml", encoding="utf-8"))
    am = yaml.safe_load(open("configs/eval/agentic_misalignment.yaml", encoding="utf-8"))
    assert provider_pin(mask["judge_model"])["allow_fallbacks"] is False
    assert provider_pin(am["classifier_model"])["allow_fallbacks"] is False


def test_an_unlisted_judge_model_is_refused():
    with pytest.raises(ValueError):
        provider_pin("some-vendor/a-model-nobody-pinned")


def test_mask_live_judge_call_carries_the_registry_pin_beside_the_hashed_body(monkeypatch):
    pin = provider_pin("google/gemini-3-flash-preview")
    mod = _mask_evaluate(monkeypatch, "mask_evaluate_pinned", json.dumps(pin))
    assert mod._JUDGE_ROUTE == {"extra_body": {"provider": pin}}
    # the collect/replay key is a hash of the body alone, so the pin is not part of it
    body = {"messages": [{"role": "user", "content": "q"}], "max_completion_tokens": 500}
    assert mod._judge_key(body) == mod._judge_key(dict(body)) and "provider" not in body


def test_mask_standalone_harness_sends_no_route_only_when_the_runner_set_none(monkeypatch):
    assert _mask_evaluate(monkeypatch, "mask_evaluate_unpinned", None)._JUDGE_ROUTE == {}


def test_agentic_classifier_call_carries_the_registry_pin(monkeypatch):
    from src.eval.misalignment.agentic_misalignment.runner import _HARNESS
    monkeypatch.syspath_prepend(str(_HARNESS))
    mc = importlib.import_module("api_client.model_client")
    seen = {}

    async def create(**kw):
        seen.update(kw)
        return None          # the client turns a missing response into an error LLMResponse

    client = object.__new__(mc.ModelClient)
    client.openrouter_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    msgs = [mc.ChatMessage(role=mc.MessageRole.USER, content="classify")]
    pin = provider_pin("google/gemini-3-flash-preview")

    monkeypatch.setenv("OPENROUTER_PROVIDER_PIN", json.dumps(pin))
    asyncio.run(client._call_openrouter("google/gemini-3-flash-preview", msgs, max_tokens=10, temperature=0.0))
    assert seen["extra_body"] == {"provider": pin} and seen["model"] == "google/gemini-3-flash-preview"

    seen.clear()
    monkeypatch.delenv("OPENROUTER_PROVIDER_PIN")
    asyncio.run(client._call_openrouter("google/gemini-3-flash-preview", msgs, max_tokens=10, temperature=0.0))
    assert "extra_body" not in seen
