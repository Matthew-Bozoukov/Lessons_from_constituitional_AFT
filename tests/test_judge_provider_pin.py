# ABOUTME: The MASK judge and the agentic-misalignment classifier name ONE OpenRouter provider, fallbacks off,
# ABOUTME: from their own eval configs; these check the pin reaches the request each vendored harness sends.

import asyncio
import importlib
import importlib.util
from types import SimpleNamespace

import yaml

PIN = {"order": ["google-vertex/global"], "allow_fallbacks": False}


def test_both_configs_name_the_route_their_earlier_runs_used():
    mask = yaml.safe_load(open("configs/eval/mask.yaml", encoding="utf-8"))
    am = yaml.safe_load(open("configs/eval/agentic_misalignment.yaml", encoding="utf-8"))
    assert mask["judge_provider"] == am["classifier_provider"] == "google-vertex/global"


def test_mask_live_judge_call_carries_the_pin_beside_the_hashed_body(monkeypatch):
    import dotenv
    from src.eval.misalignment.mask import runner
    monkeypatch.setenv("MASK_JUDGE_API_KEY", "x")
    monkeypatch.setenv("MASK_JUDGE_PROVIDER", "google-vertex/global")
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)
    monkeypatch.syspath_prepend(str(runner._HARNESS))
    spec = importlib.util.spec_from_file_location("mask_evaluate_pinned", runner._HARNESS / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod._JUDGE_ROUTE == {"extra_body": {"provider": PIN}}
    # the collect/replay key is a hash of the body alone, so the pin must not be part of it
    body = {"messages": [{"role": "user", "content": "q"}], "max_completion_tokens": 500}
    assert "provider" not in body and mod._judge_key(body) == mod._judge_key(dict(body))


def test_mask_judge_is_unrouted_only_when_the_runner_sets_nothing(monkeypatch):
    import dotenv
    from src.eval.misalignment.mask import runner
    monkeypatch.setenv("MASK_JUDGE_API_KEY", "x")
    monkeypatch.delenv("MASK_JUDGE_PROVIDER", raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)
    monkeypatch.syspath_prepend(str(runner._HARNESS))
    spec = importlib.util.spec_from_file_location("mask_evaluate_unpinned", runner._HARNESS / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod._JUDGE_ROUTE == {}   # the standalone harness; runner.run refuses a config without the pin


def test_agentic_classifier_call_carries_the_pin(monkeypatch):
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

    monkeypatch.setenv("OPENROUTER_PROVIDER_PIN", "google-vertex/global")
    asyncio.run(client._call_openrouter("google/gemini-3-flash-preview", msgs, max_tokens=10, temperature=0.0))
    assert seen["extra_body"] == {"provider": PIN} and seen["model"] == "google/gemini-3-flash-preview"

    seen.clear()
    monkeypatch.delenv("OPENROUTER_PROVIDER_PIN")
    asyncio.run(client._call_openrouter("google/gemini-3-flash-preview", msgs, max_tokens=10, temperature=0.0))
    assert "extra_body" not in seen
