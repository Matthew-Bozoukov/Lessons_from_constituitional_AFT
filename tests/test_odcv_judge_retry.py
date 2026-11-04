# ABOUTME: A dropped connection must not cost an ODCV judging pass: transient judge errors retry, and a
# ABOUTME: resumed run reuses verdicts for byte-identical transcripts instead of re-paying for them. No network.
import json
import types
from types import SimpleNamespace

import httpx
import openai
import pytest

from src.eval.misalignment.odcv import odcv_judge

REQ = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")


def _status(code):
    return openai.APIStatusError(f"status {code}", response=httpx.Response(code, request=REQ), body=None)


def test_transient_failure_is_retried_on_the_schedule():
    calls, slept = [], []

    def call():
        calls.append(1)
        if len(calls) == 1:
            raise openai.APITimeoutError(request=REQ)
        return "verdict"

    assert odcv_judge.call_with_retries(call, delays=(5, 15), sleep=slept.append) == "verdict"
    assert len(calls) == 2 and slept == [5]


@pytest.mark.parametrize("exc", [openai.APIConnectionError(request=REQ), _status(429), _status(503)])
def test_connection_rate_limit_and_5xx_count_as_transient(exc):
    assert odcv_judge._transient(exc)


@pytest.mark.parametrize("code", [400, 401, 402, 404])
def test_an_answered_request_is_never_retried(code):
    slept = []

    def call():
        raise _status(code)

    with pytest.raises(openai.APIStatusError):
        odcv_judge.call_with_retries(call, delays=(5, 15), sleep=slept.append)
    assert slept == []


def test_retries_give_up_after_the_last_delay():
    slept = []

    def call():
        raise openai.APIConnectionError(request=REQ)

    with pytest.raises(openai.APIConnectionError):
        odcv_judge.call_with_retries(call, delays=(1, 2, 3), sleep=slept.append)
    assert slept == [1, 2, 3]


def test_budgeted_client_retries_and_keeps_the_failed_reservation(tmp_path, monkeypatch):
    monkeypatch.setattr(odcv_judge, "provider_price", lambda _: {"in": 1, "out": 1})
    monkeypatch.setattr(odcv_judge, "provider_pin", lambda _: {"order": ["test"]})
    monkeypatch.setattr(odcv_judge, "JUDGE_RETRY_DELAYS_S", (0,))
    sends = []
    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
                               choices=[SimpleNamespace(finish_reason="stop")])

    class FakeCompletions:
        def create(self, **kw):
            sends.append(kw)
            if len(sends) == 1:
                raise openai.APITimeoutError(request=REQ)
            return response

    class FakeOpenAI:
        def __init__(self, *a, **k):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    mod = types.ModuleType("fakejudge")
    mod.OpenAI = FakeOpenAI
    budget = odcv_judge.JudgeBudget(tmp_path / "ledger.json", 1, 100)
    monkeypatch.setattr(odcv_judge.time, "sleep", lambda _: None)
    odcv_judge.pin_provider(mod, "test-model", budget)
    out = mod.OpenAI().chat.completions.create(model="test-model", messages=[{"role": "user", "content": "x"}])
    assert out is response and len(sends) == 2
    assert all(s["extra_body"]["provider"] == {"order": ["test"]} for s in sends)
    rows = json.loads((tmp_path / "ledger.json").read_text())
    assert [r["status"] for r in rows] == ["reserved", "settled"]


def _fake_judge(scores):
    """A judge module whose evaluation_routine records every transcript it is asked about."""
    mod = types.ModuleType("fakejudge")
    mod.OpenAI = object
    mod.calls = []

    def evaluation_routine(path):
        mod.calls.append(path)
        return scores.get(open(path, encoding="utf-8").read(), 1), "because"

    mod.evaluation_routine = evaluation_routine
    return mod


def _items(tmp_path, texts):
    for run in ("run1", "run2", "run3", "run4"):
        (tmp_path / run).mkdir()
    items = []
    for i, text in enumerate(texts):
        rec = tmp_path / "pass" / f"Scenario{i}" / "messages_record.txt"
        rec.parent.mkdir(parents=True, exist_ok=True)
        rec.write_text(text, encoding="utf-8")
        items.append(("mandated", f"Scenario{i}", rec))
    return items


def test_a_resume_reuses_verdicts_for_identical_transcripts(tmp_path, monkeypatch):
    monkeypatch.setattr(odcv_judge, "provider_pin", lambda _: {})
    shared = tmp_path / "judge_cache"
    items = _items(tmp_path, ["one", "two", "three"])
    mod = _fake_judge({})

    first = odcv_judge.score_with_judge(mod, "judge-x", items, tmp_path / "run1" / "scores.json", 2,
                                        shared_cache=shared)
    assert len(mod.calls) == 3 and len(first) == 3

    # The next attempt judges into a NEW combined dir, so its per-run cache starts empty.
    mod.calls.clear()
    second = odcv_judge.score_with_judge(mod, "judge-x", items, tmp_path / "run2" / "scores.json", 2,
                                         shared_cache=shared)
    assert mod.calls == [] and second == first


def test_a_changed_transcript_another_judge_or_no_shared_cache_is_judged_afresh(tmp_path, monkeypatch):
    monkeypatch.setattr(odcv_judge, "provider_pin", lambda _: {})
    shared = tmp_path / "judge_cache"
    items = _items(tmp_path, ["one", "two"])
    mod = _fake_judge({})
    odcv_judge.score_with_judge(mod, "judge-x", items, tmp_path / "run1" / "scores.json", 1,
                                shared_cache=shared)

    items[1][2].write_text("two, re-run", encoding="utf-8")
    mod.calls.clear()
    odcv_judge.score_with_judge(mod, "judge-x", items, tmp_path / "run2" / "scores.json", 1,
                                shared_cache=shared)
    assert len(mod.calls) == 1

    mod.calls.clear()
    odcv_judge.score_with_judge(mod, "judge-y", items, tmp_path / "run3" / "scores.json", 1,
                                shared_cache=shared)
    assert len(mod.calls) == 2

    mod.calls.clear()
    odcv_judge.score_with_judge(mod, "judge-x", items, tmp_path / "run4" / "scores.json", 1)
    assert len(mod.calls) == 2


def test_a_failed_verdict_is_not_shared(tmp_path, monkeypatch):
    monkeypatch.setattr(odcv_judge, "provider_pin", lambda _: {})
    shared = tmp_path / "judge_cache"
    items = _items(tmp_path, ["unjudgeable"])
    mod = _fake_judge({"unjudgeable": "N/A"})
    odcv_judge.score_with_judge(mod, "judge-x", items, tmp_path / "run1" / "scores.json", 1,
                                shared_cache=shared)
    mod.calls.clear()
    odcv_judge.score_with_judge(mod, "judge-x", items, tmp_path / "run2" / "scores.json", 1,
                                shared_cache=shared)
    assert len(mod.calls) == 1
