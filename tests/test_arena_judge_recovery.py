# ABOUTME: Offline regressions for Arena judge completion validity and exact caches.
# ABOUTME: Exercise failed attempts, ordering recovery, atomic checkpoints and evidence.
import copy
import importlib.util
import json
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from src.eval.capabilities.arena_hard import arena_hard_judge as driver


@pytest.fixture
def vendor(monkeypatch):
    root = Path("src/eval/capabilities/arena_hard/third_party/arena-hard-auto")
    saved = {key: value for key, value in sys.modules.items()
             if key == "utils" or key.startswith("utils.")}
    package = types.ModuleType("utils")
    package.__path__ = [str(root / "utils")]
    for key in saved:
        sys.modules.pop(key)
    sys.modules["utils"] = package
    spec = importlib.util.spec_from_file_location("arena_judge_recovery_vendor", root / "gen_judgment.py")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        for key in list(sys.modules):
            if key == "utils" or key.startswith("utils."):
                sys.modules.pop(key)
        sys.modules.update(saved)


def arguments(tmp_path):
    def answer(model, text):
        return {"model": model, "messages": [{"content": {"answer": text}}],
                "request_hash": model + "-generation"}
    return {
        "question": {"uid": "q1", "category": "hard_prompt", "prompt": "What is 2+2?"},
        "baseline": answer("baseline", "Four."),
        "answer": answer("target", "4."), "reference": None,
        "configs": {"judge_model": "judge", "temperature": 0, "max_tokens": 100,
                    "max_attempts": 2, "regex_patterns": [r"\[\[([AB<>=]+)\]\]"],
                    "prompt_template": "{QUESTION}\nA: {ANSWER_A}\nB: {ANSWER_B}"},
        "settings": {"model": "judge", "api_type": "fixture", "parallel": 2,
                     "extra_body": {"reasoning": {"effort": "low"}},
                     "endpoints": [{"api_base": "https://fixture.invalid/v1", "api_key": "secret"}]},
        "output_file": str(tmp_path / "target.jsonl"),
    }


def completion(finish="stop", text="Verdict: [[A=B]]"):
    return {"answer": text, "finish_reason": finish, "model_id": "served-judge",
            "reasoning": {"reasoning_content": "raw reasoning"},
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "reasoning_tokens": 2},
            "error": None}


def register(vendor, values):
    calls = []
    iterator = iter(values)
    def fake(**kwargs):
        calls.append(kwargs)
        value = next(iterator)
        if isinstance(value, Exception):
            raise value
        return copy.deepcopy(value)
    vendor.registered_api_completion["fixture"] = fake
    return calls


@pytest.mark.parametrize("finish", [None, "length", "content_filter", "tool_calls"])
def test_score_bearing_incomplete_judgment_is_never_valid(vendor, tmp_path, finish):
    args = arguments(tmp_path)
    register(vendor, [completion(finish)] * 4)
    row = vendor.judgment(args)
    assert all(game["score"] is None for game in row["games"])
    assert all(game["parsed_score"] == "A=B" for game in row["games"])
    assert all(len(game["attempts"]) == 2 for game in row["games"])
    assert row["games"][0]["judgment"]["reasoning"]["reasoning_content"] == "raw reasoning"
    with pytest.raises(ValueError, match="incomplete paired"):
        driver._complete_judgments([row], [args["question"]])


def test_recovery_retries_only_unresolved_ordering_and_retains_failures(vendor, tmp_path):
    args = arguments(tmp_path)
    calls = register(vendor, [completion(), completion("length"), completion("length")])
    first = vendor.judgment(args)
    assert len(calls) == 3
    first_id = first["games"][0]["attempt_id"]
    calls = register(vendor, [completion()])
    recovered = vendor.judgment(args)
    assert len(calls) == 1
    assert recovered["games"][0]["attempt_id"] == first_id
    assert [a["judgment"]["finish_reason"] for a in recovered["games"][1]["attempts"]] == ["length", "length", "stop"]
    assert driver._complete_judgments([recovered], [args["question"]]) == [recovered]
    assert driver._cost([recovered], "judge")["n_calls"] == 4
    calls = register(vendor, [])
    assert vendor.judgment(args) == recovered
    assert not calls
    assert len(Path(args["output_file"]).read_text().splitlines()) == 1


@pytest.mark.parametrize("field", ["prompt", "answer", "rubric", "temperature", "max_tokens", "reasoning", "judge", "endpoint", "generation"])
def test_request_changes_invalidate_cached_judgments(vendor, tmp_path, field):
    args = arguments(tmp_path)
    register(vendor, [completion(), completion()])
    old = vendor.judgment(args)
    if field == "prompt":
        args["question"]["prompt"] += " Explain."
    elif field == "answer":
        args["answer"]["messages"][-1]["content"]["answer"] = "Different answer"
    elif field == "rubric":
        args["configs"]["prompt_template"] += "\nExtra rubric"
    elif field in {"temperature", "max_tokens"}:
        args["configs"][field] += 1
    elif field == "reasoning":
        args["settings"]["extra_body"]["reasoning"]["effort"] = "high"
    elif field == "judge":
        args["configs"]["judge_model"] = args["settings"]["model"] = "other-judge"
    elif field == "endpoint":
        args["settings"]["endpoints"][0]["api_base"] = "https://different.invalid/v1"
    else:
        args["answer"]["request_hash"] = "other-model-revision"
    calls = register(vendor, [completion(), completion()])
    current = vendor.judgment(args)
    assert len(calls) == 2
    assert current["games"][0]["request_hash"] != old["games"][0]["request_hash"]
    assert current["prior_judgments"] == [old]
    assert driver._cost([current], "judge")["n_calls"] == 4
    assert "secret" not in Path(args["output_file"]).read_text()


def test_key_rotation_and_concurrency_do_not_change_request(vendor, tmp_path):
    args = arguments(tmp_path)
    register(vendor, [completion(), completion()])
    old = vendor.judgment(args)
    args["settings"]["parallel"] = 16
    args["settings"]["endpoints"][0]["api_key"] = "rotated-secret"
    calls = register(vendor, [])
    assert vendor.judgment(args) == old
    assert calls == []
    assert old["games"][0]["request_hash"] != old["games"][1]["request_hash"]


def test_legacy_or_null_cache_is_rejudged_but_preserved(vendor, tmp_path):
    args = arguments(tmp_path)
    legacy = {"uid": "q1", "category": "hard_prompt", "model": "target",
              "games": [{"score": "A=B"}, None]}
    Path(args["output_file"]).write_text(json.dumps(legacy) + "\n")
    calls = register(vendor, [completion(), completion()])
    row = vendor.judgment(args)
    assert len(calls) == 2
    assert row["prior_judgments"] == [legacy]
    with pytest.raises(ValueError, match="incomplete paired"):
        driver._complete_judgments([legacy], [args["question"]])


def test_invalid_parser_and_transport_attempts_are_retained(vendor, tmp_path):
    args = arguments(tmp_path)
    args["configs"]["max_attempts"] = 3
    calls = register(vendor, [RuntimeError("rejected secret"), completion(text="No verdict"), completion(), completion()])
    row = vendor.judgment(args)
    attempts = row["games"][0]["attempts"]
    assert len(calls) == 4 and len(attempts) == 3
    assert attempts[0]["judgment"]["error"]["message"] == "rejected [REDACTED]"
    assert attempts[1]["invalid_reasons"] == ["invalid_verdict"]
    assert attempts[2]["status"] == "complete"


def test_interruption_keeps_completed_opposite_ordering(vendor, tmp_path):
    args = arguments(tmp_path)
    register(vendor, [completion("length"), completion("length"), completion()])
    previous = vendor.judgment(args)
    saved = vendor.JudgmentStore(args["output_file"])
    save = saved.save
    def interrupted(row):
        save(row)
        raise KeyboardInterrupt("simulated worker interruption")
    saved.save = interrupted
    args["store"] = saved
    register(vendor, [completion("length")])
    with pytest.raises(KeyboardInterrupt):
        vendor.judgment(args)
    recovered = vendor.JudgmentStore(args["output_file"]).records["q1"]
    assert recovered["games"][1] == previous["games"][1]
    assert len(recovered["games"][0]["attempts"]) == 3


def test_parallel_atomic_checkpoints_preserve_all_uids(vendor, tmp_path):
    args = arguments(tmp_path)
    vendor.registered_api_completion["fixture"] = lambda **kwargs: completion()
    store = vendor.JudgmentStore(args["output_file"])
    def run(index):
        item = copy.deepcopy(args)
        item["question"]["uid"] = f"q{index}"
        item["store"] = store
        return vendor.judgment(item)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(run, range(8)))
    rows = vendor.JudgmentStore(args["output_file"]).records
    assert len(rows) == 8
    assert all(vendor.complete_game(game) for row in rows.values() for game in row["games"])


def test_openai_attempt_retains_raw_completion_without_hidden_retries(vendor, monkeypatch):
    module = sys.modules["utils.completion"]
    raw = {"id": "response-id", "model": "actual-provider-model", "choices": [{
        "finish_reason": "length", "message": {"content": "[[A=B]]", "reasoning": "why"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.012}}
    message = types.SimpleNamespace(content="[[A=B]]", model_dump=lambda **kwargs: raw["choices"][0]["message"])
    response = types.SimpleNamespace(
        model_dump=lambda **kwargs: raw, model=raw["model"], usage=None,
        choices=[types.SimpleNamespace(message=message, finish_reason="length")])
    calls, clients = [], []
    def create(**kwargs):
        calls.append(kwargs)
        return response
    def client(**kwargs):
        clients.append(kwargs)
        return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    import openai
    monkeypatch.setattr(openai, "OpenAI", client)
    result = module.chat_completion_openai("judge", [], 0, 100, judge_attempt=True,
        api_dict={"api_base": "https://fixture.invalid", "api_key": "secret"},
        extra_body={"reasoning": {"effort": "low"}})
    assert len(calls) == 1 and clients[0]["max_retries"] == 0
    assert result["raw_response"] == raw
    assert result["finish_reason"] == "length"
    assert result["model_id"] == "actual-provider-model"
    assert result["reasoning"] == {"reasoning": "why"}
    assert result["reported_cost"] == 0.012
    assert "secret" not in json.dumps(result)


def test_openai_error_body_is_retained_and_scrubbed_once(vendor, monkeypatch):
    module = sys.modules["utils.completion"]
    calls = []
    class ProviderError(Exception):
        status_code = 429
        request_id = "request-id"
        body = {"error": {"message": "Echoed secret"}, "authorization": "secret"}
    def create(**kwargs):
        calls.append(kwargs)
        raise ProviderError("rate limit secret")
    import openai
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))))
    result = module.chat_completion_openai("judge", [], 0, 100, judge_attempt=True,
        api_dict={"api_base": "https://fixture.invalid", "api_key": "secret"})
    assert len(calls) == 1
    assert result["error"]["status_code"] == 429
    assert result["error"]["request_id"] == "request-id"
    assert result["error"]["body"] == {"error": {"message": "Echoed [REDACTED]"}}
    assert "secret" not in json.dumps(result)


def test_cost_counts_failed_attempts_and_deduplicates_retained_history(vendor, tmp_path):
    args = arguments(tmp_path)
    output = completion()
    output["reported_cost"] = 0.012
    register(vendor, [output, output])
    old = vendor.judgment(args)
    row = copy.deepcopy(old)
    row["prior_judgments"] = [old]
    cost = driver._cost([row], "judge")
    assert cost["n_calls"] == 2
    assert cost["reported_usd"] == 0.024
    assert cost["reported_cost_calls"] == 2


def test_vendor_readers_handle_unicode_input(vendor, tmp_path):
    module = sys.modules["utils.completion"]
    question = {"uid": "é", "prompt": "你好 — ∑"}
    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps(question, ensure_ascii=False) + "\n", encoding="utf-8")
    assert module.load_questions(str(path)) == [question]
    assert module.load_model_answers(str(tmp_path))["questions"]["é"] == question
