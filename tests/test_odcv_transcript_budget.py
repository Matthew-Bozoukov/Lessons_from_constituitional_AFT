# ABOUTME: The ODCV transcript token budget: the vendored executor cuts the newest message to
# ABOUTME: fit and stops calling the model; the driver plumbs the limit from odcv.yaml.

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml

from src.eval.misalignment.odcv.odcv_rollout import COMPOSE_TEMPLATE, TOKEN_LIMIT_NOTE

EXEC_DIR = Path("src/eval/misalignment/odcv/third_party/odcv-bench/mission_executor")


def _load(name: str):
    if str(EXEC_DIR) not in sys.path:
        sys.path.insert(0, str(EXEC_DIR))
    spec = importlib.util.spec_from_file_location(name, EXEC_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


budget = _load("transcript_budget")


def _chars_count(msgs):
    """A stand-in for the server tokenizer: 4 chars per token, deterministic."""
    return sum(len(json.dumps(m)) for m in msgs) // 4


def _traj(stdout: str):
    return [{"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Analyse the log."},
            {"role": "assistant", "content": "", "reasoning": "Look first.",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "bash", "arguments": '{"command": "cat log"}'}}]},
            {"role": "tool", "tool_call_id": "c1",
             "content": json.dumps({"stdout": stdout, "stderr": "", "role": "tool", "returncode": 0})}]


def test_under_budget_is_untouched():
    traj = _traj("a,b,c\n" * 20)
    before = json.dumps(traj)
    n, cut = budget.fit_to_limit(traj, 10_000, _chars_count)
    assert not cut and n == _chars_count(traj) and json.dumps(traj) == before


def test_overflowing_tool_result_is_cut_to_fit_and_stays_json():
    traj = _traj("2023-10-01,EMP001,PAT1001,view,UNAUTH\n" * 5000)   # ~190k chars
    limit = 2_000
    n, cut = budget.fit_to_limit(traj, limit, _chars_count)
    assert cut and n <= limit
    out = json.loads(traj[-1]["content"])                # still the harness's JSON shape
    assert out["returncode"] == 0 and out["stdout"].endswith(budget.MARKER)
    assert out["stdout"].startswith("2023-10-01,EMP001")  # the head survives
    assert [m["role"] for m in traj[:-1]] == ["system", "user", "assistant"]
    assert traj[2]["reasoning"] == "Look first.", "earlier messages are never touched"


def test_the_dump_is_cut_even_when_a_later_small_result_follows_it():
    # One turn, two tool calls: the dump lands first, a tiny result after it. The walk
    # back must reach the dump; the tiny result is emptied on the way, the dump is cut.
    traj = _traj("2023-10-01,EMP001,PAT1001,view,UNAUTH\n" * 5000)
    traj[2]["tool_calls"].append({"id": "c2", "type": "function",
                                  "function": {"name": "bash", "arguments": '{"command": "wc -l log"}'}})
    traj.append({"role": "tool", "tool_call_id": "c2",
                 "content": json.dumps({"stdout": "5000 log\n", "stderr": "", "role": "tool", "returncode": 0})})
    n, cut = budget.fit_to_limit(traj, 2_000, _chars_count)
    assert cut and n <= 2_000
    assert budget.MARKER in traj[-2]["content"] and traj[-2]["content"].startswith('{"stdout": "2023-10-01')
    assert traj[2]["reasoning"] == "Look first.", "model output is never touched"


def test_overflow_from_a_message_too_small_to_absorb_it_still_stops():
    # The model's own long reply overflowed and the newest message is a short nudge: it
    # cannot absorb the overshoot, so it becomes the marker and the caller stops anyway.
    traj = _traj("x" * 400) + [{"role": "user", "content": "please continue"}]
    n, cut = budget.fit_to_limit(traj, 50, _chars_count)
    assert cut and traj[-1]["content"] == budget.MARKER and n > 50


def test_estimate_is_conservative_for_dense_text():
    msgs = [{"role": "tool", "content": "2023-10-01,EMP001,PAT1001,view,UNAUTH\n" * 100}]
    assert budget.estimate_count(msgs) > len(msgs[0]["content"]) // 4


def test_executor_falls_back_to_the_estimate_when_tokenize_is_unreachable(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")   # nothing listens
    monkeypatch.setenv("CONTEXT_WINDOW_TOKENS", "28000")
    agent = _load("agent_main")
    assert agent.CONTEXT_WINDOW_TOKENS == 28000
    count = agent._count_tokens(tools=[])
    msgs = [{"role": "user", "content": "hello " * 100}]
    assert count(msgs) == budget.estimate_count(msgs)
    assert count(msgs) == budget.estimate_count(msgs), "fallback sticks after the first failure"


def test_the_served_window_is_the_one_value_and_reaches_the_container():
    cfg = yaml.safe_load(Path("configs/eval/odcv.yaml").read_text())
    assert "transcript_token_limit" not in cfg, "one value: the served window is the budget"
    assert cfg["serving"]["context_window"] == 28000
    assert "CONTEXT_WINDOW_TOKENS=${CONTEXT_WINDOW_TOKENS}" in COMPOSE_TEMPLATE
    assert "COPY transcript_budget.py" in (EXEC_DIR / "Dockerfile").read_text()
    assert TOKEN_LIMIT_NOTE == budget.NOTE_PREFIX, "the driver greps the note the executor writes"


def test_rejection_is_recognised_and_trims_this_steps_results_to_the_window():
    class Fault(Exception):
        pass

    vllm = Fault("Error code: 400 - {'error': {'message': \"This model's maximum context "
                 "length is 28000 tokens. However, you requested 51000 tokens\"}}")
    assert budget.is_context_length_error(vllm)
    assert budget.is_context_length_error(Fault("code: context_length_exceeded"))
    assert not budget.is_context_length_error(Fault("Connection reset by peer"))

    traj = _traj("2023-10-01,EMP001,PAT1001,view,UNAUTH\n" * 5000)
    snap = list(traj)                          # the archive shares the message dicts
    note = budget.trim_after_rejection(traj, snap, window=2_000, count=_chars_count, cycle=3)
    assert _chars_count(traj) <= 2_000
    assert budget.MARKER in snap[3]["content"], "the cut shows in the archived record"
    assert snap[-1] is note and note["content"].startswith(budget.NOTE_PREFIX)
    assert "truncated to fit" in note["content"] and "cycle 3" in note["content"]


def test_a_reply_cut_off_by_the_window_gets_its_own_note():
    note = budget.length_stop_note(28000, 7)
    assert note["role"] == "system" and note["content"].startswith(budget.NOTE_PREFIX)
    assert "finish_reason=length" in note["content"] and "cycle 7" in note["content"]
