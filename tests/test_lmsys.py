# ABOUTME: Offline tests for the LMSYS chat-quality eval: judge messages and outcome
# ABOUTME: mapping, parity position-randomization, subset selection/hashing, reference refusal.

from __future__ import annotations

import json

import pytest

from src.eval.capabilities.lmsys.runner import (
    first_user_turn,
    judge_messages,
    judge_outcome,
    pair_for_judging,
    select_prompts,
    subset_hash,
    summarize,
)


def row(cid: str, text: str, language: str = "English", redacted: bool = False,
        conversation: list[dict] | None = None) -> dict:
    """One synthetic lmsys-chat-1m row (single user turn unless a conversation is given)."""
    return {
        "conversation_id": cid,
        "language": language,
        "redacted": redacted,
        "conversation": conversation or [{"role": "user", "content": text}],
    }


# --- Judge messages and outcome mapping ----------------------------------------------


def test_judge_messages_carries_prompt_and_answers_in_position_order():
    messages = judge_messages("the prompt", "answer-a", "answer-b")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "impartial judge" in messages[0]["content"]
    user = messages[1]["content"]
    assert "the prompt" in user
    # A's slot precedes B's, and each answer sits under its own label.
    assert user.index("[Response A]\nanswer-a") < user.index("[Response B]\nanswer-b")


def test_pair_for_judging_randomizes_position_by_id_parity():
    # Even id: target answers as A. Odd id: target answers as B.
    assert pair_for_judging(0, "tgt", "ref") == ("tgt", "ref", True)
    assert pair_for_judging(1, "tgt", "ref") == ("ref", "tgt", False)
    positions = [pair_for_judging(i, "t", "r")[2] for i in range(10)]
    assert positions.count(True) == positions.count(False) == 5


@pytest.mark.parametrize(
    "winner,target_is_a,expected",
    [
        ("A", True, "target"),
        ("B", True, "reference"),
        ("A", False, "reference"),
        ("B", False, "target"),
        ("tie", True, "tie"),
        ("tie", False, "tie"),
    ],
)
def test_judge_outcome_maps_verdicts_back_through_the_position_assignment(
        winner, target_is_a, expected):
    assert judge_outcome(winner, target_is_a) == expected


def test_judge_outcome_rejects_a_garbage_verdict_instead_of_counting_it():
    with pytest.raises(ValueError, match="unrecognized winner"):
        judge_outcome("C", True)
    with pytest.raises(ValueError, match="unrecognized winner"):
        judge_outcome("", False)


# --- Prompt-subset selection ----------------------------------------------------------


def test_select_prompts_is_deterministic_and_applies_every_filter():
    rows = [
        row("keep0", "What is the capital of France, and why?"),
        row("lang", "Une question en français, assez longue.", language="French"),
        row("redacted", "Tell me about NAME_1 and their long career.", redacted=True),
        row("short", "hi"),
        row("long", "x" * 5000),
        row("noturn", "", conversation=[{"role": "assistant", "content": "hello there"}]),
        row("keep1", "Explain quantum entanglement to a ten year old."),
        row("keep2", "Write a limerick about a stubborn compiler."),
    ]
    kwargs = dict(n=3, language="English", min_chars=20, max_chars=2000)
    selected = select_prompts(list(rows), **kwargs)
    assert [p["source_id"] for p in selected] == ["keep0", "keep1", "keep2"]
    assert [p["id"] for p in selected] == [0, 1, 2]  # dense ids — parity stays balanced
    # Pure function of the (already seed-shuffled) stream: same input, same subset.
    assert select_prompts(list(rows), **kwargs) == selected


def test_select_prompts_takes_the_first_user_turn_not_the_first_message():
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What should I plant in a shady garden bed?"},
        {"role": "assistant", "content": "Ferns."},
        {"role": "user", "content": "And in full sun?"},
    ]
    [p] = select_prompts([row("c", "", conversation=conversation)],
                         n=1, language="English", min_chars=1, max_chars=2000)
    assert p["prompt"] == "What should I plant in a shady garden bed?"
    assert first_user_turn([{"role": "assistant", "content": "no user turn"}]) is None


def test_select_prompts_fails_fast_when_the_stream_runs_dry():
    with pytest.raises(ValueError, match="1/5 usable"):
        select_prompts([row("only", "A single usable prompt, sadly.")],
                       n=5, language="English", min_chars=1, max_chars=2000)


def test_subset_hash_is_stable_and_detects_changed_prompts():
    prompts = [{"id": 0, "prompt": "alpha"}, {"id": 1, "prompt": "beta"}]
    assert subset_hash(prompts) == subset_hash(list(reversed(prompts)))  # order-free
    changed_text = [{"id": 0, "prompt": "alpha"}, {"id": 1, "prompt": "BETA"}]
    changed_ids = [{"id": 0, "prompt": "alpha"}, {"id": 2, "prompt": "beta"}]
    assert subset_hash(changed_text) != subset_hash(prompts)
    assert subset_hash(changed_ids) != subset_hash(prompts)


# --- The cached reference flow (local cache backend, stubbed clients) -----------------


class _Choice:
    def __init__(self, text):
        self.message = type("M", (), {"content": text, "reasoning_content": None,
                                      "reasoning": None})()
        self.finish_reason = "stop"


class _StubOpenAI:
    """Offline endpoint stand-in: echoes a per-model canned reply."""

    reply = "<think>hm</think>ok"

    def __init__(self, **kwargs):
        stub = self

        class _Completions:
            def create(self, **kw):
                return type("R", (), {"choices": [_Choice(stub.reply)]})()

        self.chat = type("C", (), {"completions": _Completions()})()


class _StubJudge:
    """OpenRouterClient stand-in: reference (position-mapped B or A) always ties."""

    def chat(self, model, messages, **kw):
        return type("R", (), {"content": '{"winner": "tie", "reason": "same"}'})()


def _target(hf_path, model_key, mode="think", boots: list | None = None):
    class _Spec:
        pass

    spec = _Spec()
    spec.hf_path, spec.model_key, spec.mode = hf_path, model_key, mode

    class _T:
        def __init__(self):
            self.spec = spec
            self.model_name = model_key
            self.api_key = "EMPTY"

        @property
        def base_url(self):
            if boots is None:
                raise AssertionError("cached arm must not touch the endpoint")
            boots.append(hf_path)
            return "http://localhost:0/v1"

    return _T()


def _cfg(tmp_path):
    import json as _json

    prompts = [{"id": i, "prompt": f"question {i}"} for i in range(4)]
    prompts_file = tmp_path / "prompts.json"
    prompts_file.write_text(_json.dumps(prompts))
    from omegaconf import OmegaConf

    return OmegaConf.create({
        "prompts_path": str(prompts_file),
        "cache": {"repo": str(tmp_path / "cache"), "mirror": str(tmp_path / "mirror"),
                  "refresh": False},
        "generation": {"api_key": "EMPTY", "temperature": 0.7, "top_p": 0.95,
                       "max_tokens": 64, "parallel": 2, "request_timeout": 5,
                       "max_retries": 0},
        "judge": {"model": "stub/judge", "temperature": 0.0, "max_tokens": 8,
                  "parallel": 1},
    })


def test_reference_then_target_flow_through_the_cache(tmp_path, monkeypatch):
    from src.eval.capabilities.lmsys import runner

    monkeypatch.setattr(runner, "OpenAI", _StubOpenAI)
    monkeypatch.setattr(runner, "OpenRouterClient", _StubJudge)
    monkeypatch.setattr("src.endpoints.vllm_server.resolve_target",
                        lambda hf: _target(hf, "ref_model").spec)
    cfg = _cfg(tmp_path)

    # Arm 1: the reference (as run_eval orders it) — generates, pushes, no judging.
    boots: list = []
    ref_out = tmp_path / "run_ref"
    ref_out.mkdir()
    summary = runner.run(_target("org/ref", "ref_model", boots=boots), cfg, ref_out,
                         reference="org/ref")
    assert summary["reference_arm"] is True and boots == ["org/ref"]
    assert "target_wins" not in summary

    # Arm 2: a target — generates its own answers, judges against the cached reference.
    t_out = tmp_path / "run_t"
    t_out.mkdir()
    summary = runner.run(_target("org/ft", "ft_model", boots=boots), cfg, t_out,
                         reference="org/ref")
    assert summary["ties"] == 4 and summary["reference"] == "org/ref"

    # Arm 2 again: fully cached — MUST complete without touching any endpoint
    # (boots=None makes base_url raise), proving the zero-vLLM cache-hit path.
    t2_out = tmp_path / "run_t2"
    t2_out.mkdir()
    summary = runner.run(_target("org/ft", "ft_model", boots=None), cfg, t2_out,
                         reference="org/ref")
    assert summary["ties"] == 4


def test_cross_mode_reference_is_refused(tmp_path, monkeypatch):
    from src.eval.capabilities.lmsys import runner

    monkeypatch.setattr(runner, "OpenAI", _StubOpenAI)
    monkeypatch.setattr("src.endpoints.vllm_server.resolve_target",
                        lambda hf: _target(hf, "ref_model", mode="nothink").spec)
    cfg = _cfg(tmp_path)
    out = tmp_path / "run"
    out.mkdir()
    with pytest.raises(RuntimeError, match="cross-mode"):
        runner.run(_target("org/ft", "ft_model", mode="think", boots=[]), cfg, out,
                   reference="org/ref")


# --- Summary arithmetic ---------------------------------------------------------------


def test_summarize_win_rates():
    summary = summarize(["target"] * 3 + ["reference"] + ["tie"] * 2)
    assert summary["n"] == 6
    assert (summary["target_wins"], summary["reference_wins"], summary["ties"]) == (3, 1, 2)
    assert summary["winrate_excl_ties_pct"] == 75.0                  # 3 of 4 decisive
    assert summary["winrate_ties_half_pct"] == pytest.approx(66.7)   # (3 + 1) / 6


def test_summarize_with_no_decisive_judgments_reports_no_excl_ties_rate():
    summary = summarize(["tie"] * 4)
    assert summary["winrate_excl_ties_pct"] is None
    assert summary["winrate_ties_half_pct"] == 50.0
