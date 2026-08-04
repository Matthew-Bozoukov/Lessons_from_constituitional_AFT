# ABOUTME: Offline tests for the LMSYS chat-quality eval: judge messages and outcome
# ABOUTME: mapping, parity position-randomization, subset selection/hashing, reference refusal.

from __future__ import annotations

import json

import pytest

from src.eval.capabilities.lmsys_eval import (
    check_reference,
    first_user_turn,
    judge_messages,
    judge_outcome,
    load_reference,
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


# --- Reference loading and refusal ----------------------------------------------------


def write_reference(tmp_path, subset="abc123", mode="think"):
    (tmp_path / "answers.jsonl").write_text(
        json.dumps({"id": 0, "prompt": "p", "think": "t", "answer": "a",
                    "finish_reason": "stop"}) + "\n")
    (tmp_path / "answers_meta.json").write_text(
        json.dumps({"target": "org/ref-arm", "mode": mode, "subset_hash": subset}))


def test_load_reference_accepts_a_run_dir_or_the_answers_file_itself(tmp_path):
    write_reference(tmp_path)
    for ref in (tmp_path, tmp_path / "answers.jsonl"):
        records, meta = load_reference(str(ref))
        assert records[0]["answer"] == "a"
        assert meta["mode"] == "think" and meta["subset_hash"] == "abc123"


def test_load_reference_requires_the_sidecar(tmp_path):
    write_reference(tmp_path)
    (tmp_path / "answers_meta.json").unlink()
    with pytest.raises(FileNotFoundError, match="answers_meta.json"):
        load_reference(str(tmp_path))
    with pytest.raises(FileNotFoundError, match="answers not found"):
        load_reference(str(tmp_path / "nope"))


def test_check_reference_refuses_a_different_prompt_subset():
    with pytest.raises(RuntimeError, match="different prompt set"):
        check_reference({"subset_hash": "aaaa", "mode": "think"}, "bbbb", "think")


def test_check_reference_refuses_cross_mode_pairing():
    with pytest.raises(RuntimeError, match="cross-mode"):
        check_reference({"subset_hash": "aaaa", "mode": "nothink"}, "aaaa", "think")
    # Matching hash and mode passes silently.
    check_reference({"subset_hash": "aaaa", "mode": "think"}, "aaaa", "think")


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


# --- Bootstrap (first arm of a ladder) ------------------------------------------------


class _Choice:
    def __init__(self, text):
        self.message = type("M", (), {"content": text, "reasoning_content": None,
                                      "reasoning": None})()
        self.finish_reason = "stop"


class _StubOpenAI:
    """Offline stand-in for the served endpoint: echoes a canned reply per prompt."""

    def __init__(self, **kwargs):
        outer = self

        class _Completions:
            def create(self, **kw):
                return type("R", (), {"choices": [_Choice("<think>hm</think>ok")]})()

        self.chat = type("C", (), {"completions": _Completions()})()


def test_bootstrap_writes_the_answers_artifact_and_skips_judging(tmp_path, monkeypatch):
    import json

    from omegaconf import OmegaConf

    from src.eval.capabilities import lmsys_eval

    prompts = [{"id": i, "prompt": f"question {i}"} for i in range(4)]
    prompts_file = tmp_path / "prompts.json"
    prompts_file.write_text(json.dumps(prompts))

    cfg = OmegaConf.create({
        "reference": "bootstrap",
        "prompts_path": str(prompts_file),
        "generation": {"api_key": "EMPTY", "temperature": 0.7, "top_p": 0.95,
                       "max_tokens": 64, "parallel": 2, "request_timeout": 5,
                       "max_retries": 0},
        "judge": {"model": "never/called", "temperature": 0.0, "max_tokens": 8,
                  "parallel": 1},
    })
    target = type("T", (), {"base_url": "http://localhost:0/v1", "model_name": "stub",
                            "spec": type("S", (), {"mode": "think", "hf_path": "org/m",
                                                   "model_key": "m"})()})()
    monkeypatch.setattr(lmsys_eval, "OpenAI", _StubOpenAI)

    out_dir = tmp_path / "run"
    out_dir.mkdir()
    summary = lmsys_eval.run(target, cfg, out_dir)

    assert summary["bootstrap"] is True
    assert summary["empty_think_rate"] == 0.0
    assert "target_wins" not in summary  # no judging happened
    meta = json.loads((out_dir / "answers_meta.json").read_text())
    assert meta["mode"] == "think" and meta["subset_hash"] == summary["subset_hash"]
    answers = [json.loads(l) for l in (out_dir / "answers.jsonl").read_text().splitlines()]
    assert [a["answer"] for a in answers] == ["ok"] * 4
    assert all(a["think"] == "hm" for a in answers)
