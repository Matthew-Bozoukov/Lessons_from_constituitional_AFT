# ABOUTME: Real canonical-tokenizer checks of Gemma reasoning, history and tool loss boundaries.
# ABOUTME: Verifies labels against inference prefixes and refuses unsupported ablations/truncation.
import copy
import pytest
from src.model_profile import model_profile, render_chat, ModelProfile
from src.train.masking import build_labels
from src.train.gemma4_mask import expected_from_messages, gate


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained("google/gemma-4-31B-it",
            revision="842da3794eaa0b77d5f08bae87a17459d91ff475", local_files_only=True)
    except OSError:
        pytest.skip("Pinned Gemma tokenizer is not cached; tests never download it")


def row():
    return {"tools": [{"type": "function", "function": {"name": "bash", "description": "Run",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}],
        "messages": [
        {"role": "user", "content": "Old question"},
        {"role": "assistant", "content": "OLD_ANSWER", "reasoning_content": "OLD_REASON"},
        {"role": "user", "content": "Now inspect"},
        {"role": "assistant", "content": "", "reasoning_content": "FIRST_REASON",
         "tool_calls": [{"type": "function", "function": {"name": "bash", "arguments": {"command": "ls"}}}]},
        {"role": "tool", "content": "PRIVATE_TOOL_OUTPUT"},
        {"role": "assistant", "content": "FINAL_ANSWER", "reasoning_content": "NEXT_REASON"}]}


def test_actual_generation_prefixes_and_tool_handoff(tok):
    profile = model_profile("gemma4")
    r = row()
    text, expected = expected_from_messages(tok, r, profile)
    out = build_labels(text, tok, 8704, profile)
    decoded = tok.decode([v for v in out["labels"] if v != -100])
    assert decoded == expected
    assert "FIRST_REASON" in decoded and "NEXT_REASON" in decoded and "FINAL_ANSWER" in decoded
    assert "OLD_ANSWER" not in decoded and "PRIVATE_TOOL_OUTPUT" not in decoded
    assert decoded.count("<|channel>thought\n") == 1  # initial generated; continuation forced
    assert "<|tool_response>" in decoded  # generated handoff
    assert "<tool_response|>" not in decoded  # supplied by the executor
    assert gate([text], tok, 8704, profile, True, ["full"], [r])["checked"] == 1


def test_parallel_tool_results_are_context(tok):
    r = row()
    r["messages"][3]["tool_calls"].append(copy.deepcopy(r["messages"][3]["tool_calls"][0]))
    r["messages"].insert(5, {"role": "tool", "content": "SECOND_PRIVATE_RESULT"})
    profile = model_profile("gemma4")
    text, expected = expected_from_messages(tok, r, profile)
    out = build_labels(text, tok, 8704, profile)
    got = tok.decode([v for v in out["labels"] if v != -100])
    assert got == expected and "PRIVATE" not in got
    assert got.count("<|tool_response>") == 1


def test_gate_refuses_truncation_and_unqualified_ablations(tok):
    profile = model_profile("gemma4"); r = row()
    text, _ = expected_from_messages(tok, r, profile)
    with pytest.raises((AssertionError, ValueError)):
        gate([text], tok, 100, profile, True, ["full"], [r])
    with pytest.raises(ValueError, match="full supervision only"):
        build_labels(text, tok, 8704, profile, supervise="cot")
    assert ModelProfile.from_dict(profile.to_dict()).masking_format == "gemma4"
