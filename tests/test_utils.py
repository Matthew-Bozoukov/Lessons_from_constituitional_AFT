# ABOUTME: Unit tests for JSON extraction, parsing and think-trace helpers used across
# ABOUTME: the pipeline. Fast, no network; run with: uv run pytest tests/test_utils.py


import pytest


from src.utils import ParseError, extract_json, resolve_trace, split_think  # noqa: E402


def test_plain_object():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_plain_array():
    assert extract_json('[{"x": 1}, {"y": 2}]') == [{"x": 1}, {"y": 2}]


def test_fenced_json():
    text = 'Here you go:\n```json\n[{"user_message": "hi"}]\n```\nthanks'
    assert extract_json(text) == [{"user_message": "hi"}]


def test_prose_wrapped_object():
    text = 'Sure. {"score": 8, "reason": "ok"} done.'
    assert extract_json(text) == {"score": 8, "reason": "ok"}


def test_braces_inside_strings():
    text = '{"msg": "use { and } carefully", "n": 2}'
    assert extract_json(text) == {"msg": "use { and } carefully", "n": 2}


def test_no_json_raises():
    with pytest.raises(ParseError):
        extract_json("no json here at all")


def test_malformed_raises():
    with pytest.raises(ParseError):
        extract_json('{"a": 1, ')


# --- Think-trace splitting -----------------------------------------------------------


def test_split_think_separates_trace_from_answer():
    think, answer = split_think("<think>weighing options</think>The answer is 4.")
    assert think == "weighing options"
    assert answer == "The answer is 4."


def test_unterminated_think_yields_empty_answer():
    # A cut-off trace must not be handed to the judge as if it were prose.
    think, answer = split_think("<think>still reasoning and then the generation stopped")
    assert answer == ""
    assert think.startswith("still reasoning")


def test_plain_text_has_no_trace():
    assert split_think("Just an answer.") == ("", "Just an answer.")


def test_prefilled_open_tag_close_only_shape():
    # Thinking-mode serving prefills `<think>\n` in the prompt, so vLLM's completion
    # carries the trace with only the close tag. The trace must not leak into the answer.
    think, answer = split_think("weighing options\n</think>\n\nThe answer is 4.")
    assert think == "weighing options"
    assert answer == "The answer is 4."


def test_prefilled_empty_trace_is_empty_think():
    think, answer = split_think("\n</think>\n\nJust the answer.")
    assert think == ""
    assert answer == "Just the answer."


# --- Reasoning-trace shapes across vLLM versions -------------------------------------
# Regression tests for a bug found against the live vLLM 0.26 endpoint: the out-of-band
# trace field is `reasoning` there and `reasoning_content` on 0.8.x. Reading only one of
# them reports every trace as empty and trips the gotcha-2 collapse alarm on a model that
# is reasoning perfectly normally.


def test_resolve_trace_inline_think_tags():
    think, answer = resolve_trace("<think>weighing it up</think>\n\nAnswer: B", None)
    assert think == "weighing it up"
    assert answer == "Answer: B"


def test_resolve_trace_out_of_band_keeps_content_as_the_answer():
    """vLLM 0.26 shape: content holds only the visible answer, trace arrives separately."""
    think, answer = resolve_trace("\n\nB", "The user wants the bone the pituitary sits in.")
    assert think == "The user wants the bone the pituitary sits in."
    assert answer == "B"


def test_resolve_trace_without_any_reasoning():
    assert resolve_trace("B", None) == ("", "B")


def test_resolve_trace_truncated_mid_trace_yields_no_answer():
    """Cut off inside the trace: the answer must stay empty so it scores unparseable.

    Borrowing the trace text as the answer would let a truncated generation score as a
    correct answer whenever a letter happened to appear in the scratchpad.
    """
    think, answer = resolve_trace("", "Let me think about whether it is A or B")
    assert answer == ""
    assert think


def test_resolve_trace_handles_none_content():
    assert resolve_trace(None, None) == ("", "")
