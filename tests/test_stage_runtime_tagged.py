# ABOUTME: Offline tests for the tagged-block parser behind every `llm_tagged` stage.
# ABOUTME: Run: uv run pytest tests/test_stage_runtime_tagged.py -q

import pytest

from src.data.synth.stage_runtime import _parse_tagged

KEYS = ("reasoning", "response")


def test_closed_blocks_parse_and_strip():
    text = "<reasoning>\n why \n</reasoning>\n<response>\n the reply \n</response>\n"
    assert _parse_tagged(text, KEYS) == {"reasoning": "why", "response": "the reply"}


def test_unclosed_final_tag_is_accepted():
    """Gemini 3.7 Flash ends completed replies without the last closing tag."""
    text = "<reasoning>why</reasoning>\n<response>\nthe reply\n\nmore of it"
    assert _parse_tagged(text, KEYS) == {"reasoning": "why",
                                         "response": "the reply\n\nmore of it"}


def test_unclosed_non_final_tag_is_rejected():
    text = "<reasoning>why <response>the reply</response>"
    with pytest.raises(ValueError, match="missing <reasoning> block"):
        _parse_tagged(text, KEYS)


def test_missing_final_tag_is_still_rejected():
    with pytest.raises(ValueError, match="missing <response> block"):
        _parse_tagged("<reasoning>why</reasoning> and nothing else", KEYS)


def test_three_keys_only_the_last_may_be_open():
    keys = ("reasoning", "response", "changes")
    text = "<reasoning>a</reasoning><response>b</response><changes>c"
    assert _parse_tagged(text, keys)["changes"] == "c"
    with pytest.raises(ValueError, match="missing <response> block"):
        _parse_tagged("<reasoning>a</reasoning><response>b<changes>c</changes>", keys)
