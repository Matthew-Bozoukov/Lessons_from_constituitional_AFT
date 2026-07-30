# ABOUTME: Unit tests for JSON extraction and parsing helpers used across the pipeline.
# ABOUTME: Fast, no network; run with: uv run pytest tests/test_utils.py


import pytest


from src.utils import ParseError, extract_json  # noqa: E402


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
