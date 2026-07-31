# ABOUTME: Robust JSON extraction from model text plus judge-verdict parsing.
# ABOUTME: Judges must fail loudly on malformed output rather than scoring a guess.

from __future__ import annotations

import json
from typing import Any


class ParseError(ValueError):
    """Raised when model output cannot be parsed as the expected structure."""


def extract_json(text: str) -> Any:
    """Extract the first top-level JSON array or object from model text.

    Handles responses wrapped in prose or ```json fences.

    Args:
        text: Raw model output.

    Returns:
        The parsed JSON value.

    Raises:
        ParseError: If no valid JSON array/object can be parsed.
    """
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start = min((i for i in (stripped.find("["), stripped.find("{")) if i != -1), default=-1)
    if start == -1:
        raise ParseError(f"No JSON found in response: {text[:200]!r}")

    open_ch = stripped[start]
    close_ch = "]" if open_ch == "[" else "}"
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(stripped)):
        ch = stripped[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                candidate = stripped[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as e:
                    raise ParseError(f"Malformed JSON: {e}: {candidate[:200]!r}") from e
    raise ParseError(f"Unterminated JSON in response: {text[:200]!r}")


def split_thinking(text: str, tag: str = "think") -> tuple[str, str]:
    """Split a `<think>...</think>` prefix off a completion.

    Qwen3-style models emit reasoning inline when served without a separate
    reasoning field. The suite scores the *answer*, and reports think length as a
    side-effect metric, so the two have to be separated before judging.

    Args:
        text: Raw completion text.
        tag: Reasoning tag name.

    Returns:
        Tuple of (thinking, answer). Thinking is "" when no tag is present.
    """
    open_tag, close_tag = f"<{tag}>", f"</{tag}>"
    end = text.find(close_tag)
    if end == -1:
        return "", text.strip()

    start = text.find(open_tag)
    if start == -1 or start > end:
        # Many chat templates PRE-FILL the opening tag in the prompt, so the completion
        # contains only the closing one. Requiring both tags here silently returned
        # reasoning+answer as the answer, which would hand a judge one arm's private
        # reasoning and not the other's - manufacturing a difference between recipes that
        # is really a difference in serving stack.
        return text[:end].strip(), text[end + len(close_tag) :].strip()

    thinking = text[start + len(open_tag) : end].strip()
    answer = (text[:start] + text[end + len(close_tag) :]).strip()
    return thinking, answer


def parse_verdict(text: str, fields: list[str]) -> dict[str, Any]:
    """Parse a judge response and check every required field is present.

    Args:
        text: Raw judge output.
        fields: Required top-level keys.

    Returns:
        The parsed verdict mapping.

    Raises:
        ParseError: If the payload is not an object or a required field is absent.
    """
    parsed = extract_json(text)
    if not isinstance(parsed, dict):
        raise ParseError(f"Judge returned {type(parsed).__name__}, expected object")
    missing = [f for f in fields if f not in parsed]
    if missing:
        raise ParseError(f"Judge verdict missing fields {missing}: {sorted(parsed)}")
    return parsed


def coerce_score(value: Any, scale_max: float) -> float:
    """Coerce a rubric score to a float and range-check it.

    Args:
        value: Raw score from the judge (int, float, or numeric string).
        scale_max: Top of the rubric scale.

    Returns:
        The score as a float.

    Raises:
        ParseError: If the value is not numeric or falls outside [0, scale_max].
    """
    try:
        score = float(value)
    except (TypeError, ValueError) as e:
        raise ParseError(f"Non-numeric judge score {value!r}") from e
    if not 0.0 <= score <= scale_max:
        raise ParseError(f"Judge score {score} outside [0, {scale_max}]")
    return score
