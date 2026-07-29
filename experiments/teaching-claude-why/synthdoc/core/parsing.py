# ABOUTME: Tolerant JSON extraction and Turn parsing for model output, plus the
# ABOUTME: human-readable document rendering shown to revisers and raters.

from __future__ import annotations

import json
import re
from typing import Any

from .types import ROLES, Turn


class ParseError(ValueError):
    """Raised when model output cannot be read as the expected structure."""


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> Any:
    """Extract the first top-level JSON object or array from model text.

    Handles bare JSON, fenced JSON, and JSON wrapped in prose.

    Args:
        text: Raw model output.

    Returns:
        The parsed JSON value.

    Raises:
        ParseError: If nothing parseable is found.
    """
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    fenced = _FENCE.search(stripped)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    candidates = [i for i in (stripped.find("{"), stripped.find("[")) if i != -1]
    if not candidates:
        raise ParseError(f"No JSON found in output: {stripped[:200]!r}")
    start = min(candidates)
    open_ch = stripped[start]
    close_ch = "}" if open_ch == "{" else "]"

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
                block = stripped[start : i + 1]
                try:
                    return json.loads(block)
                except json.JSONDecodeError as e:
                    raise ParseError(f"Malformed JSON: {e}: {block[:200]!r}") from e
    raise ParseError(f"Unterminated JSON in output: {stripped[:200]!r}")


def parse_turns(text: str) -> list[Turn]:
    """Parse a model response into Turns.

    Accepts either {"turns": [...]} or a bare list of turn objects.

    Args:
        text: Raw model output.

    Returns:
        The parsed turns.

    Raises:
        ParseError: If the structure is wrong or no usable turn survives.
    """
    payload = extract_json(text)
    if isinstance(payload, dict):
        items = payload.get("turns")
        if items is None:
            raise ParseError(f"JSON object has no 'turns' key: {sorted(payload)[:8]}")
    elif isinstance(payload, list):
        items = payload
    else:
        raise ParseError(f"Expected object or array, got {type(payload).__name__}")

    if not isinstance(items, list):
        raise ParseError(f"'turns' is {type(items).__name__}, expected a list")

    turns: list[Turn] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ParseError(f"turns[{i}] is {type(item).__name__}, expected an object")
        role = str(item.get("role", "")).strip().lower()
        if role not in ROLES:
            raise ParseError(f"turns[{i}] has role {role!r}; expected one of {ROLES}")
        tool_calls = item.get("tool_calls") or ""
        if not isinstance(tool_calls, str):
            tool_calls = json.dumps(tool_calls)
        turns.append(
            Turn(
                role=role,
                content=str(item.get("content") or "").strip(),
                thinking=str(item.get("thinking") or "").strip(),
                tool_calls=tool_calls,
            )
        )
    turns = [t for t in turns if t.content or t.thinking or t.tool_calls]
    if not turns:
        raise ParseError("All parsed turns were empty")
    return turns


def parse_scores(text: str, criteria: list[str], scale: int) -> tuple[dict[str, float], float, str]:
    """Parse an autorater response.

    Args:
        text: Raw rater output.
        criteria: Expected criterion names.
        scale: Maximum valid score.

    Returns:
        Tuple of (per-criterion scores, overall score, justification).

    Raises:
        ParseError: If the payload is not an object or has no usable scores.
    """
    payload = extract_json(text)
    if not isinstance(payload, dict):
        raise ParseError(f"Rater returned {type(payload).__name__}, expected an object")
    raw = payload.get("scores")
    if not isinstance(raw, dict):
        raw = {k: v for k, v in payload.items() if k in criteria}
    scores: dict[str, float] = {}
    for name in criteria:
        if name in raw:
            try:
                scores[name] = _clamp(float(raw[name]), scale)
            except (TypeError, ValueError):
                continue
    if not scores:
        raise ParseError(f"No criteria from {criteria} present in rater output")
    overall = payload.get("overall")
    try:
        overall_f = _clamp(float(overall), scale)
    except (TypeError, ValueError):
        overall_f = sum(scores.values()) / len(scores)
    return scores, overall_f, str(payload.get("justification") or "")[:500]


def _clamp(value: float, scale: int) -> float:
    """Clamp a score into [1, scale]."""
    return max(1.0, min(float(scale), value))


def render_document(turns: list[Turn]) -> str:
    """Render turns as readable text for revision and rating prompts.

    Args:
        turns: The document's turns.

    Returns:
        A plain-text rendering with explicit role and thinking markers.
    """
    parts: list[str] = []
    for t in turns:
        head = f"### {t.role.upper()}"
        body: list[str] = []
        if t.thinking:
            body.append(f"[thinking]\n{t.thinking}")
        if t.tool_calls:
            body.append(f"[tool_calls]\n{t.tool_calls}")
        if t.content:
            body.append(t.content)
        parts.append(head + "\n" + "\n\n".join(body))
    return "\n\n".join(parts)
