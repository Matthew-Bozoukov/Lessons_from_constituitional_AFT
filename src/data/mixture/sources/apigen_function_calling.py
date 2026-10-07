# ABOUTME: APIGen function calling (smoltalk's apigen-80k subset) — tool-use conversations,
# ABOUTME: parsed out of their xLAM prompt text into native `tools` + `tool_calls` fields.

"""The raw rows write tool use as TEXT in one family's syntax (xLAM's): the system turn
is a fixed wrapper around a `<tools>[...]</tools>` JSON blob plus a tail teaching the
`<tool_call>[...]</tool_call>` output format, and the assistant turn is that text. Trained
as-is, every model would learn xLAM's syntax instead of its own. So this adapter parses
the text into the interchange fields — schemas into `tools`, calls into `tool_calls` —
and drops the format tail; each family's chat template then renders them in its native
syntax at train time (`render_chat`, src/model_profile.py). Nothing here names a model.

The schemas arrive in two dialects (about half each): OpenAI function schemas, kept as
they are, and xLAM's flat `{"arg": {"type": "List[int]", ...}}`, converted to JSON-schema
by `_xlam_schema`. A row whose text does not parse, or that calls a tool whose types have
no JSON-schema form, is dropped, never repaired (an uncalled such tool is left off its menu).
"""

from __future__ import annotations

import json
import re

from src.data.mixture.sources.base import SourceAdapter, clean_messages, clean_tools

# The wrapper every apigen-80k system turn uses (one head, one tail over all rows). The
# head's instructions are kept (they are what the refusal rows answer to); the tools line,
# the blob and the xLAM output-format tail are the syntax this adapter replaces.
_SYSTEM = re.compile(
    r"(?P<head>.*?)\s*You have access to the following tools:\s*"
    r"<tools>(?P<tools>.*)</tools>.*",
    re.S,
)
_CALL = re.compile(r"\s*<tool_call>(?P<calls>.*)</tool_call>\s*", re.S)

# xLAM's Python-style type names -> JSON-schema types. Containers become `array`/`object`
# (with `items` when the element type converts); anything else (Callable, a bare Union)
# has no JSON-schema form, so its row is dropped.
_SCALARS = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "dict": "object",
}
_ARRAYS = {"list", "tuple", "set"}


def _xlam_type(name: str) -> dict | None:
    """JSON schema for one xLAM type string (`List[int]`), or None when it has none."""
    m = re.fullmatch(r"\s*(\w+)\s*(?:\[(.*)\])?\s*", name)
    if not m:
        return None
    base, inner = m.group(1).lower(), m.group(2)
    if base in _SCALARS:
        return {"type": _SCALARS[base]}
    if base in _ARRAYS:
        items = _xlam_type(inner) if base == "list" and inner else None
        return {"type": "array", **({"items": items} if items else {})}
    return None


def _xlam_schema(tool: dict) -> dict | None:
    """An xLAM tool (`{"name", "description", "parameters": {arg: {type, ...}}}`) as an
    OpenAI function schema; `", optional"` on a type is what leaves an arg off `required`."""
    props, required = {}, []
    for arg, spec in (tool.get("parameters") or {}).items():
        if not isinstance(spec, dict) or not isinstance(spec.get("type"), str):
            return None
        type_name, optional = re.subn(r",\s*optional\s*$", "", spec["type"])
        schema = _xlam_type(type_name)
        if schema is None:
            return None
        props[arg] = {**schema, **{k: v for k, v in spec.items() if k != "type"}}
        if not optional:
            required.append(arg)
    fn = {
        "name": tool.get("name"),
        "description": tool.get("description", ""),
        "parameters": {"type": "object", "properties": props, "required": required},
    }
    return {"type": "function", "function": fn}


def _parse_system(row: dict) -> tuple[str, list] | None:
    msgs = row.get("messages") or []
    if not msgs or msgs[0].get("role") != "system":
        return None
    m = _SYSTEM.fullmatch(msgs[0].get("content") or "")
    if not m:
        return None
    try:
        return m.group("head").strip(), json.loads(m.group("tools"))
    except ValueError:
        return None


def _calls(row: dict) -> list[list[dict]] | None:
    """Each assistant turn's parsed call list (None for a text turn), or None if unparseable."""
    out = []
    for m in row["messages"][1:]:
        call = _CALL.fullmatch(m.get("content") or "") if m.get("role") == "assistant" else None
        if call is None:
            out.append(None)
            continue
        try:
            calls = json.loads(call.group("calls"))
        except ValueError:
            return None
        if not isinstance(calls, list) or not calls:
            return None
        out.append(calls)
    return out


def _convert(row: dict) -> tuple[str, list[dict], list] | None:
    """(system head, converted schemas, per-turn calls), or None when the row is unusable.

    A tool whose schema has no JSON-schema form (a `Callable` argument) is left off the
    menu when the row never calls it — it is one distractor among several — and makes the
    row unusable when the row does call it, or when it was the only tool offered.
    """
    parsed, calls = _parse_system(row), None
    if parsed is not None:
        calls = _calls(row)
    if parsed is None or calls is None:
        return None
    head, raw_tools = parsed
    called = {c.get("name") for turn in calls if turn for c in turn}
    tools = []
    for t in raw_tools:
        schema = t if "function" in t else _xlam_schema(t)
        if schema is None:
            if t.get("name") in called:
                return None
            continue
        tools.append(schema)
    if raw_tools and not tools:
        return None
    if tools and clean_tools(tools) is None:
        return None
    return head, tools, calls


def to_tools(row: dict) -> list[dict] | None:
    """The row's schemas in the interchange shape; None for the few rows offered no tools."""
    converted = _convert(row)
    return (converted[1] or None) if converted else None


def to_messages(row: dict) -> list[dict] | None:
    """System head + user + the assistant turn as native `tool_calls` (or its refusal text)."""
    converted = _convert(row)
    if converted is None:
        return None
    head, _, calls = converted
    msgs = [{"role": "system", "content": head}]
    for m, turn in zip(row["messages"][1:], calls):
        msgs.append(m if turn is None else {
            "role": "assistant", "content": "",
            "tool_calls": [{"type": "function", "function": c} for c in turn]})
    return clean_messages(msgs)


ADAPTER = SourceAdapter(
    name="apigen_function_calling",
    repo="HuggingFaceTB/smoltalk",
    hf_config="apigen-80k",
    to_messages=to_messages,
    to_tools=to_tools,
)
