# ABOUTME: The SourceAdapter contract and the shared message normaliser every source
# ABOUTME: adapter builds on; the interchange format is documented in this package's README.

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

_ROLES = ("system", "user", "assistant", "tool")
_CARRIED = ("role", "content", "reasoning_content", "tool_calls")


def clean_tool_calls(calls) -> list[dict] | None:
    """Normalise an assistant turn's tool calls to the one interchange shape, or None.

    The shape is `{"type": "function", "function": {"name": str, "arguments": mapping}}`
    — the OpenAI form with ONE change: arguments are stored as a mapping, never as the
    wire form's JSON string, because that is what chat templates consume and what makes
    the stored row model-agnostic. A JSON-string argument (what an API client hands
    back, and what a rollout record holds) is parsed here; anything else malformed
    makes the row unusable, the same policy as `clean_messages`. Only the call itself
    is kept — a wire `id` is transport, not semantics.
    """
    if not isinstance(calls, list) or not calls:
        return None
    out = []
    for c in calls:
        fn = c.get("function") if isinstance(c, dict) else None
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            return None
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                return None
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return None
        out.append({"type": "function",
                    "function": {"name": fn["name"], "arguments": args}})
    return out


def clean_messages(messages) -> list[dict] | None:
    """Validate and strip a message list to the interchange fields, or None if unusable.

    Unusable rows are dropped rather than repaired: a silently mangled example is worse
    than a slightly smaller sample. A conversation must end with an assistant turn and
    contain at least one user turn; every turn needs a known role and string content
    (empty content is allowed only on assistant turns that carry tool_calls, and those
    are normalised by `clean_tool_calls` — a malformed call makes the row unusable).
    """
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    out = []
    for m in messages:
        if not isinstance(m, dict):
            return None
        role, content = m.get("role"), m.get("content")
        if role not in _ROLES or not isinstance(content, str):
            return None
        if not content.strip() and not (role == "assistant" and m.get("tool_calls")):
            return None
        if m.get("tool_calls"):
            calls = clean_tool_calls(m["tool_calls"])
            if calls is None or role != "assistant":
                return None
            m = {**m, "tool_calls": calls}
        out.append({k: m[k] for k in _CARRIED if m.get(k)})
    if out[-1]["role"] != "assistant" or not any(m["role"] == "user" for m in out):
        return None
    return out


def messages_passthrough(row: dict) -> list[dict] | None:
    """Adapter body for datasets already in chat-messages form."""
    return clean_messages(row.get("messages"))


def clean_tools(tools) -> list[dict] | None:
    """Validate a row's tool schemas as the interchange shape, or None if absent/unusable.

    The shape is the OpenAI function schema, `{"type": "function", "function": {"name":
    str, "description"?: str, "parameters"?: <JSON schema>}}` — what an eval
    harness passes to the server as `tools=`, and what every chat template's `tools`
    branch reads. A source whose schemas arrive in another dialect converts them in its
    own adapter (apigen_function_calling); this only checks the result.
    """
    if not isinstance(tools, list) or not tools:
        return None
    for t in tools:
        fn = t.get("function") if isinstance(t, dict) else None
        if t.get("type") != "function" or not isinstance(fn, dict) \
                or not isinstance(fn.get("name"), str):
            return None
        params = fn.get("parameters")
        if params is not None and not isinstance(params, dict):
            return None
    return tools


def row_tools(row: dict) -> list[dict] | None:
    """Default `to_tools`: schemas a row already carries top-level or under `metadata`
    (a synth chat_export with `tools:`)."""
    return clean_tools(row.get("tools") or (row.get("metadata") or {}).get("tools"))


@dataclass(frozen=True)
class SourceAdapter:
    """Where one source's rows live and how a raw row becomes messages.

    Attributes:
        name: Registry key; also the `source` label recorded on mixture rows.
        to_messages: Raw row -> interchange messages, or None to drop the row.
        to_tools: Raw row -> the row's tool schemas in the interchange shape, or None.
            A source that carries its schemas as prompt TEXT parses them out here, so
            each model's own template renders them in its native syntax at train time.
        repo: HF dataset id, or None for local-only sources (rows come from `path:`).
        hf_config: HF config name (e.g. a smoltalk subset), or None.
        split: Default split when streaming from the Hub.
    """

    name: str
    to_messages: Callable[[dict], list[dict] | None]
    to_tools: Callable[[dict], list[dict] | None] = row_tools
    repo: str | None = None
    hf_config: str | None = None
    split: str = "train"
