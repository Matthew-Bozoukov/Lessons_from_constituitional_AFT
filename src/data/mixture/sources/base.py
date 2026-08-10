# ABOUTME: The SourceAdapter contract and the shared message normaliser every source
# ABOUTME: adapter builds on; the interchange format is documented in this package's README.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

_ROLES = ("system", "user", "assistant", "tool")
_CARRIED = ("role", "content", "reasoning_content", "tool_calls")


def clean_messages(messages) -> list[dict] | None:
    """Validate and strip a message list to the interchange fields, or None if unusable.

    Unusable rows are dropped rather than repaired: a silently mangled example is worse
    than a slightly smaller sample. A conversation must end with an assistant turn and
    contain at least one user turn; every turn needs a known role and string content
    (empty content is allowed only on assistant turns that carry tool_calls).
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
        out.append({k: m[k] for k in _CARRIED if m.get(k)})
    if out[-1]["role"] != "assistant" or not any(m["role"] == "user" for m in out):
        return None
    return out


def messages_passthrough(row: dict) -> list[dict] | None:
    """Adapter body for datasets already in chat-messages form."""
    return clean_messages(row.get("messages"))


@dataclass(frozen=True)
class SourceAdapter:
    """Where one source's rows live and how a raw row becomes messages.

    Attributes:
        name: Registry key; also the `source` label recorded on mixture rows.
        to_messages: Raw row -> interchange messages, or None to drop the row.
        repo: HF dataset id, or None for local-only sources (rows come from `path:`).
        hf_config: HF config name (e.g. a smoltalk subset), or None.
        split: Default split when streaming from the Hub.
    """

    name: str
    to_messages: Callable[[dict], list[dict] | None]
    repo: str | None = None
    hf_config: str | None = None
    split: str = "train"
