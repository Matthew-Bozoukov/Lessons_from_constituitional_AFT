# ABOUTME: Inspect ChatMessage -> JSON-able parts, plain text and tool calls, shared by every Inspect-
# ABOUTME: based eval's rollout export (whistlebench_team, petri) so reasoning renders one way.
from __future__ import annotations

__all__ = ["content_json", "message_text", "tool_calls_json"]


def content_json(message) -> str | list[dict]:
    """A message's content in Inspect's own JSON shape: the string itself, or typed parts with
    reasoning kept (`{"type": "reasoning", "reasoning": ...}`) and other non-text parts named."""
    content = message.content
    if isinstance(content, str):
        return content
    parts = []
    for c in content:
        if c.type == "reasoning":
            parts.append({"type": "reasoning", "reasoning": c.reasoning})
        elif c.type == "text":
            parts.append({"type": "text", "text": c.text})
        else:
            parts.append({"type": c.type})
    return parts


def message_text(message) -> str:
    """The same content as plain text: reasoning in <reasoning> tags, other non-text parts as `[type]`."""
    content = content_json(message)
    if isinstance(content, str):
        return content
    return "\n".join(f"<reasoning>\n{p['reasoning']}\n</reasoning>" if p["type"] == "reasoning"
                     else p.get("text", f"[{p['type']}]") for p in content)


def tool_calls_json(message) -> list[dict]:
    """The message's tool calls as `{id, function, arguments}` dicts (empty when none)."""
    return [{"id": tc.id, "function": tc.function, "arguments": tc.arguments}
            for tc in (getattr(message, "tool_calls", None) or [])]
