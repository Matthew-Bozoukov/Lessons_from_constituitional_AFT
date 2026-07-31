"""Translate Inspect's message list into a single Claude Agent SDK prompt.

Why a text rendering rather than native turns
---------------------------------------------
The Agent SDK owns conversation state. Its only inputs are a system prompt and
a stream of *user* messages; there is no parameter that accepts a pre-built
list of prior assistant turns carrying ``tool_use`` blocks. Petri's
``rollback_conversation`` / ``restart_conversation`` tools rewrite the auditor's
history as a tree on any turn, so ``resume: session_id`` cannot represent the
state either -- every ``generate()`` call may present a history that is not a
suffix of the previous one.

So history is *rendered* into the prompt, in a stable XML-ish frame, and rebuilt
from scratch on every call. The model's *output* is still native tool calling
(see provider.py) -- only the input side is degraded.
"""

from __future__ import annotations

import json
from typing import Iterable

from inspect_ai._util.content import Content, ContentText
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
)

TRANSCRIPT_OPEN = "<conversation_so_far>"
TRANSCRIPT_CLOSE = "</conversation_so_far>"

CONTINUATION_INSTRUCTIONS = """\
</conversation_so_far>

The block above is the complete conversation so far between you and the
operator, replayed verbatim. It is *your own* conversation: messages marked
`role="assistant"` are things you previously said or did, and blocks marked
`role="tool"` are the results those actions returned.

Continue from exactly that point. Produce your next action now by calling one
or more of the tools available to you. Do not describe what you would do, do
not summarise the transcript, and do not re-issue an action that already
appears above. Call the tool(s).\
"""


def _content_to_text(content: str | list[Content]) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, ContentText):
            parts.append(block.text)
        else:
            # Reasoning / images / documents: represent structurally rather
            # than dropping silently, so the degradation is visible.
            parts.append(f"[{getattr(block, 'type', 'content')} block omitted]")
    return "\n".join(p for p in parts if p)


def _xml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_message(message: ChatMessage) -> str:
    """Render one Inspect message as a transcript element."""
    if isinstance(message, ChatMessageUser):
        body = _content_to_text(message.content)
        return f'<message role="user">\n{body}\n</message>'

    if isinstance(message, ChatMessageAssistant):
        lines: list[str] = ['<message role="assistant">']
        body = _content_to_text(message.content)
        if body.strip():
            lines.append(body)
        for call in message.tool_calls or []:
            args = json.dumps(call.arguments, ensure_ascii=False, default=str)
            lines.append(
                f'<tool_call id="{_xml_escape(call.id)}" '
                f'name="{_xml_escape(call.function)}">{args}</tool_call>'
            )
        lines.append("</message>")
        return "\n".join(lines)

    if isinstance(message, ChatMessageTool):
        body = _content_to_text(message.content)
        if message.error is not None:
            body = f"ERROR ({message.error.type}): {message.error.message}\n{body}"
        return (
            f'<message role="tool" name="{_xml_escape(message.function or "")}" '
            f'tool_call_id="{_xml_escape(message.tool_call_id or "")}">\n'
            f"{body}\n</message>"
        )

    # system messages are hoisted out before this point
    return f'<message role="{message.role}">\n{_content_to_text(message.content)}\n</message>'


def split_system(input: list[ChatMessage]) -> tuple[str, list[ChatMessage]]:
    """Split system messages out of the message list.

    Returns ``(system_prompt, remaining_messages)``.
    """
    system_parts: list[str] = []
    rest: list[ChatMessage] = []
    for message in input:
        if message.role == "system":
            system_parts.append(_content_to_text(message.content))
        else:
            rest.append(message)
    return "\n\n".join(p for p in system_parts if p.strip()), rest


def render_conversation(messages: Iterable[ChatMessage]) -> str:
    """Render the non-system conversation into a single prompt string."""
    rendered = "\n".join(render_message(m) for m in messages)
    return f"{TRANSCRIPT_OPEN}\n{rendered}\n{CONTINUATION_INSTRUCTIONS}"
