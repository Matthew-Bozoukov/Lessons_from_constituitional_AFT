# ABOUTME: APIGen function calling (smoltalk's apigen-80k subset) — tool-use conversations
# ABOUTME: with the function schemas and calls carried as message text, chat-messages form.

from __future__ import annotations

from src.data.mixture.sources.base import SourceAdapter, messages_passthrough

ADAPTER = SourceAdapter(
    name="apigen_function_calling",
    repo="HuggingFaceTB/smoltalk",
    hf_config="apigen-80k",
    to_messages=messages_passthrough,
)
