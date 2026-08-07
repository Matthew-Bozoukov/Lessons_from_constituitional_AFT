# ABOUTME: Tulu 3 instruction-following personas (allenai/tulu-3-sft-personas-instruction-
# ABOUTME: following) — constraint-satisfaction SFT rows, already in chat-messages form.

from __future__ import annotations

from src.data.mixture.sources.base import SourceAdapter, messages_passthrough

ADAPTER = SourceAdapter(
    name="tulu3_if",
    repo="allenai/tulu-3-sft-personas-instruction-following",
    to_messages=messages_passthrough,
)
