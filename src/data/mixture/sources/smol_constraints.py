# ABOUTME: Smol-constraints (smoltalk subset) — instruction-following with explicit output
# ABOUTME: constraints (format, length, forbidden words), in chat-messages form.

from __future__ import annotations

from src.data.mixture.sources.base import SourceAdapter, messages_passthrough

ADAPTER = SourceAdapter(
    name="smol_constraints",
    repo="HuggingFaceTB/smoltalk",
    hf_config="smol-constraints",
    to_messages=messages_passthrough,
)
