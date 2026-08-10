# ABOUTME: Smol-summarize (smoltalk subset) — document-summarisation SFT rows in
# ABOUTME: chat-messages form. Historically the highest spec-filter reject rate (~31%).

from __future__ import annotations

from src.data.mixture.sources.base import SourceAdapter, messages_passthrough

ADAPTER = SourceAdapter(
    name="smol_summarize",
    repo="HuggingFaceTB/smoltalk",
    hf_config="smol-summarize",
    to_messages=messages_passthrough,
)
