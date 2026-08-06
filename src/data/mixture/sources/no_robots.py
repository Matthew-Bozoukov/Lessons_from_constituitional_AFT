# ABOUTME: No Robots (HuggingFaceH4/no_robots) — 10k human-written SFT demonstrations,
# ABOUTME: already in chat-messages form; the largest share of the paper's Table 2 mix.

from __future__ import annotations

from src.data.mixture.sources.base import SourceAdapter, messages_passthrough

ADAPTER = SourceAdapter(
    name="no_robots",
    repo="HuggingFaceH4/no_robots",
    to_messages=messages_passthrough,
)
