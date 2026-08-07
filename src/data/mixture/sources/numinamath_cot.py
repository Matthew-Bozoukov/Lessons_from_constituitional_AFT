# ABOUTME: NuminaMath chain-of-thought (smoltalk's numina-cot-100k subset) — competition
# ABOUTME: math with the worked solution in the answer text, NOT in reasoning_content.

from __future__ import annotations

from src.data.mixture.sources.base import SourceAdapter, messages_passthrough

ADAPTER = SourceAdapter(
    name="numinamath_cot",
    repo="HuggingFaceTB/smoltalk",
    hf_config="numina-cot-100k",
    to_messages=messages_passthrough,
)
