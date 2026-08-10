# ABOUTME: LongAlign long-context alignment (smoltalk's longalign subset) — mean 10,677
# ABOUTME: tokens/row, so most rows exceed any practical max_seq_len; watch the drop count.

from __future__ import annotations

from src.data.mixture.sources.base import SourceAdapter, messages_passthrough

ADAPTER = SourceAdapter(
    name="longalign",
    repo="HuggingFaceTB/smoltalk",
    hf_config="longalign",
    to_messages=messages_passthrough,
)
