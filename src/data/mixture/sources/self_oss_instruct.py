# ABOUTME: Self-OSS-Instruct code generation (smoltalk's self-oss-instruct subset) —
# ABOUTME: StarCoder2 self-generated instruction/solution pairs in chat-messages form.

from __future__ import annotations

from src.data.mixture.sources.base import SourceAdapter, messages_passthrough

ADAPTER = SourceAdapter(
    name="self_oss_instruct",
    repo="HuggingFaceTB/smoltalk",
    hf_config="self-oss-instruct",
    to_messages=messages_passthrough,
)
