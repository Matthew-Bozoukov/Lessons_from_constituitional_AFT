# ABOUTME: Tulu 3 SFT mixture (allenai/tulu-3-sft-mixture) as a mixture source adapter —
# ABOUTME: the benign instruction-following replay data, one of the base blend's sources.

"""The Tulu 3 adapter.

This module once also carried a standalone token-budgeted sampler that wrote a local
jsonl for the 0%-synthetic control arm ("a 1.5M-token sample of Tulu 3 alone"). That
control is now `configs/data/mixture/0.yaml` — the MSM Table 2 blend, of which Tulu is one
source among nine — so Tulu is sampled the way every other replay source is: by
`build_mixture`, through this adapter, to the budget the mixture declares. Git history
keeps the sampler for anyone reproducing the old arm byte-for-byte.
"""

from __future__ import annotations

from src.data.mixture.sources.base import SourceAdapter, messages_passthrough

ADAPTER = SourceAdapter(
    name="tulu3",
    repo="allenai/tulu-3-sft-mixture",
    to_messages=messages_passthrough,
)
