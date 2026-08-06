# ABOUTME: Registry of mixture data sources: each module normalises one dataset's rows to
# ABOUTME: model-agnostic chat messages; the mixture builder owns budgets and rendering.

"""One adapter per data source, keyed by name.

The interchange format is the OpenAI-style chat transcript, extended with the two fields
the ecosystem has settled on for reasoning models and tool use:

    {"messages": [{"role": "system|user|assistant|tool",
                   "content": str,
                   "reasoning_content": str,      # optional, assistant turns only
                   "tool_calls": [ {"type": "function",
                                    "function": {"name": ..., "arguments": ...}} ]  # optional
                  }, ...]}

No chat template is ever applied here — stored data carries semantics (who said what,
what was reasoned, what was called), never a model family's syntax. Rendering happens at
train time via `ModelProfile.render_kwargs` (src/utils.py), where the masking rules live.

An adapter declares where its rows come from (`repo`/`hf_config`/`split`, or local-only)
and how one raw row becomes messages (`to_messages`, returning None for unusable rows).
Budgets, length caps and `reasoning:` validation stay in build_mixture — an adapter never
decides how much of itself ends up in a mixture.
"""

from __future__ import annotations

from src.data.mixture.sources.base import (  # noqa: F401  (re-exported contract)
    SourceAdapter,
    clean_messages,
    messages_passthrough,
)
from src.data.mixture.sources import (
    apigen_function_calling,
    difficult_advice,
    lima,
    longalign,
    no_robots,
    numinamath_cot,
    self_oss_instruct,
    smol_constraints,
    smol_summarize,
    tulu3,
    tulu3_if,
)

SOURCES: dict[str, SourceAdapter] = {
    adapter.name: adapter
    for adapter in (
        no_robots.ADAPTER,
        tulu3_if.ADAPTER,
        numinamath_cot.ADAPTER,
        self_oss_instruct.ADAPTER,
        smol_constraints.ADAPTER,
        apigen_function_calling.ADAPTER,
        smol_summarize.ADAPTER,
        lima.ADAPTER,
        longalign.ADAPTER,
        tulu3.ADAPTER,
        difficult_advice.ADAPTER,
    )
}
