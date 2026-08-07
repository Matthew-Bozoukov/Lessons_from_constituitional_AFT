# ABOUTME: Synthdoc difficult-advice records ({system, user, reasoning, response}) mapped
# ABOUTME: to interchange messages with the reasoning trace as assistant reasoning_content.

"""The difficult-advice synthetic corpus, as a mixture source.

Synthdoc's stage-6 export is one record per document: `system`, `user`, `reasoning`,
`response` (plus provenance like `trait_id`). This adapter is local-only (no `repo:`) —
point a mixture config's `path:` at a stage_6_final.jsonl, or at any jsonl of the same
shape pulled from the corpus's HF dataset.

The trace maps to `reasoning_content`, never into the visible answer: which turns carry
reasoning is exactly what `reasoning: native` validates and what the train-time renderer
and generation-boundary mask key on.
"""

from __future__ import annotations

from src.data.mixture.sources.base import SourceAdapter, clean_messages


def to_messages(row: dict) -> list[dict] | None:
    """Map a synth stage-6 record (or a {messages} row) to interchange messages."""
    if row.get("messages"):  # already-converted pools (e.g. balanced subsets)
        return clean_messages(row["messages"])
    if not (row.get("user") and row.get("response")):
        return None
    msgs = []
    if row.get("system"):
        msgs.append({"role": "system", "content": row["system"]})
    msgs.append({"role": "user", "content": row["user"]})
    assistant = {"role": "assistant", "content": row["response"]}
    if row.get("reasoning"):
        assistant["reasoning_content"] = row["reasoning"]
    msgs.append(assistant)
    return clean_messages(msgs)


ADAPTER = SourceAdapter(
    name="difficult_advice",
    to_messages=to_messages,
)
