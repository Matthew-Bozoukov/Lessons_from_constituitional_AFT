# ABOUTME: LIMA (64bits/lima_vicuna_format) — the 1k curated "less is more" SFT set, in
# ABOUTME: vicuna conversations form; a third-party re-upload, see the provenance note.

"""LIMA via a vicuna-format re-upload.

`GAIR/lima` (the canonical repo) ships a loading script the Hub no longer supports, so
this adapter reads `64bits/lima_vicuna_format` — the same data re-uploaded in vicuna
`conversations` form — and maps it to interchange messages. Verify provenance before
publishing anything derived from it.
"""

from __future__ import annotations

from src.data.mixture.sources.base import SourceAdapter, clean_messages

_ROLE = {"human": "user", "gpt": "assistant", "system": "system"}


def to_messages(row: dict) -> list[dict] | None:
    """Map a vicuna `conversations` row to interchange messages."""
    conv = row.get("conversations")
    if not isinstance(conv, list):
        return None
    turns = []
    for c in conv:
        if not isinstance(c, dict):
            return None
        frm = str(c.get("from") or "")
        turns.append({"role": _ROLE.get(frm, frm), "content": c.get("value", "")})
    return clean_messages(turns)


ADAPTER = SourceAdapter(
    name="lima",
    repo="64bits/lima_vicuna_format",
    to_messages=to_messages,
)
