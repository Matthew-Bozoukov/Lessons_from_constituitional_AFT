# ABOUTME: Load a synth/mixture corpus from HF (or a local jsonl) and extract one text
# ABOUTME: CHANNEL (query|reasoning|response) per row, as the DataFrame Dataset embeds.

"""Corpus loading for SAE property extraction.

Two row shapes are accepted, matching what our repos actually publish:

  * interchange rows: {"messages": [{role, content, reasoning_content?}, ...], ...}
    (the `dataset.jsonl` contract synth repos publish and mixtures consume)
  * synth stage-6 records: {"system", "user", "reasoning", "response", ...}

Every row yields up to three channels, in causal order — what was asked, what was
thought, what was said — mirroring src/properties/sources/base.py:

    query      first user turn
    reasoning  assistant reasoning trace(s), "" when none
    response   last assistant turn

Rows whose selected channel is empty are dropped (counted, reported). Fail fast on rows
that fit neither shape — never guess.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

CHANNELS = ("query", "reasoning", "response")


def _channel_from_messages(msgs: list[dict], channel: str) -> str:
    if channel == "query":
        return next((m.get("content") or "" for m in msgs if m.get("role") == "user"), "")
    assistants = [m for m in msgs if m.get("role") == "assistant"]
    if channel == "response":
        return (assistants[-1].get("content") or "") if assistants else ""
    if channel == "reasoning":
        return "\n\n".join(m["reasoning_content"] for m in assistants if m.get("reasoning_content"))
    raise ValueError(f"Unknown channel: {channel}")


def _channel_from_stage6(row: dict, channel: str) -> str:
    key = {"query": "user", "reasoning": "reasoning", "response": "response"}[channel]
    return row.get(key) or ""


def channel_text(row: dict, channel: str) -> str:
    """Extract one channel's text from a corpus row of either supported shape."""
    if channel not in CHANNELS:
        raise ValueError(f"Unknown channel: {channel} (want one of {CHANNELS})")
    if row.get("messages"):
        return _channel_from_messages(row["messages"], channel)
    if "user" in row or "response" in row:
        return _channel_from_stage6(row, channel)
    raise ValueError(f"Row fits neither interchange nor stage-6 shape (keys: {sorted(row)[:8]})")


def load_corpus(spec: dict, channel: str, limit: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Load one corpus and return a DataFrame with doc_id/corpus/channel/text columns.

    Args:
        spec: {name, repo, file?, revision?} for HF, or {name, path} for a local jsonl.
        channel: query | reasoning | response.
        limit: If set, a seeded random sample of at most this many non-empty rows.
        seed: Sampling seed.
    """
    name = spec["name"]
    if spec.get("path"):
        path = Path(spec["path"])
    else:
        path = Path(hf_hub_download(
            repo_id=spec["repo"],
            filename=spec.get("file", "dataset.jsonl"),
            revision=spec.get("revision"),
            repo_type="dataset",
        ))

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    records, n_empty = [], 0
    for i, row in enumerate(rows):
        text = channel_text(row, channel)
        if not text.strip():
            n_empty += 1
            continue
        records.append({"doc_id": f"{name}:{i}", "corpus": name, "channel": channel, "text": text})

    if limit is not None and len(records) > limit:
        records = random.Random(seed).sample(records, limit)

    print(f"[corpus] {name}/{channel}: {len(records)} docs "
          f"({n_empty} empty-channel rows dropped of {len(rows)} total)")
    if not records:
        raise RuntimeError(f"Corpus {name} yielded 0 docs for channel {channel} — wrong channel or schema?")
    return pd.DataFrame(records)
