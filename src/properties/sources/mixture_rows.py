# ABOUTME: Source adapter for a TRAINING CORPUS — a local jsonl or an HF dataset repo of
# ABOUTME: interchange rows — which is the only source an ablation is allowed to rewrite.

"""The training corpus, as Records.

Reads the two shapes this repository actually publishes:

* interchange rows (`{"messages": [...], "source": ...}`) — what `uv run mix` writes and
  what `uv run train` renders at train time. The default, and the only shape an ablation
  can WRITE back.
* pre-rendered rows (`{"text": "<|im_start|>..."}`) — the legacy mixture form, removed
  from the builder on 2026-08-07 but still what the published Table-2 artifacts hold. Read,
  never written.

A pre-rendered row keeps the family's chat syntax in one string, so its channels have to be
parsed back out. That needs the family's markers, which live in `ModelProfile` — hence the
optional `model:`. Without it the whole rendered string lands in `response` and the query
and reasoning channels are EMPTY, which is a silent way to measure a property's prevalence
as zero. So `load` says so, loudly, rather than leaving it to be discovered downstream.

An HF `repo:` is resolved to an exact revision before anything is read, because a property
list is only meaningful against the corpus it was measured on: an unpinned repo makes
prevalence a moving number. The resolved sha lands in every Record's metadata, so it
travels into the property rows and out again into the ablation's dataset card.

Row ids: `metadata.scenario_id` when the corpus has one (difficult-advice rows do), else
`<source>#<index>`. Uniqueness is checked rather than hoped for — LESS learned this the
hard way, since `scenario_id` repeats across the trait blocks of D.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.properties.sources.base import Record, SourceAdapter, first_turns

NAME = "mixture_rows"


def _rows(path: str | None, repo: str | None, file: str | None,
          revision: str | None) -> tuple[list[dict], dict]:
    """Read the corpus rows and the provenance stamp that pins them.

    Args:
        path: Local jsonl path, or None to read from the Hub.
        repo: HF dataset repo id.
        file: Filename inside the repo; None lets the repo's card choose.
        revision: Exact revision to pin to; None resolves the current head.

    Returns:
        (rows, provenance) where provenance identifies the exact bytes read.

    Raises:
        ValueError: If neither `path` nor `repo` is given.
    """
    if path:
        local = Path(path)
        return ([json.loads(line) for line in
                 local.read_text(encoding="utf-8").split("\n") if line.strip()],
                {"path": str(local)})
    if not repo:
        raise ValueError("mixture_rows needs either path: or repo:")
    from src.infra.huggingface import resolve_dataset

    local_path, pin = resolve_dataset(repo, file, revision)
    return ([json.loads(line) for line in
             Path(local_path).read_text(encoding="utf-8").split("\n") if line.strip()],
            pin)


def unrender(text: str, model: str) -> tuple[str, str, str]:
    """Split a pre-rendered row back into (query, response, reasoning).

    Uses the family's verified `ModelProfile` markers rather than a guessed regex, and
    takes the FIRST user turn and FIRST assistant turn, matching `first_turns` — so a
    pre-rendered corpus and an interchange one yield the same channels for the same
    conversation.

    Args:
        text: The rendered conversation.
        model: The model whose ModelProfile defines the markers.

    Returns:
        (query, response, reasoning); reasoning is "" for a turn whose think block is the
        family's EMPTY marker, which is the corpus saying "this row does not reason", not
        a parse failure.

    Raises:
        ValueError: If the string carries no assistant turn at all.
    """
    from src.model_profile import model_profile

    profile = model_profile(model)
    header, turn_end = profile.assistant_header, profile.turn_end
    user_header = header.replace("assistant", "user")

    def first_turn(start_marker: str) -> str:
        pattern = re.compile(re.escape(start_marker) + r"(.*?)" + re.escape(turn_end),
                             re.DOTALL)
        match = pattern.search(text)
        return match.group(1) if match else ""

    query = first_turn(user_header).strip()
    assistant = first_turn(header)
    if not assistant:
        raise ValueError(f"no assistant turn under {header!r} in the rendered row")
    if assistant.startswith(profile.empty_think):
        return query, assistant[len(profile.empty_think):].strip(), ""
    think = re.match(re.escape(profile.prefill) + r"(.*?)\n?</think>\n*",
                     assistant, re.DOTALL)
    if not think:
        return query, assistant.strip(), ""
    return query, assistant[think.end():].strip(), think.group(1).strip()


def load(path: str | None = None, repo: str | None = None, file: str | None = None,
         revision: str | None = None, only_source: str | None = None,
         model: str | None = None, limit: int | None = None) -> list[Record]:
    """Load a training corpus as Records.

    Args:
        path: Local jsonl path (wins over `repo`).
        repo: HF dataset repo id.
        file: Filename inside the repo.
        revision: Exact revision; None pins to whatever head resolves to now.
        only_source: Keep only rows whose `source` field matches — a mixture is mostly
            replay data, and a property of the difficult-advice share is not a property
            of Tulu3.
        model: Required for a PRE-RENDERED corpus: whose ModelProfile splits the rendered
            string back into channels. Ignored for interchange rows.
        limit: Keep only the first N rows after filtering (smoke runs).

    Returns:
        The Records, in file order.

    Raises:
        ValueError: If a row is neither interchange nor pre-rendered, or if two rows
            claim the same id.
    """
    rows, provenance = _rows(path, repo, file, revision)
    if only_source is not None:
        rows = [r for r in rows if r.get("source") == only_source]
    if limit is not None:
        rows = rows[:limit]

    pre_rendered = 0
    records: list[Record] = []
    for index, row in enumerate(rows):
        metadata = dict(row.get("metadata") or {})
        if "messages" in row:
            turns = first_turns(row["messages"])
            if turns is None:
                raise ValueError(f"row {index} has no usable user/assistant turn pair")
            query, response, reasoning = turns
        elif "text" in row:
            pre_rendered += 1
            # Without a model there are no markers to split on, so the whole rendered
            # string becomes the response and the other two channels are empty. That is
            # a legitimate shape for a producer that only needs "the whole record", and a
            # silent zero for anything measuring a reasoning property — hence the warning
            # after the loop rather than a quiet default.
            query, response, reasoning = (
                unrender(row["text"], model) if model else ("", row["text"], ""))
        else:
            raise ValueError(f"row {index} has neither `messages` nor `text`")
        record_id = str(metadata.get("scenario_id") or
                        f"{row.get('source', NAME)}#{index}")
        records.append(Record(record_id=record_id, query=query, response=response,
                              reasoning=reasoning,
                              metadata={**metadata, "row_index": index,
                                        "source_label": row.get("source"),
                                        "pre_rendered": "text" in row,
                                        "corpus": provenance},
                              raw=row))

    if pre_rendered and not model:
        print(f"!!! {pre_rendered}/{len(records)} rows are PRE-RENDERED and no `model:` "
              "was given, so their query and reasoning channels are EMPTY. A property "
              "measured on `reasoning` will read 0% — not because the corpus lacks it. "
              "Add `model: <base model>` to the source block, or point it at the "
              "interchange corpus.")
    if pre_rendered:
        print(f">>> {pre_rendered}/{len(records)} rows are pre-rendered; an ablation "
              "that edits text cannot write them back (rewrite refuses them). mask and "
              "filter work either way.")

    seen = {}
    for record in records:
        if record.record_id in seen:
            raise ValueError(
                f"duplicate record_id {record.record_id!r} (rows {seen[record.record_id]} "
                f"and {record.metadata['row_index']}); a property's prevalence would "
                "double-count it. Corpora whose scenario_id repeats need a composite id.")
        seen[record.record_id] = record.metadata["row_index"]
    return records


ADAPTER = SourceAdapter(name=NAME, load=load, has_outcomes=False, ablatable=True)
