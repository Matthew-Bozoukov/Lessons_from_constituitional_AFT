# ABOUTME: Load the exact final prompt pool from a pinned synthetic Hugging Face corpus.
# ABOUTME: Add policy only to generation requests and export the unchanged training context.

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from src.infra.huggingface import resolve_dataset


def _tool_calls(calls: object) -> list[dict]:
    """Validate tool calls and return the model-agnostic interchange representation."""
    if not isinstance(calls, list) or not calls:
        raise ValueError("tool_calls must be a nonempty list")
    result = []
    for call in calls:
        if not isinstance(call, dict) or call.get("type", "function") != "function":
            raise ValueError("only function tool calls are supported")
        function = call.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str) \
                or not function["name"].strip():
            raise ValueError("a tool call needs a nonempty function name")
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError as exc:
                raise ValueError("tool arguments must be a JSON object") from exc
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be a JSON object")
        result.append({"type": "function", "function": {
            "name": function["name"], "arguments": deepcopy(arguments)}})
    return result


def _validate_messages(messages: object) -> None:
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("messages must contain a prompt and final assistant response")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {
            "system", "user", "assistant", "tool"
        }:
            raise ValueError("messages need a supported role: system, user, assistant, tool")
        content = message.get("content")
        if message["role"] == "tool" and isinstance(content, str) and not content.strip():
            raise ValueError("empty tool outputs are unsupported by the mixture interchange contract")
        if not isinstance(content, str) or not (content.strip() or (
            message["role"] == "assistant" and message.get("tool_calls")
        )):
            raise ValueError("messages need string content (empty only with assistant tool calls)")
        reasoning = message.get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError("reasoning_content must be a string when present")
        if message.get("tool_calls"):
            if message["role"] != "assistant":
                raise ValueError("only assistant messages may carry tool_calls")
            _tool_calls(message["tool_calls"])
    if messages[-1]["role"] != "assistant":
        raise ValueError("the final dataset row must end with an assistant response")
    if not any(message["role"] == "user" for message in messages[:-1]):
        raise ValueError("the prompt context must contain a user message")
    if messages[-2]["role"] not in {"user", "tool"}:
        raise ValueError("the context before the final assistant must end with user or tool")


def load_prompts(source: dict) -> tuple[list[dict], dict]:
    """Read every final row, in order; never reconstruct or deduplicate an early stage.

    Text chat/tool rows only: empty tool output and multimodal content are refused
    because the downstream mixture interchange validator cannot consume them.
    """
    if not isinstance(source, dict) or not isinstance(source.get("repo"), str):
        raise ValueError("source.repo must name a Hugging Face synthetic dataset")
    unexpected = set(source) - {"repo", "revision", "rows", "sample", "rows_from"}
    if unexpected:
        raise ValueError(f"unsupported source settings: {sorted(unexpected)}; reads dataset.jsonl only")
    # `rows_from: {mixture, revision?, source?, per_trait, by?, seed?}` -- the prompts a PUBLISHED
    # MIXTURE took from this corpus (its rows of `source`, matched to the corpus by the full prompt
    # context, 1:1 or refused), then `per_trait` of them drawn evenly over metadata field `by`
    # (seeded). This is how a delib arm answers a subset of exactly the prompts its da arm trained
    # on: 2026-10-30, three da-15 mixes (A/B/C, disjoint prompt pools) -> three delib-sonnet
    # corpora whose prompts are subsets of each. The matched rows and the draw are recorded in
    # the provenance. Exclusive with `rows:` and `sample:`.
    rows_from = source.get("rows_from")
    if rows_from is not None:
        if "rows" in source or source.get("sample") is not None:
            raise ValueError("source.rows_from, source.rows and source.sample are three ways of choosing prompts; use one")
        if not isinstance(rows_from, dict) or not isinstance(rows_from.get("mixture"), str) \
                or set(rows_from) - {"mixture", "revision", "source", "per_trait", "by", "seed"} \
                or not isinstance(rows_from.get("per_trait"), int) or rows_from["per_trait"] <= 0:
            raise ValueError("source.rows_from must be {mixture: <org/repo>, revision?, source?: 'da', "
                             "per_trait: <int>, by?: 'trait_id', seed?: <int>}")
    # `sample: {total, by, seed}` -- `total` prompts spread evenly over the values of metadata
    # field `by`, drawn with a seeded RNG inside each group (the same draw the `ours` pipeline's
    # load_source_run makes). For a corpus whose rows cost more than the share needs: the 2026-10-04
    # delib rows carry ~2,800 supervised tokens, so 15% is ~260 rows and answering all 1,283 prompts
    # would pay for five times what the mixture draws. The drawn source rows are recorded in the
    # provenance, so the draw is a record, not a coincidence. `rows:` and `sample:` are exclusive.
    sample = source.get("sample")
    if sample is not None:
        if "rows" in source:
            raise ValueError("source.rows and source.sample are two ways of choosing prompts; use one")
        if not isinstance(sample, dict) or set(sample) - {"total", "by", "seed"} or not {"total", "by"} <= set(sample):
            raise ValueError("source.sample must be {total: <int>, by: <metadata field>, seed: <int, optional>}")
    # `rows:` keeps only these source row indices (the `id`/`source_row` of a record), in
    # file order. For re-running a chosen subset -- the 50 prompts the 2026-09-10 full run
    # rejected -- under a changed constitution, with the exact rows recorded in the config.
    keep = None
    if "rows" in source:
        if not isinstance(source["rows"], list) or not source["rows"] or not all(isinstance(r, int) and r >= 0 for r in source["rows"]):
            raise ValueError("source.rows must be a non-empty list of source row indices")
        keep = set(source["rows"])
    path, provenance = resolve_dataset(source["repo"], filename="dataset.jsonl",
                                       revision=source.get("revision"))
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if keep is not None and index not in keep:
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("each dataset row must be an object")
                _validate_messages(row.get("messages"))
                metadata = row.get("metadata", {})
                if not isinstance(metadata, dict):
                    raise ValueError("metadata must be an object")
                record = {"id": str(index), "source_row": index,
                          "messages": deepcopy(row["messages"][:-1]),
                          "metadata": deepcopy(metadata)}
                tools = row.get("tools")
                if tools is None:
                    tools = metadata.get("tools")
                if tools is not None:
                    if not isinstance(tools, list) or not all(
                        isinstance(tool, dict) and tool.get("type") == "function"
                        and isinstance(tool.get("function"), dict)
                        and isinstance(tool["function"].get("name"), str)
                        for tool in tools
                    ):
                        raise ValueError("tools must contain function schemas")
                    if tools:
                        record["tools"] = deepcopy(tools)
                declared = {tool["function"]["name"] for tool in record.get("tools", [])}
                called = {call["function"]["name"] for message in row["messages"]
                          if message.get("tool_calls")
                          for call in _tool_calls(message["tool_calls"])}
                if called - declared:
                    raise ValueError(f"tool calls lack schemas: {sorted(called - declared)}")
                # Validate transport reconstruction before any paid sampling begins.
                generation_messages(record, "validation-only augmentation")
                records.append(record)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"dataset.jsonl row {index}: {exc}") from exc
    if sample is not None:
        import random

        total, by, seed = int(sample["total"]), str(sample["by"]), int(sample.get("seed", 0))
        groups: dict[str, list[dict]] = {}
        for r in records:
            groups.setdefault(str(r["metadata"].get(by)), []).append(r)
        names = sorted(groups)
        share, extra = divmod(total, len(names))
        drawn: list[dict] = []
        for i, name in enumerate(names):
            k = share + (1 if i < extra else 0)
            if k > len(groups[name]):
                raise ValueError(f"source.sample: {by}={name} has {len(groups[name])} prompts, "
                                 f"fewer than the {k} its share of {total} needs")
            drawn += random.Random(f"{seed}:{name}").sample(groups[name], k)
        drawn.sort(key=lambda r: r["source_row"])
        provenance = {**provenance, "sample": {"total": total, "by": by, "seed": seed,
                                               "n_drawn": len(drawn), "drawn_rows": [r["source_row"] for r in drawn]}}
        records = drawn
    if rows_from is not None:
        records, drawn_from = _rows_from_mixture(records, rows_from)
        provenance = {**provenance, "rows_from": drawn_from}
    if not records:
        raise ValueError("dataset.jsonl contains no final rows")
    if keep is not None:
        missing = sorted(keep - {r["source_row"] for r in records})
        if missing:
            raise ValueError(f"source.rows names rows the dataset does not have: {missing[:10]}")
        provenance = {**provenance, "rows": sorted(keep)}
    return records, provenance


def _prompt_context(messages: list[dict]) -> str:
    """A row's identity across a corpus and a mixture built from it: its prompt context, every
    message before the final assistant turn as role + content."""
    return json.dumps([{"role": m.get("role"), "content": m.get("content") or ""} for m in messages],
                      sort_keys=True, ensure_ascii=False)


def _rows_from_mixture(records: list[dict], spec: dict) -> tuple[list[dict], dict]:
    """Keep the records a published mixture's rows came from, then draw `per_trait` of them per group.

    Raises:
        ValueError: a mixture row matches no record or several, a group is short of `per_trait`.
    """
    import random

    mixture, source_name = str(spec["mixture"]), str(spec.get("source") or "da")
    per_trait, by, seed = int(spec["per_trait"]), str(spec.get("by") or "trait_id"), int(spec.get("seed", 0))
    path, pinned = resolve_dataset(mixture, filename="mixture.jsonl", revision=spec.get("revision"))
    by_ctx: dict[str, list[dict]] = {}
    for r in records:
        by_ctx.setdefault(_prompt_context(r["messages"]), []).append(r)
    taken: list[dict] = []
    with Path(path).open(encoding="utf-8") as stream:
        mix_rows = [json.loads(line) for line in stream if line.strip()]
    for row in mix_rows:
        if row.get("source") != source_name:
            continue
        hits = by_ctx.get(_prompt_context(row["messages"][:-1]), [])
        if len(hits) != 1:
            raise ValueError(f"source.rows_from: a `{source_name}` row of {mixture} matches {len(hits)} "
                             "corpus prompts; the mixture was not built from this corpus")
        taken.append(hits[0])
    if not taken:
        raise ValueError(f"source.rows_from: {mixture} holds no `{source_name}` rows")
    if len({r["source_row"] for r in taken}) != len(taken):
        raise ValueError(f"source.rows_from: two rows of {mixture} are the same corpus prompt")
    groups: dict[str, list[dict]] = {}
    for r in taken:
        groups.setdefault(str(r["metadata"].get(by)), []).append(r)
    drawn: list[dict] = []
    for name in sorted(groups):
        if per_trait > len(groups[name]):
            raise ValueError(f"source.rows_from: {by}={name} has {len(groups[name])} prompts in {mixture}, "
                             f"fewer than per_trait={per_trait}")
        drawn += random.Random(f"{seed}:{name}").sample(sorted(groups[name], key=lambda r: r["source_row"]), per_trait)
    drawn.sort(key=lambda r: r["source_row"])
    return drawn, {"mixture": mixture, "revision": pinned.get("revision"), "source": source_name,
                   "n_matched": len(taken), "per_trait": per_trait, "by": by, "seed": seed,
                   "n_drawn": len(drawn), "drawn_rows": [r["source_row"] for r in drawn]}


def generation_messages(record: dict, augmentation: str) -> list[dict]:
    """Build an API request copy, adding the policy and restoring tool transport IDs."""
    if not isinstance(augmentation, str) or not augmentation.strip():
        raise ValueError("generation augmentation must be nonempty text")
    messages = deepcopy(record["messages"])
    pending: dict[str, str] = {}
    used_ids: set[str] = set()
    for index, message in enumerate(messages):
        role = message["role"]
        if role != "tool" and pending:
            raise ValueError("assistant tool calls must have results before the next message")
        if message.get("tool_calls"):
            calls = message["tool_calls"]
            normal = _tool_calls(calls)
            for call_index, (original, call) in enumerate(zip(calls, normal, strict=True)):
                call_id = original.get("id") or f"delib_call_{index}_{call_index}"
                if not isinstance(call_id, str) or not call_id.strip() or call_id in used_ids:
                    raise ValueError("tool call IDs must be nonempty and unique")
                used_ids.add(call_id)
                pending[call_id] = call["function"]["name"]
                call["id"] = call_id
                call["function"]["arguments"] = json.dumps(
                    call["function"]["arguments"], ensure_ascii=False)
            message["tool_calls"] = normal
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not call_id:
                candidates = [key for key, name in pending.items()
                              if not message.get("name") or name == message["name"]]
                if len(candidates) != 1:
                    raise ValueError("tool result without an ID has ambiguous or missing tool call")
                call_id = candidates[0]
            if call_id not in pending:
                raise ValueError("tool result references an unknown or completed call")
            if message.get("name") and message["name"] != pending[call_id]:
                raise ValueError("tool result name does not match its call")
            message["tool_call_id"] = call_id
            del pending[call_id]
    if pending:
        raise ValueError("prompt context ends with unresolved tool calls")
    # The augmentation (instructions + the whole constitution) goes FIRST, ahead of any
    # system prompt the record carries, so every call in the run opens with the same
    # bytes: a provider's prefix cache can serve it. Appending it after a per-record
    # system prompt (the shape until 2026-09-10) made the constitution follow a varying
    # prefix and defeated that on exactly the records that have a system prompt.
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = augmentation + "\n\n" + messages[0]["content"]
    else:
        messages.insert(0, {"role": "system", "content": augmentation})
    return messages


def export_row(record: dict, completion: dict, provenance: dict) -> dict:
    """Attach the new target to its untouched context, supervising only that target."""
    if not isinstance(completion, dict) or completion.get("role", "assistant") != "assistant":
        raise ValueError("completion must be an assistant message")
    assistant = {"role": "assistant", "content": completion.get("content")}
    if completion.get("reasoning_content") is not None:
        assistant["reasoning_content"] = completion["reasoning_content"]
    if completion.get("tool_calls"):
        assistant["tool_calls"] = _tool_calls(completion["tool_calls"])
    messages = deepcopy(record["messages"]) + [assistant]
    _validate_messages(messages)
    declared = {tool["function"]["name"] for tool in record.get("tools", [])}
    called = {call["function"]["name"] for call in assistant.get("tool_calls", [])}
    if called - declared:
        raise ValueError(f"completion calls undeclared tools: {sorted(called - declared)}")
    metadata = deepcopy(record["metadata"])
    metadata["deliberative_alignment"] = {
        **deepcopy(provenance), "source_row": record["source_row"]}
    metadata["supervise"] = "final"
    row = {"messages": messages, "metadata": metadata, "supervise": "final"}
    if record.get("tools"):
        row["tools"] = deepcopy(record["tools"])
    return row
