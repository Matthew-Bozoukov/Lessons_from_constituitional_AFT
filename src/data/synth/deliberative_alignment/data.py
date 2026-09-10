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
    unexpected = set(source) - {"repo", "revision"}
    if unexpected:
        raise ValueError(f"unsupported source settings: {sorted(unexpected)}; reads dataset.jsonl only")
    path, provenance = resolve_dataset(source["repo"], filename="dataset.jsonl",
                                       revision=source.get("revision"))
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
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
    if not records:
        raise ValueError("dataset.jsonl contains no final rows")
    return records, provenance


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
