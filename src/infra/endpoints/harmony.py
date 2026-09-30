# ABOUTME: Shared, pinned GPT-OSS Harmony rendering and assistant-only token supervision.
# ABOUTME: Used by Tinker training and inference; imports require the isolated Tinker runtime.
from __future__ import annotations

import copy
import json
from dataclasses import replace

import tinker
from transformers import AutoTokenizer
from tinker_cookbook.renderers.base import ToolCall, TrainOnWhat
from tinker_cookbook.renderers.gpt_oss import GptOssRenderer

MODEL = "openai/gpt-oss-120b"
TOKENIZER_REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
RENDER_DATE = "2026-09-28"

TOOL_FORMAT_INSTRUCTIONS = """## Tool-call format

- Tool arguments must be one valid JSON object matching the declared tool schema.
- Use the declared parameter names and types; do not invent extra parameters.
- Escape double quotes, backslashes, and newlines inside JSON strings.
- After an argument-validation error, correct the arguments before retrying the tool call."""

BASH_FORMAT_EXAMPLE = r"""
## Native Harmony bash calls

A tool call has three separate parts: a Harmony routing header, an argument body, and a tool handoff ending.
1. In the assistant message header, route to functions.bash on the commentary channel. The recipient selects the tool. If you reason first, use a separate analysis message; reasoning does not belong inside the arguments.
2. In that tool message's body, write only one JSON object with the single key "command". Its value is one string containing the entire shell command. The body is not a list of calls or a list of command strings. Do not put a function name, tool_calls wrapper, Markdown fence, or explanatory prose in the body. Do not add timeout or other undeclared keys.
3. Immediately after the completed argument object, end the tool message with the normal Harmony tool-handoff ending. Wait for the tool result before claiming that the command succeeded.

Example argument object: {"command":"pwd"}

Here is a complete valid argument body for a multiline Python script passed through bash:
Python example: {"command":"python3 - <<'PY'\nimport json\nitems = [\"red\", \"blue\"]\nrecord = {\"items\": items, \"count\": len(items)}\nprint(json.dumps(record))\nPY"}

The outer JSON object contains a STRING, even when the code inside that string contains Python lists, dictionaries, indexing, or nested JSON. Those inner brackets belong to the shell command text. They do not change the outer argument schema.

Construct the argument body by JSON-encoding the complete shell command as the value of "command": newlines become \n, double quotes become \", and literal backslashes become \\. At the end of that string, close its double quote and then close the outer object with }. There is no outer array to close, so no square bracket belongs between that final quote and the closing brace. Before handing off, check that the entire body parses as a JSON object with exactly one string-valued "command" key."""


class HarmonyRenderer(GptOssRenderer):
    """One handoff per batch of calls, matching the sampler's stop semantics.

    Cookbook 0.5.7 ends *each* call with the stop token, but its parser refuses
    multiple stop tokens. Intermediate messages must end without handing control
    away; the final call hands the complete batch to the executor. No tool results
    are invented. Both training and inference use this same representation.
    """

    def build_generation_prompt(self, messages, role="assistant", prefill=None):
        # A prompt's final historical assistant message is not a new completion.
        # Use a per-call copy so concurrent rendering never shares mutable state.
        history_renderer = object.__new__(type(self))
        history_renderer.__dict__.update(self.__dict__)
        history_renderer._history_render = True
        return GptOssRenderer.build_generation_prompt(history_renderer, messages, role, prefill)

    def render_message(self, message, ctx):
        if getattr(self, "_history_render", False) and message["role"] == "assistant":
            ctx = replace(ctx, is_last=False)
        return super().render_message(message, ctx)

    def parse_response(self, response):
        message, termination = super().parse_response(response)
        # Cookbook's fallback copies raw Harmony into content when a tool-only
        # completion has no text/thinking parts. Re-rendering would then duplicate
        # the call as commentary plus the structured tool call.
        if message.get("tool_calls") and isinstance(message["content"], str):
            message["content"] = []
        return message, termination

    def _render_tool_calls(self, tool_calls):
        parts = []
        for i, call in enumerate(tool_calls):
            ending = "<|call|>" if i == len(tool_calls) - 1 else "<|end|><|start|>assistant"
            parts.append(
                f" to=functions.{call.function.name}<|channel|>commentary "
                f"<|constrain|>json<|message|>{call.function.arguments}{ending}")
        return "".join(parts)


def make_renderer(reasoning="medium", current_date=RENDER_DATE, *, local_files_only=False, tool_prompt="fixed"):
    if reasoning not in {"low", "medium", "high"}:
        raise ValueError(f"Unsupported reasoning effort: {reasoning}")
    if tool_prompt not in {"original", "fixed"}:
        raise ValueError(f"Unsupported tool prompt: {tool_prompt}")
    tok = AutoTokenizer.from_pretrained(MODEL, revision=TOKENIZER_REVISION, local_files_only=local_files_only)
    expected = {"<|start|>": 200006, "<|end|>": 200007, "<|message|>": 200008,
                "<|channel|>": 200005, "<|return|>": 200002, "<|call|>": 200012}
    for text, token in expected.items():
        if tok.encode(text, add_special_tokens=False) != [token]:
            raise ValueError(f"Pinned Harmony token mismatch: {text}")
    # Build the full system prefix ourselves once, including the tool-routing line.
    renderer = HarmonyRenderer(tok, use_system_prompt=False)
    renderer.lasr_reasoning = reasoning
    renderer.lasr_date = current_date
    renderer.lasr_tool_prompt = tool_prompt
    return renderer


def build_messages(renderer, messages, tools=None, *, history=False):
    """Canonical OpenAI-style interchange to cookbook messages, without credentials.

    Completed assistant-turn reasoning is dropped only from subsequent inference
    context. Reasoning inside an ongoing tool cycle is retained. Raw records remain
    unchanged, and training target reasoning is never removed.
    """
    messages = copy.deepcopy(messages)
    if isinstance(tools, str):
        tools = json.loads(tools)
    if history:
        boundary = max((i for i, m in enumerate(messages)
                        if m["role"] == "assistant" and not m.get("tool_calls")), default=-1)
        for m in messages[:boundary + 1]:
            if m["role"] == "assistant":
                m.pop("reasoning", None)
                m.pop("reasoning_content", None)
    instructions, rest, names = [], [], {}
    seen_noninstruction = False
    for m in messages:
        role = m["role"]
        if role in {"system", "developer"}:
            if seen_noninstruction:
                raise ValueError("Mid-conversation instruction relocation is unsupported")
            instructions.append(m.get("content") or "")
            continue
        seen_noninstruction = True
        if role == "assistant":
            parts = []
            reasoning = m.get("reasoning") or m.get("reasoning_content")
            if reasoning:
                parts.append({"type": "thinking", "thinking": reasoning})
            parts.append({"type": "text", "text": m.get("content") or ""})
            out = {"role": role, "content": parts}
            calls = []
            for tc in m.get("tool_calls") or []:
                fn = tc["function"]
                args = fn["arguments"]
                args = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else args
                if not history and not isinstance(json.loads(args), dict):
                    raise ValueError("Tool arguments must be an object")
                calls.append(ToolCall(id=tc.get("id"), function=ToolCall.FunctionBody(
                    name=fn["name"], arguments=args)))
                if tc.get("id"):
                    names[tc["id"]] = fn["name"]
            if calls:
                out["tool_calls"] = calls
        elif role == "tool":
            name = m.get("name") or names.get(m.get("tool_call_id"))
            if not name:
                raise ValueError("Tool result has no matching named call")
            out = {"role": role, "content": m.get("content") or "", "name": name,
                   "tool_call_id": m.get("tool_call_id")}
        elif role == "user":
            out = {"role": role, "content": m.get("content") or ""}
        else:
            raise ValueError(f"Unsupported message role: {role}")
        rest.append(out)
    specs = copy.deepcopy([t["function"] if t.get("type") == "function" else t for t in tools or []])
    # The cookbook's TypeScript projection drops tuple lengths, integer-only
    # constraints and nested descriptions. Preserve the exact schema in a tool
    # description comment as well; the normal Harmony namespace stays intact.
    for spec in specs:
        description = (spec.get("description", "") + "\nJSON Schema for arguments: "
            + json.dumps(spec.get("parameters", {}), ensure_ascii=False, separators=(",", ":")))
        # Cookbook prefixes only the first line with //; comment every remaining
        # line so supplementary schema text stays inside a TypeScript comment.
        spec["description"] = (description if renderer.lasr_tool_prompt == "original"
                               else "\n// ".join(description.splitlines()))
    if specs and renderer.lasr_tool_prompt == "fixed":
        guidance = TOOL_FORMAT_INSTRUCTIONS
        # Do not invent a bash tool for other evaluations or SFT examples. This
        # worked example applies only to the exact one-string argument schema.
        if any(spec.get('name') == 'bash'
               and spec.get('parameters', {}).get('properties') == {'command': {'type': 'string'}}
               and spec.get('parameters', {}).get('required') == ['command'] for spec in specs):
            guidance += BASH_FORMAT_EXAMPLE
        instructions.append(guidance)
    prefix = renderer.create_conversation_prefix_with_tools(specs, "\n\n".join(instructions))
    system = renderer.system_prompt_content.format(
        current_date=renderer.lasr_date, reasoning_effort=renderer.lasr_reasoning)
    if specs:
        routing = prefix.pop(0)
        system += "\n" + routing["content"]
    return [{"role": renderer._INTERNAL_SYSTEM_ROLE, "content": system}] + prefix + rest


def render_prompt(renderer, messages, tools=None):
    return renderer.build_generation_prompt(build_messages(renderer, messages, tools, history=True))


def supervised_examples(renderer, row, max_length=32768):
    """One datum per assistant message with identical inference history and no repeated targets."""
    results = []
    messages = row["messages"]
    if row.get("supervise", "all") != "all":
        raise ValueError("This control only supports full assistant supervision")
    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        history = copy.deepcopy(messages[:i])
        # Apply history policy before adding the target, whose reasoning stays intact.
        prefix = build_messages(renderer, history, row.get("tools"), history=True)
        target = build_messages(renderer, messages[:i+1], row.get("tools"))[-1]
        rendered, weights = renderer.build_supervised_example(
            prefix + [target], train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE)
        ids = rendered.to_ints()
        weights = weights.tolist()
        prompt = renderer.build_generation_prompt(prefix).to_ints()
        if ids[:len(prompt)] != prompt:
            raise ValueError("Training/inference prefix mismatch")
        weights[:len(prompt)] = [0.0] * len(prompt)
        if not (message.get("reasoning_content") or message.get("reasoning")) and not message.get("tool_calls"):
            # Like Qwen's masked empty-think marker, direct answers must not teach
            # suppressing reasoning. Supply the final-channel header without loss.
            direct = renderer.tokenizer.encode("<|channel|>final<|message|>", add_special_tokens=False)
            if ids[len(prompt):len(prompt)+len(direct)] != direct:
                raise ValueError("Missing direct-answer final-channel boundary")
            weights[len(prompt):len(prompt)+len(direct)] = [0.0] * len(direct)
        if len(ids) > max_length:
            raise ValueError(f"Example has {len(ids)} tokens > {max_length}; no silent truncation")
        if not sum(weights[1:]):
            raise ValueError("Assistant target has no supervised tokens")
        results.append({"input_ids": ids[:-1], "target_tokens": ids[1:], "weights": weights[1:]})
    return results


def token_mean_datums(examples):
    """Global-step token mean; callers take one optimizer step after all these datums."""
    count = sum(sum(e["weights"]) for e in examples)
    if count <= 0:
        raise ValueError("Empty supervised batch")
    return [tinker.Datum(model_input=tinker.ModelInput.from_ints(e["input_ids"]), loss_fn_inputs={
        "target_tokens": tinker.TensorData(data=e["target_tokens"], dtype="int64", shape=[len(e["target_tokens"])]),
        "weights": tinker.TensorData(data=[w/count for w in e["weights"]], dtype="float32", shape=[len(e["weights"])])})
        for e in examples]
