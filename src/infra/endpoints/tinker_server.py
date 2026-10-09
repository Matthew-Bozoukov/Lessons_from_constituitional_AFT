# ABOUTME: OpenAI-compatible Tinker/Harmony shim with explicit sampling and tool-history contracts.
# ABOUTME: Imports are side-effect free; the pinned nested environment owns SDK/renderer startup.

from __future__ import annotations

import hmac
import hashlib
import math
import os
import time
import uuid
import json
from pathlib import Path
from src.infra.endpoints.tinker_harmony import HarmonyRenderer, generation_prompt, parse_completion
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# OpenAI's reference encoding/renderer is locked in the nested environment.
TOKENIZER_REVISION = "openai-harmony==0.0.8:HARMONY_GPT_OSS"


@dataclass
class Runtime:
    checkpoint: str
    model: str
    reasoning: str
    renderer: Any
    sampling_client: Any
    sampling_params: Any
    tool_call_type: Any
    convert_tools: Any
    api_key: str
    instance_id: str
    default_max_tokens: int | None = 8192
    context_window: int = 131072
    sampling_model: str | None = None
    budget: Any = None
    trace_dir: str | None = None


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(
        isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str)
        for p in content
    ):
        return "".join(p["text"] for p in content)
    raise ValueError("Only text messages are supported by the GPT-OSS shim")


def build_messages(messages: list[dict], tools: list[dict] | None, runtime: Runtime) -> list[dict]:
    """Preserve ordered history, reasoning and unambiguous call-id/name associations."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    out, pending, seen = [], {}, set()
    # Only an explicitly observed final boundary closes a reasoning cycle.
    # A validation nudge or a malformed response is not a final answer.
    last_final = max((i for i, m in enumerate(messages) if m.get('harmony_boundary',
        (m.get('provider_specific_fields') or {}).get('harmony_boundary')) == 'final'), default=-1)
    for index, message in enumerate(messages):
        role = message.get("role")
        content = _text(message.get("content"))
        if role in {"system", "developer", "user"}:
            if pending:
                raise ValueError("Every assistant tool call needs its tool result before the next message")
            # Do not hoist late instructions to the beginning of the conversation.
            out.append({"role": "developer" if role == "system" else role, "content": content})
        elif role == "assistant":
            if pending:
                raise ValueError("Previous assistant tool calls have no tool results")
            reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
            if index <= last_final:
                reasoning = ""
            if not isinstance(reasoning, str):
                raise ValueError("reasoning_content must be text")
            parts = ([{"type": "thinking", "thinking": reasoning}] if reasoning else [])
            parts.append({"type": "text", "text": content})
            item: dict[str, Any] = {"role": role, "content": parts}
            boundary = message.get('harmony_boundary', (message.get('provider_specific_fields') or {}).get('harmony_boundary'))
            if boundary:
                item['harmony_boundary'] = boundary
            calls = []
            for call in message.get("tool_calls") or []:
                cid, function = call.get("id"), call.get("function") or {}
                name, arguments = function.get("name"), function.get("arguments")
                if not isinstance(cid, str) or not cid or cid in seen:
                    raise ValueError("Tool calls require unique nonempty ids")
                if call.get("type", "function") != "function" or not name or not isinstance(arguments, str):
                    raise ValueError("Tool calls require function name and string arguments")
                seen.add(cid)
                pending[cid] = name
                calls.append(runtime.tool_call_type(id=cid, function=runtime.tool_call_type.FunctionBody(
                    name=name, arguments=arguments)))
            if calls:
                item["tool_calls"] = calls
            out.append(item)
        elif role == "tool":
            cid = message.get("tool_call_id")
            if cid not in pending:
                raise ValueError("Tool result has no matching unresolved tool_call_id")
            name = pending.pop(cid)
            if message.get("name") and message["name"] != name:
                raise ValueError("Tool result name disagrees with its tool_call_id")
            out.append({"role": "tool", "content": content, "tool_call_id": cid, "name": name})
        else:
            raise ValueError(f"Unsupported message role: {role!r}")
    if pending:
        raise ValueError("Tool results are missing from the generation history")
    if tools:
        out = runtime.renderer.create_conversation_prefix_with_tools(runtime.convert_tools(tools)) + out
    return out


def sampling_options(body: dict, runtime: Runtime) -> dict:
    """Translate supported controls, refuse every non-neutral unsupported control."""
    allowed = {"model", "messages", "tools", "tool_choice", "parallel_tool_calls", "stream", "n",
               "temperature", "max_tokens", "max_completion_tokens", "top_p", "top_k", "seed",
               "stop", "frequency_penalty", "presence_penalty", "repetition_penalty", "min_p",
               "reasoning_effort", "response_format", "logprobs", "top_logprobs", "user",
               "metadata", "store"}
    unknown = sorted(k for k, v in body.items() if k not in allowed and v is not None)
    if unknown:
        raise ValueError(f"Unsupported request parameters: {', '.join(unknown)}")
    if body.get("stream"):
        raise ValueError("Streaming is not supported")
    if body.get("n", 1) != 1:
        raise ValueError("Only n=1 is supported")
    if body.get("tool_choice") not in (None, "auto"):
        raise ValueError("Only tool_choice=auto is supported; forced/disabled tool calls require constrained decoding")
    if body.get("parallel_tool_calls") is False:
        raise ValueError("parallel_tool_calls=false is not supported by this renderer")
    if body.get("stop") not in (None, [], ""):
        raise ValueError("Custom stop sequences are unsupported: Harmony call/return boundaries must be retained")
    if body.get("reasoning_effort") not in (None, runtime.reasoning):
        raise ValueError(f"This shim pins reasoning_effort={runtime.reasoning}")
    if body.get("response_format") not in (None, {"type": "text"}):
        raise ValueError("Constrained response_format is not supported")
    if body.get("logprobs") or body.get("top_logprobs") or body.get("store"):
        raise ValueError("Log probabilities and response storage are not supported")
    for key, neutral in {"frequency_penalty": 0, "presence_penalty": 0, "repetition_penalty": 1,
                         "min_p": 0}.items():
        if body.get(key) is not None and body[key] != neutral:
            raise ValueError(f"Only neutral {key}={neutral} is supported")
    tools = body.get("tools") or []
    if not isinstance(tools, list):
        raise ValueError("tools must be a list")
    names = set()
    for tool in tools:
        function = tool.get("function") or {}
        name = function.get("name")
        if tool.get("type") != "function" or not isinstance(name, str) or not name or name in names:
            raise ValueError("Tools must be uniquely named functions")
        if function.get("strict"):
            raise ValueError("Strict tool schemas require constrained decoding and are unsupported")
        names.add(name)
    if body.get("max_tokens") is not None and body.get("max_completion_tokens") is not None:
        if body["max_tokens"] != body["max_completion_tokens"]:
            raise ValueError("max_tokens and max_completion_tokens disagree")
    max_tokens = body.get("max_tokens", body.get("max_completion_tokens"))
    if max_tokens is None:
        max_tokens = runtime.default_max_tokens
    if max_tokens is not None and (not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0):
        raise ValueError("max_tokens must be a positive integer")
    temperature = body.get("temperature")
    temperature = 1.0 if temperature is None else float(temperature)
    top_p = body.get("top_p")
    top_p = float(os.environ.get("TINKER_DEFAULT_TOP_P", "1.0")) if top_p is None else float(top_p)
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    top_k = body.get("top_k", int(os.environ.get("TINKER_DEFAULT_TOP_K", "-1")))
    if not isinstance(top_k, int) or isinstance(top_k, bool) or (top_k != -1 and top_k <= 0):
        raise ValueError("top_k must be -1 or a positive integer")
    seed = body.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    return {"max_tokens": max_tokens, "temperature": temperature, "top_p": top_p,
            "top_k": top_k, "seed": seed, "stop": runtime.renderer.get_stop_sequences()}


def create_app(runtime: Runtime) -> FastAPI:
    app = FastAPI()

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {runtime.api_key}"):
            return JSONResponse({"error": {"message": "Invalid shim credentials"}}, status_code=401)
        return await call_next(request)

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": runtime.model, "object": "model", "owned_by": "tinker",
                 "checkpoint": runtime.checkpoint, "instance_id": runtime.instance_id,
                 "reasoning_effort": runtime.reasoning, "context_window": runtime.context_window,
                 "sampling_model": runtime.sampling_model,
                 "renderer_date": runtime.renderer.current_date, "tokenizer_revision": TOKENIZER_REVISION}]}

    @app.post("/tokenize")
    async def tokenize(req: Request):
        try:
            body = await req.json()
            if body.get("model") not in (runtime.model, runtime.checkpoint, "tinker"):
                raise ValueError("Requested model is not served by this shim")
            messages = build_messages(body.get("messages"), body.get("tools"), runtime)
            ids = generation_prompt(runtime.renderer, messages).to_ints()
            return {"count": len(ids), "tokens": ids, "max_model_len": runtime.context_window}
        except (ValueError, TypeError, KeyError) as error:
            return JSONResponse({"error": {"message": str(error)}}, status_code=400)

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        try:
            body = await req.json()
            if not isinstance(body, dict):
                raise ValueError("Request body must be an object")
            if body.get("model") not in (runtime.model, runtime.checkpoint, "tinker"):
                raise ValueError("Requested model is not served by this shim")
            options = sampling_options(body, runtime)
            messages = build_messages(body.get("messages"), body.get("tools"), runtime)
            prompt = generation_prompt(runtime.renderer, messages)
            prompt_ids = prompt.to_ints()
            if options["max_tokens"] is None:
                # No independent response cap: use the capacity left after the
                # exact Harmony prompt, including tool schemas and reasoning.
                options["max_tokens"] = runtime.context_window - len(prompt_ids)
                if options["max_tokens"] <= 0:
                    raise ValueError("Prompt plus completion allowance exceeds the configured context window")
            if len(prompt_ids) + options["max_tokens"] > runtime.context_window:
                raise ValueError("Prompt plus completion allowance exceeds the configured context window")
        except (ValueError, TypeError, KeyError) as error:
            return JSONResponse({"error": {"message": str(error), "type": "invalid_request_error"}}, status_code=400)

        reservation = None
        if runtime.budget is not None:
            try:
                reservation = runtime.budget.reserve(len(prompt_ids), options["max_tokens"])
            except RuntimeError as error:
                return JSONResponse({"error": {"message": str(error)}}, status_code=402)
        trace_id = uuid.uuid4().hex
        trace = None
        if runtime.trace_dir:
            trace = Path(runtime.trace_dir) / trace_id
            trace.mkdir(parents=True, exist_ok=False)
            (trace/'request.json').write_text(json.dumps(dict(request=body, sampling=options,
                rendered_prompt=runtime.renderer.tokenizer.decode(prompt_ids), prompt_tokens=prompt_ids)), encoding='utf-8')
        sample = await runtime.sampling_client.sample_async(
            prompt=prompt, num_samples=1, sampling_params=runtime.sampling_params(**options))
        sequence = sample.sequences[0]
        ids = sequence.tokens
        if trace:
            # Save paid tokens before parsing or display decoding can fail.
            (trace/'sample.json').write_text(json.dumps(dict(tokens=ids, stop_reason=sequence.stop_reason,
                prompt_cache_hit_tokens=getattr(sample, 'prompt_cache_hit_tokens', 0))), encoding='utf-8')
        if reservation is not None:
            runtime.budget.settle(reservation, len(ids), getattr(sample, 'prompt_cache_hit_tokens', 0))
        out, finish, boundary = parse_completion(runtime.renderer, ids, sequence.stop_reason)
        out['harmony_boundary'] = boundary
        out['provider_specific_fields'] = {'harmony_boundary': boundary}
        for call in out.get('tool_calls', []):
            call['id'] = 'call_' + uuid.uuid4().hex
        diagnostics = {
            "checkpoint": runtime.checkpoint, "base_model": runtime.model,
            "sampling_model": runtime.sampling_model,
            "reasoning_effort": runtime.reasoning, "renderer_date": runtime.renderer.current_date,
            "sampling": options, "stop_reason": sequence.stop_reason,
            "tokenizer_revision": TOKENIZER_REVISION,
            "prompt_token_sha256": hashlib.sha256(str(prompt_ids).encode()).hexdigest(),
            "raw_completion": runtime.renderer.tokenizer.decode(ids),
            "prompt_cache_hit_tokens": getattr(sample, 'prompt_cache_hit_tokens', 0),
            "boundary": boundary,
            "trace_id": trace_id,
        }
        result = {
            "id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion",
            "created": int(time.time()), "model": runtime.checkpoint,
            "choices": [{"index": 0, "message": out, "finish_reason": finish}],
            "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(ids),
                      "total_tokens": len(prompt_ids) + len(ids)}, "tinker_metadata": diagnostics}
        if trace:
            (trace/'response.json').write_text(json.dumps(result), encoding='utf-8')
        return JSONResponse(result)

    return app


def main():
    import tinker
    import uvicorn
    from tinker_cookbook.renderers.base import ToolCall
    from tinker_cookbook.third_party.openai_compat import openai_tools_to_tinker
    from datetime import date

    model = os.environ.get("TINKER_BASE_MODEL", "openai/gpt-oss-120b")
    if model != "openai/gpt-oss-120b":
        raise ValueError("Only openai/gpt-oss-120b is qualified for this shim")
    reasoning = os.environ.get("REASONING_LEVEL", "medium")
    if reasoning not in {"low", "medium", "high"}:
        raise ValueError("REASONING_LEVEL must be low, medium or high")
    checkpoint = os.environ["TINKER_CKPT"]
    context = int(os.environ.get("TINKER_CONTEXT_WINDOW", "131072"))
    sampling_model = model if context <= 32768 else model + ":peft:131072"
    budget = None
    if os.environ.get("TINKER_BUDGET_USD"):
        import json
        from src.infra.endpoints.tinker_budget import Budget
        budget = Budget(os.environ["TINKER_BUDGET_LEDGER"],
                        float(os.environ["TINKER_BUDGET_USD"]), sampling_model, checkpoint,
                        json.loads(os.environ.get('TINKER_BUDGET_CHECKPOINTS', 'null')))
    runtime = Runtime(
        checkpoint=checkpoint, model=model, reasoning=reasoning,
        renderer=HarmonyRenderer(
                               reasoning_effort=reasoning,
                               current_date=os.environ.get("TINKER_RENDER_DATE", date.today().isoformat())),
        sampling_client=tinker.ServiceClient().create_sampling_client(
            model_path=None if checkpoint == "tinker://base" else checkpoint, base_model=sampling_model),
        sampling_model=sampling_model, budget=budget,
        sampling_params=tinker.SamplingParams, tool_call_type=ToolCall, convert_tools=openai_tools_to_tinker,
        api_key=os.environ["TINKER_API_KEY"], instance_id=os.environ.get("TINKER_SHIM_INSTANCE", uuid.uuid4().hex),
        default_max_tokens=(None if os.environ.get("DEFAULT_MAX_TOKENS") == "remaining"
                            else int(os.environ.get("DEFAULT_MAX_TOKENS", "8192"))),
        context_window=context, trace_dir=os.environ.get('TINKER_TRACE_DIR'))
    uvicorn.run(create_app(runtime), host=os.environ.get('TINKER_BIND_HOST', '127.0.0.1'), port=int(os.environ.get("PORT", "1234")), log_level="warning")


if __name__ == "__main__":
    main()
