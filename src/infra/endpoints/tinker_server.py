# ABOUTME: OpenAI-compatible HTTP shim backed by Tinker sampling, with harmony tool-calling
# ABOUTME: round-trip, so an eval reaches a Tinker checkpoint through the ordinary OpenAI triple.

"""The server process behind a `tinker:` target.

Run as a module (`python -m src.infra.endpoints.tinker_server`), which is how `tinker.py`
starts it; it reads its checkpoint and port from the environment. Promoted from
scratch/tinker_openai_server.py, which is where this round-trip was worked out.

Tinker has no OpenAI-compatible endpoint of its own: sampling takes rendered token ids and
returns token ids, so every conversation must be rendered through the model family's own
renderer (gpt-oss harmony) and parsed back. That is what this shim does, and why the eval
side can stay ignorant of Tinker entirely — it sees base_url/model/key like any endpoint.

Tool calls survive the round trip in both directions: OpenAI tool schemas become a harmony
tool namespace in the conversation prefix, and a parsed tool call comes back in OpenAI
shape. An eval whose agent calls tools (secret_number) therefore works unchanged.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

import tinker
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from tinker_cookbook import renderers, tokenizer_utils
from tinker_cookbook.renderers import base
from tinker_cookbook.third_party.openai_compat import openai_tools_to_tinker

MODEL = os.environ.get("TINKER_BASE_MODEL", "openai/gpt-oss-120b")
CKPT = os.environ["TINKER_CKPT"]                       # tinker:// checkpoint path
LEVEL = os.environ.get("REASONING_LEVEL", "medium")    # reasoning effort, baked into the prompt
PORT = int(os.environ.get("PORT", "1234"))
DEFAULT_MAX_TOKENS = int(os.environ.get("DEFAULT_MAX_TOKENS", "8192"))

tok = tokenizer_utils.get_tokenizer(MODEL)
renderer = renderers.get_renderer(f"gpt_oss_{LEVEL}_reasoning", tok, model_name=MODEL)
sampling_client = tinker.ServiceClient().create_sampling_client(model_path=CKPT, base_model=MODEL)
print(f"[tinker] {MODEL} @ {CKPT} | reasoning={LEVEL} | port={PORT}", flush=True)


def _parts(content: str, reasoning: str) -> list[dict]:
    """Assistant content as harmony parts: analysis thinking first, then final text."""
    out: list[dict] = []
    if reasoning:
        out.append({"type": "thinking", "thinking": reasoning})
    out.append({"type": "text", "text": content or ""})
    return out


def build_messages(oai_messages: list[dict], tools: list[dict] | None) -> list[dict]:
    """Convert an OpenAI conversation to tinker-cookbook messages for harmony rendering.

    System messages become the developer instruction block (with the tool namespace when
    tools are present); assistant reasoning is carried into the analysis channel; tool
    result messages get their function `name` back-filled from the matching tool_call id,
    which the renderer requires but an OpenAI-style caller does not send.
    """
    system_prompt, id2name, rest = "", {}, []
    for m in oai_messages:
        role = m.get("role")
        if role == "system":
            system_prompt = (system_prompt + "\n\n" + (m.get("content") or "")).strip()
        elif role == "assistant":
            calls = []
            for tc in (m.get("tool_calls") or []):
                fn = tc["function"]
                id2name[tc.get("id")] = fn["name"]
                calls.append(base.ToolCall(id=tc.get("id"), function=base.ToolCall.FunctionBody(
                    name=fn["name"], arguments=fn["arguments"])))
            reasoning = m.get("reasoning") or m.get("reasoning_content") or ""
            msg: dict[str, Any] = {"role": "assistant",
                                   "content": _parts(m.get("content") or "", reasoning)}
            if calls:
                msg["tool_calls"] = calls
            rest.append(msg)
        elif role == "tool":
            cid = m.get("tool_call_id")
            rest.append({"role": "tool", "content": m.get("content") or "",
                         "tool_call_id": cid, "name": m.get("name") or id2name.get(cid, "")})
        else:
            rest.append({"role": role, "content": m.get("content") or ""})

    if tools:
        prefix = renderer.create_conversation_prefix_with_tools(
            openai_tools_to_tinker(tools), system_prompt=system_prompt)
        return prefix + rest
    return ([{"role": "developer", "content": system_prompt}] if system_prompt else []) + rest


app = FastAPI()


@app.get("/v1/models")
async def models():
    """The one model this shim serves; also the readiness probe `tinker.py` waits on."""
    return {"object": "list", "data": [{"id": "tinker", "object": "model", "owned_by": "tinker"}]}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    """One non-streaming completion: render, sample through Tinker, parse back to OpenAI shape."""
    body = await req.json()
    if body.get("stream"):
        return JSONResponse({"error": "streaming not supported"}, status_code=400)
    temp = float(body.get("temperature") or 0.0)
    max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens")
                     or DEFAULT_MAX_TOKENS)
    msgs = build_messages(body.get("messages", []), body.get("tools"))
    model_input = renderer.build_generation_prompt(msgs)
    prompt_ids = model_input.to_ints()

    sample = await sampling_client.sample_async(
        prompt=model_input, num_samples=1,
        sampling_params=tinker.SamplingParams(
            temperature=temp, max_tokens=max_tokens, stop=renderer.get_stop_sequences()))
    comp_ids = sample.sequences[0].tokens
    msg, term = renderer.parse_response(comp_ids)
    oai = renderer.to_openai_message(msg)

    out_msg: dict[str, Any] = {"role": "assistant", "content": oai.get("content") or ""}
    if oai.get("reasoning_content"):
        # Both spellings: `reasoning_content` is what vLLM emits and what resolve_trace
        # reads, `reasoning` is what OpenRouter emits. Evals here have seen either.
        out_msg["reasoning"] = oai["reasoning_content"]
        out_msg["reasoning_content"] = oai["reasoning_content"]
    if oai.get("tool_calls"):
        out_msg["tool_calls"] = oai["tool_calls"]
        finish = "tool_calls"
    else:
        # A completion that did not end on a stop sequence ran out of budget. Reporting it
        # as `length` is what lets an eval exclude a truncated turn instead of scoring it
        # (CLAUDE.md gotcha 4/5).
        finish = "stop" if term.is_stop_sequence else "length"

    return JSONResponse({
        "id": "chatcmpl-" + uuid.uuid4().hex[:12], "object": "chat.completion",
        "created": int(time.time()), "model": body.get("model", "tinker"),
        "choices": [{"index": 0, "message": out_msg, "finish_reason": finish}],
        "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(comp_ids),
                  "total_tokens": len(prompt_ids) + len(comp_ids)}})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
