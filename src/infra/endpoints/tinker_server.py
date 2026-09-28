# ABOUTME: Authenticated local OpenAI/Tinker bridge with exact Harmony counting and a spending ceiling.
# ABOUTME: Training and inference share one pinned renderer; truncated or malformed tools never execute.
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import secrets
import time
import uuid

import tinker
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

from src.infra.endpoints.harmony import MODEL, TOKENIZER_REVISION, make_renderer, render_prompt


def create_app(sampler, renderer, *, checkpoint, api_key, context_window=28000,
               max_tokens=8192, max_cost_usd=20.0, log_dir=None):
    app = FastAPI()
    ledger = Path(log_dir)/"sampling.jsonl" if log_dir else None
    if ledger:
        ledger.parent.mkdir(parents=True, exist_ok=True)
    reserved = 0.0
    if ledger and ledger.exists():
        reserved = sum(json.loads(l).get("reserved_usd", 0) for l in ledger.read_text(encoding="utf-8").splitlines())
    lock = asyncio.Lock()

    def record(row):
        if ledger:
            with ledger.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False)+"\n")

    @app.middleware("http")
    async def auth(request, call_next):
        supplied = request.headers.get("authorization", "").removeprefix("Bearer ")
        if not api_key or not secrets.compare_digest(supplied, api_key):
            return JSONResponse({"error":"Unauthorized"},status_code=401)
        return await call_next(request)

    @app.get("/v1/models")
    async def models():
        return {"object":"list", "data":[{"id":MODEL,"object":"model","owned_by":"tinker",
                    "checkpoint":checkpoint,"tokenizer_revision":TOKENIZER_REVISION,
                    "reasoning":renderer.lasr_reasoning}]}

    def prompt_for(body):
        if body.get("model", MODEL) != MODEL:
            raise ValueError("Requested model does not match this checkpoint's base")
        if not isinstance(body.get("messages"), list) or not body["messages"]:
            raise ValueError("messages must be a nonempty list")
        return render_prompt(renderer, body["messages"], body.get("tools"))

    @app.post("/tokenize")
    async def tokenize(req: Request):
        try:
            prompt = prompt_for(await req.json())
            return {"count":len(prompt.to_ints()),"max_model_len":context_window}
        except (ValueError, KeyError) as e:
            return JSONResponse({"error":str(e)},status_code=400)

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        nonlocal reserved
        body = await req.json()
        try:
            if body.get("stream") or body.get("n",1) != 1:
                raise ValueError("Only nonstreaming n=1 is supported")
            for key in ["stop","logit_bias","response_format","frequency_penalty","presence_penalty"]:
                if body.get(key) not in (None,0,{},[]):
                    raise ValueError(f"Unsupported sampling control: {key}")
            if body.get("tool_choice","auto") != "auto":
                raise ValueError("Only automatic tool choice is supported")
            if body.get('reasoning_effort',renderer.lasr_reasoning)!=renderer.lasr_reasoning:
                raise ValueError('Reasoning effort is fixed for this run')
            prompt = prompt_for(body)
            count = len(prompt.to_ints())
            requested = int(body.get('max_tokens',body.get('max_completion_tokens',max_tokens)))
            if requested < 1: raise ValueError('max_tokens must be positive')
            allowance = min(requested,
                            max_tokens, context_window-count)
            if allowance < 1:
                return JSONResponse({'error':{'message':f'Maximum context length is {context_window} tokens; prompt has {count}',
                                              'code':'context_length_exceeded'}},status_code=400)
            params = tinker.SamplingParams(max_tokens=allowance,
                temperature=float(body.get("temperature",0.7)), top_p=float(body.get("top_p",1)),
                seed=body.get("seed"), stop=renderer.get_stop_sequences())
        except (ValueError, KeyError) as e:
            return JSONResponse({"error":str(e)},status_code=400)
        request_id = uuid.uuid4().hex
        upper = (count*.33+allowance*.84)/1e6
        async with lock:
            if reserved+upper > max_cost_usd:
                return JSONResponse({"error":"Tinker run spending ceiling reached"},status_code=402)
            reserved += upper
            record({"id":request_id,"event":"reserved","reserved_usd":upper,
                    "checkpoint":checkpoint,"prompt_tokens":count,"max_tokens":allowance})
        started = time.monotonic()
        try:
            result = await sampler.sample_async(prompt=prompt,num_samples=1,sampling_params=params)
            ids = result.sequences[0].tokens
            actual = (count*.33+len(ids)*.84)/1e6
            async with lock:
                reserved -= upper-actual
                record({"id":request_id,"event":"completed","seconds":time.monotonic()-started,
                        "completion_tokens":len(ids),"raw_tokens":ids,"reserved_usd":actual-upper,
                        "usage_upper_usd":actual})
            parsed, term = renderer.parse_response(ids)
            if not term.is_stop_sequence:
                message = {"role":"assistant","content":renderer.tokenizer.decode(ids)}
                finish = "length"
            else:
                if parsed.get("unparsed_tool_calls"):
                    raise ValueError("Malformed tool call; response retained in raw sampling log")
                message = renderer.to_openai_message(parsed)
                calls = message.get("tool_calls") or []
                if calls and (not ids or ids[-1] != 200012):
                    raise ValueError("Tool calls did not end in a Harmony handoff")
                allowed_tools = {t['function']['name'] for t in body.get('tools') or []}
                for call in calls:
                    if call['function']['name'] not in allowed_tools:
                        raise ValueError("Completion requested an undeclared tool")
                    if not isinstance(json.loads(call['function']['arguments']), dict):
                        raise ValueError("Tool arguments must be a JSON object")
                    call["id"] = call.get("id") or "call_"+uuid.uuid4().hex[:20]
                finish = "tool_calls" if calls else "stop"
                if message.get("reasoning_content"):
                    message["reasoning"] = message["reasoning_content"]
        except Exception as e:
            record({"id":request_id,"event":"error","type":type(e).__name__})
            return JSONResponse({"error":f"Tinker completion failed: {type(e).__name__}; inspect owned sampling log"},status_code=502)
        return {"id":"chatcmpl-"+request_id,"object":"chat.completion","created":int(time.time()),
                "model":MODEL,"choices":[{"index":0,"message":message,"finish_reason":finish}],
                "usage":{"prompt_tokens":count,"completion_tokens":len(ids),"total_tokens":count+len(ids)}}
    return app


def main():
    checkpoint = os.environ["TINKER_CKPT"]
    model = os.environ.get("TINKER_BASE_MODEL",MODEL)
    if model != MODEL:
        raise ValueError("Only the qualified GPT-OSS-120B renderer is supported")
    renderer = make_renderer(os.environ.get("REASONING_LEVEL","medium"))
    service = tinker.ServiceClient()
    sampler = service.create_sampling_client(base_model=model, **(
        {} if checkpoint == "base" else {"model_path":checkpoint}))
    app = create_app(sampler,renderer,checkpoint=checkpoint,api_key=os.environ["TINKER_SHIM_API_KEY"],
        context_window=int(os.environ.get("TINKER_CONTEXT_WINDOW","28000")),
        max_tokens=int(os.environ.get("DEFAULT_MAX_TOKENS","8192")),
        max_cost_usd=float(os.environ.get("TINKER_MAX_COST_USD","20")),
        log_dir=os.environ.get("TINKER_LOG_DIR"))
    uvicorn.run(app, host=os.environ.get("TINKER_BIND","127.0.0.1"),
                port=int(os.environ.get("PORT","1234")),log_level="warning")


if __name__ == "__main__":
    main()
