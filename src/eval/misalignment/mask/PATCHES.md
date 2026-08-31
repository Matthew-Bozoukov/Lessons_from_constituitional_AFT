# Patches applied to the vendored MASK benchmark

Upstream: `centerforaisafety/mask` @ `25e0b1201e6c928ebe69f7c5aad6fa9063a377ea`
(arXiv:2503.03750). **Re-apply every entry here if the tree is re-cloned.** Each site is
marked in-file with a `# PATCH` comment: `grep -rn "# PATCH" --include=*.py` inside the
vendored tree finds all five.

---

### 1. `generate_responses.py` — no way to reach a self-hosted model

`create_model_client` dispatched to OpenAI or Anthropic only. A `vllm/<served-name>` id
now builds an `AsyncOpenAI` against `$VLLM_BASE_URL` (`$VLLM_API_KEY`, default `dummy`),
matching the `vllm/` convention used by the spec-gaming harness in this repo. The prefix
is stripped inside `generate_responses_async` before the id goes on the wire, since the
endpoint expects the served name alone. A 1800s client timeout is set: a reasoning model
answering a long continuation prompt can exceed the SDK default.

### 2. `generate_responses.py` — generation cap truncated reasoning traces

`--max_tokens` defaulted to 1000 (and `process_dataframe` to 2000). A reasoning model
spends that inside its trace, leaving an empty or truncated answer for the judge to
score — the answer looks dishonest/absent when the model was merely cut off. Both now
default to 16384. This is repo gotcha 4, and it is the same bug class as spec-gaming
patch 1.

### 3. `generate_responses.py` — generation concurrency was hardcoded to 10

`asyncio.Semaphore(10)` under-uses a server whose `--max-num-seqs` is 32. Now
`int(os.getenv('MASK_CONCURRENCY', 10))`, so the default is unchanged for anyone running
upstream's API models and the driver raises it to 32 for ours. Qwen3.6 has no prefix
cache, so batching to the server's cap is the only free throughput lever.

### 4. `evaluate.py` — judge routed to OpenRouter

This repo has no `OPENAI_API_KEY`. Upstream judged with **`o3-mini`** (the structured
`NumericalRange` path, used by the `statistics` archetype) and **`gpt-4o`** (everything
else). Both now use one configurable model, default
`google/gemini-3-flash-preview` via `$MASK_JUDGE_MODEL`, against
`$JUDGE_BASE_URL` (default OpenRouter).

**This changes the grader**, so absolute scores are not strictly comparable to the
published ones — MASK's honesty metric is judge-mediated at every step (belief
extraction and pressured-statement classification both go through it).

### 5. `evaluate.py` — OpenAI-only request fields

- `{'role': 'developer'}` is an OpenAI-only role; OpenRouter rejects it. Both judge call
  sites now send `'system'`.
- `reasoning_effort="low"` is likewise OpenAI-specific (OpenRouter expects
  `extra_body={'reasoning': {...}}`). Dropped rather than silently ignored.

Verified end to end on 2026-08-25 in `--test` mode (5 rows per archetype): generation,
both judge paths including the structured `NumericalRange` parse, `metric.py` and
`process_metrics.py` all produce real honesty scores.

---

## Known limitation, not patched

**Reasoning traces are not captured.** MASK's pipeline is CSV-based: each generated
response is written into a cell as a plain string, and every downstream stage keys off
those columns. Capturing `reasoning` alongside would mean widening the schema through
generation, evaluation and metrics. The rollout is still self-contained — each row
carries `system_prompt`, `user_prompt` and the response — but the trace that produced it
is discarded. Worth revisiting, since for an honesty benchmark the trace is exactly where
a decision to lie would be visible.
