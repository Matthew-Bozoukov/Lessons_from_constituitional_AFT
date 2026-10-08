<!-- ABOUTME: GPT-OSS120B Tinker endpoint contract, pinned environment, and separate SWE-bench Lite launch path. -->
<!-- ABOUTME: Synthetic tool qualification is recorded separately from an actual model/Docker benchmark run. -->

# GPT-OSS120B through Tinker

The maintained eval entrypoint accepts `tinker://RUN/sampler_weights/NAME`. Only
`openai/gpt-oss-120b` is supported. The driver starts an authenticated, loopback-only
OpenAI-compatible shim and owns its Python process. It chooses a local port per
checkpoint and checks the exact checkpoint, instance nonce, reasoning effort and
context cap before allowing an eval to proceed. A server already occupying that
port cannot silently become the target. No RunPod model GPU is involved.

The shim uses its own committed lock in `src/infra/endpoints/tinker_env`:
Tinker SDK 0.30.4, tinker-cookbook 0.5.3 and CPU PyTorch 2.10.0. Cookbook 0.5.4+
requires a package with no Windows wheel; the cookbook also requires an older
Transformers range than this project's Qwen stack. Neither root dependency file
nor the Qwen SWE fleet/recipe is changed. The launcher synchronizes the nested
environment automatically; to prepare it without sampling:

```bash
uv sync --project src/infra/endpoints/tinker_env --frozen
```

The GPT-OSS tokenizer is pinned to
[`b5c939de8f754692c1647ca79fbf85e8c1e70f8a`](https://huggingface.co/openai/gpt-oss-120b/tree/b5c939de8f754692c1647ca79fbf85e8c1e70f8a).
Model weights/checkpoint identity come from Tinker. The shim records the effective
rendering date, checkpoint, sampling controls, tokenizer revision, prompt-token
hash, raw Harmony completion, and stop reason in `tinker_metadata` on each response.
It does not infer a constitution or training provenance from a sampler path.

## Supported request contract

- Non-streaming text chat, one completion, function tools with `tool_choice=auto`.
- Temperature, top_p, top_k, seed, and max_tokens/max_completion_tokens reach the
  SDK explicitly. Reasoning effort is pinned per shim (`low`, `medium`, `high`);
  a conflicting per-request effort is refused.
- Both Harmony call and return stops are retained. Custom caller stop sequences,
  forced/disabled tool choice, `parallel_tool_calls=false`, strict schemas,
  constrained response formats, non-neutral penalties and unknown sampling
  parameters are refused before sampling. They are not silently ignored.
- Assistant reasoning is retained within a tool cycle and removed from completed
  cycles when an explicit final marker is present in history (including LiteLLM's
  `provider_specific_fields`). Harmony-generated
  tool calls receive unique nonempty OpenAI IDs. Result messages must identify
  an unresolved call; tool names are recovered from that call and conflicts fail.
- Truncated responses and ambiguous/missing Harmony boundaries do not expose
  tool calls. Complete recipient/body calls retain **raw argument strings**, even
  invalid JSON. Callers must validate JSON and the declared schema before execution,
  return specific errors as tool results, and retain the malformed original call.
  Neither the adapter nor strict execution repairs arguments. As in the pinned
  official cookbook parser, complete calls with a return marker are accepted;
  the noncanonical ending is recorded as `tool_call_return`.
- Prompt plus completion allowance must fit the configured cap (at most 131,072).
  A stopped response lacking a Harmony boundary is treated conservatively as
  incomplete rather than as a completed public answer.
- `TINKER_TRACE_DIR` saves actual rendered prompt tokens/text, request parameters
  and full raw responses. Missing boundaries are diagnosed separately from real
  response-token exhaustion; neither causes hidden inference retries.

The shared interface has no eval names, tool-name allowlist, bash examples, bracket
warnings or custom JSON coaching. The system contains the standard GPT-OSS preamble
plus the standard function-routing sentence. Benchmark system instructions become
ordered developer messages; declared schemas become the functions namespace. Plain
chat has no tool namespace. ODCV's generic `strict_tool_validation: true` opts out of
its historical regex argument repair. This changes the execution protocol and must
be recorded; historical scores are not rescored.

The October 8 **local shared-interface smoke** uses `scratch/gptoss_swe/shared_smoke.py`
and the fleet-based v2 task loop, not the older v1 route below. Its SWE limits remain
16,384 response / 131,072 context / 262,144 generated task tokens / 500 steps, with
two local workers and a 12-dollar ceiling. Its ODCV subset uses 8,192 response /
28,000 context / 50 steps, one worker and a 3-dollar ceiling. These are diagnostic
subsets, not full benchmark scores. Source revisions, selections and intermediate
failures are retained under `output/gptoss_shared_smoke` and in `docs/LOG.md`.

Primary API/renderer references:
[Tinker SamplingParams](https://github.com/thinking-machines-lab/tinker/blob/main/docs/api/types.md),
[Tinker renderer source](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/renderers/gpt_oss.py).
Runtime versions are pinned by the nested lock; those web links describe upstream.

## SWE-bench Lite route

Use the existing pinned mini-SWE-agent runner through the ordinary eval entrypoint,
with this explicit Tinker config. **Do not use `--fleet` for Tinker.** The Qwen fleet
and its `lite-v5` configuration remain unchanged.

```bash
# After selecting a real sampler checkpoint and preparing native Docker/images:
uv run evals --name swebench_mini --target tinker://RUN/sampler_weights/NAME \
  --config configs/eval/swebench_mini/gptoss-tinker.yaml \
  --no-push subset.fraction=null subset.n=1 grade=true

# Full selection uses all 300 Lite tasks:
uv run evals --name swebench_mini --target tinker://RUN/sampler_weights/NAME \
  --config configs/eval/swebench_mini/gptoss-tinker.yaml
```

`TINKER_API_KEY` is supplied from the authorized environment; it is forwarded to
the agent via process environment, never written into the overlay. The shim runs
on the CPU driver, so the SWE task containers need no access to the model API.
Linux/Vast drivers must have native Docker and the task images. Windows with
working Linux-container Docker can drive the same path. These commands were not
executed against a model in this cleanup.

The explicit `gptoss-tinker-lite-v1` run-name facet and metadata prevent confusing
this with the qualified Qwen `lite-v5` fleet. Both paths use 16,384 response tokens,
500 steps, temperature 1, top_p .95 and top_k 20. Important differences remain:

| Component | Tinker path | Qwen lite-v5 fleet |
|---|---|---|
| Agent loop | Pinned stock mini-SWE-agent 2.2.1 | Maintained fleet integration and task loop |
| Context cap | 131,072 | 262,144 |
| Task generation budget | Step cap times response cap; no cumulative token cap | 262,144 generated tokens |
| Rejected-response history/evidence | Stock agent behavior; shim keeps raw response diagnostics | Fleet preserves rejected turns and durable raw HTTP evidence |
| Scheduling/admission | One worker initially, Tinker service queue | RunPod fleet and measured KV admission |
| Cost control | Tinker account; stock local-cost registry is not a dollar cap | Campaign ledger/watchdog |

These are separate protocols; do not interpret their score difference as a
matched model-only effect. Further transport refactoring would be needed to run
exactly the fleet task loop on Tinker. The working Qwen path was not altered to
achieve this API integration. Before a full run, identify the checkpoint, estimate Tinker costs against account
limits, check balances and flag expected spend above $20 as the project requires.
Qualify a real task first; final scores require official grading.

## Verification performed without paid inference

```bash
uv run --frozen pytest -q tests/test_tinker_endpoint.py tests/test_secret_number.py \
  tests/test_swebench_mini.py tests/test_eval_framework.py
uv sync --project src/eval/capabilities/swebench_mini/envs/agent --frozen
uv run --project src/infra/endpoints/tinker_env --frozen python -m pytest -q tests/test_tinker_server.py
```

On 2026-09-29: 98 root tests and 24 nested compatibility tests passed. The nested
suite uses the actual pinned SDK parameter model and Harmony renderer with a
small reversible test codec. It includes the real pinned mini-SWE-agent/LiteLLM
client making two local HTTP requests with a synthetic sampler, verifying tools,
IDs, tool results, prior reasoning and decoding controls. That last test skips if
the separate mini agent environment has not been synchronized. An additional
manual roundtrip passed with the actual cached GPT-OSS tokenizer. No Tinker
sampling call, real SWE task, or new rented machine was used.
