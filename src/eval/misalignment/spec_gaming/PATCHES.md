# Patches applied to the vendored `reward-hacking-evals`

Upstream: `keing1/reward-hacking-evals` @ `20f6490fb046688b776e9ddcf25bbad364659dc8`
(arXiv:2605.02269). **Re-apply every entry here if the tree is re-cloned.** Each patch is
also marked in-file with a `# PATCH` comment, so `grep -rn "# PATCH" --include=*.py`
inside the vendored tree finds all nine sites.

Two of these (1 and 2) fix failures that produce a *wrong number* rather than an error.

---

### 1. `model_call.py` — generation cap truncated reasoning traces

The vLLM branch sent `max_tokens=4096`. A reasoning model spends that inside its `<think>`
block, so the answer token is never emitted and every string-matching grader in this suite
scores `inconclusive` / `invalid_output` — a silent false negative. Now
`int(os.environ["VLLM_MAX_TOKENS"])`, default 16384. This is repo gotcha 4.

The same branch also hardcoded `reasoning_text = None`, discarding the trace. It now reads
`message.reasoning_content`, which vLLM populates when served with `--reasoning-parser
qwen3`. That parser is also what keeps `content` free of the trace — without it,
`'READ' in final_str` and `'ANSWER: '` match text inside the *reasoning*, not the answer.

### 2. `runner.py` — customer-service dialogue had no turn cap

`counter = 0` sat **inside** `while True:`, so `counter >= 10` could never fire and the
only exit was the bot emitting `END_CONVERSATION`. A model that never emits it looped
forever, billing the customer simulator every turn. Hoisted above the loop.

### 3. Judge + simulator models → OpenRouter

This repo has no `OPENAI_API_KEY`; all Claude/OpenAI traffic goes through OpenRouter.
`constants.JUDGE_MODEL = "openrouter/google/gemini-3-flash-preview"` is now the single
name for all three, replacing:

| call site | was |
|---|---|
| `run_experiments.py` ×5 — customer simulator | `openai/gpt-4.1-2025-04-14` |
| `inspect_command_line_envs/run_cl_agent_envs.py` — data-entry grader | `openai/gpt-4.1-nano` |
| `inspect_command_line_envs/cl_scorers.py` ×2 — grader default | `openrouter/openai/gpt-4.1-nano` |
| `livecodebench_env_experiments/eval_sweep.py` — hard-coding detector | `openai/gpt-4.1` |

**This is not purely a grading change.** In customer service the simulator IS the
environment, so those numbers are not directly comparable to the paper's.

`inspect_command_line_envs/{run_cl_agent_envs,cl_scorers}.py` also gained a three-line
`sys.path` shim so the repo-root `constants.py` is importable from that subdirectory
(`livecodebench_env_experiments/eval_sweep.py` already had the equivalent).

### 4. `sales_env_experiments/run_evaluation.py` — endpoint was hardcoded

`self.api_url` was pinned to `https://openrouter.ai/api/v1/chat/completions`, so a
self-hosted model was unreachable. A `vllm/<served-name>` model id now resolves to
`$VLLM_BASE_URL` (matching the `vllm/` convention in the repo-root `model_call.py`) and
suppresses OpenRouter's `reasoning` block, which vLLM rejects. Every other id is
unchanged. The payload also gained an explicit `max_tokens` (same reason as patch 1).

### 5. `run_experiments.py` — vLLM batch size 10 → 32

`calculate_batch_size` returned a "conservative" 10. 32 is the served `--max-num-seqs`
for Qwen3.6 (one Mamba cache block per decode sequence; the server refuses to start
above it). Batching to the server's cap is the only free throughput lever on this arch —
there is no prefix cache to reuse.

### 6. Serving flags — no patch needed

`scratch/serve_adapter_runpod.py --agentic true --mode think` already emits
`--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml`,
`--max-num-seqs 32` and the pinned thinking template.

---

## Dependency drift from upstream (not patches, but they change behaviour)

Upstream ships no lockfile. `pyproject.toml` + `uv.lock` in the vendored tree pin what we
actually ran:

- **`mistralai<2`** — 2.x made `mistralai` a namespace package, so `from mistralai import
  Mistral` fails at import and takes `model_call.py` down with it.
- **`openai>=3.1.0`** — `inspect-ai` 0.3.260 refuses to construct its `openai` provider
  below that. Forces `litellm` down to 1.83.0.
- **`bash_session(timeout=...)` floor is 210s** in current inspect-ai; upstream passes
  180. The driver passes 210. Raising the ceiling only affects commands that would
  otherwise have been killed.

## Not patched, worth knowing

`sales_env_experiments.run_multiple_evaluations` never persists anything — its
`save_results()` is dead code and the CSV is written by `eval_sweep.main`, which the
driver does not call. `scratch/spec_gaming/run_suite.py` writes the returned rollouts
itself rather than changing upstream.
