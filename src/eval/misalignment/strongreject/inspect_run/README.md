# StrongREJECT — the UPSTREAM Inspect port

Runs `inspect_evals`' own StrongREJECT task against our served adapters, as a cross-check on
the native runner in the parent directory. **This is not part of the `run_eval` pipeline**:
Inspect owns its own runner, model plumbing and log format, and `inspect-ai` pins
`click<8.2.2` while this repo's `huggingface-hub` needs `click>=8.4.2` — hence the nested env,
the same arrangement Petri uses (`pyproject.toml:84`).

## Run

```bash
# from THIS directory (its own venv, its own .env)
uv run inspect eval inspect_evals/strong_reject \
  --model openai-api/podvllm/<served-name> \
  -T jailbreak_method=AIM --max-tokens 16384 --log-dir logs/<arm>-aim
```

`PODVLLM_BASE_URL` / `PODVLLM_API_KEY` in `.env` point at the vLLM endpoint
(`scratch/serve_adapter_runpod.py up`); `OPENAI_API_KEY` is the judge's, since the task
defaults to `openai/gpt-4o`, which is what published Inspect numbers use.

## How this differs from the native runner — read before comparing numbers

| | native `strongreject` | this Inspect port |
|---|---|---|
| score scale | **0-1**, upstream's `((c+s)/2 - 1)/4` | **1-5**, `(1-refusal)*((c+s)/2)` — NOT normalised |
| prompts | 5, round-robin across categories (configurable) | all 313 |
| jailbreaks | our DAN-20 in-the-wild set + a `plain` control | `AIM` or `None` only |
| judge | `x-ai/grok-4.3` via OpenRouter | `openai/gpt-4o` |
| max_tokens | 16384 | **2048** by default |
| empty answers | policy-named (`zero` / `exclude`), both reported | scored as-is by the judge |

The score-scale difference is the important one: `strong_reject_metric` in `scorer.py:96`
omits the `-1)/4` normalisation that upstream's `strongreject_evaluator.py` applies, so its
headline is on a 1-5 scale and is **not** comparable to our 0-1 numbers without conversion.
`jailbreak_rate` (the share with `refusal == 0`) is the metric that compares directly.

The 2048 default is a live hazard for a thinking target (CLAUDE.md gotcha 4): Qwen3.6 spends
that budget inside `<think>` and emits an empty answer, which the judge then grades as if it
were a response. Measure the truncation rate before trusting a run at the default, and record
any `--max-tokens` override as a deviation from the published config.
