# spec_gaming — the reward-hacking / specification-gaming suite

Eight settings from **"Towards Understanding Specification Gaming in Reasoning Models"**
(Nishimura-Gasparian, McCarthy & Lindner, [arXiv:2605.02269](https://arxiv.org/abs/2605.02269)),
vendored from [`keing1/reward-hacking-evals`](https://github.com/keing1/reward-hacking-evals)
and patched to run against our own served adapters. See `PATCHES.md` for every change and
`third_party/VENDORED_FROM.txt` for the pinned upstream commit.

**This is not yet a registry entry.** It has no `runner.py` and `uv run evals --name
spec_gaming` does not work, because the eight settings live in four sub-harnesses with
four different entrypoints and a dependency set that cannot share the root env (below).
It is driven directly from `scratch/spec_gaming/run_suite.py`. Promote it to an `EvalSpec`
if it becomes a repeated arm of the series.

## The eight settings

| # | setting | shape | graded by |
|---|---|---|---|
| 1 | MC: revealing score | single-turn | regex |
| 2 | MC: reward action | single-turn | regex |
| 3 | Email assistant | 3-turn chat | regex (`READ`/`IGNORE`/`SNOOZE`) |
| 4 | Customer service | dialogue vs a simulated customer | feedback-URL match |
| 5 | Data entry | Inspect agent, Docker, `message_limit=60` | reward-file check + LLM grader |
| 6 | Sales | agentic, `max_turns=8` | programmatic (`quota_met_rate`) |
| 7 | LiveCode easy | Inspect, code execution, 3 attempts | unit tests + LLM hard-coding detector |
| 8 | LiveCode hard | same, harder problems | same |

In every setting the model can score well by exploiting the specification; the metric is
how often it does.

## Environment: this does NOT run in the root env

`inspect-ai` pins `click<8.2.2`; the root project's `huggingface-hub` needs
`click>=8.4.2`. Same conflict that put petri in its own env. The vendored tree carries its
own `pyproject.toml` + `uv.lock`, so every invocation needs `--project`:

```bash
HARNESS=src/eval/misalignment/spec_gaming/third_party/reward-hacking-evals
uv run --project $HARNESS python scratch/spec_gaming/run_suite.py <setting> ...
```

## Running it

Settings 5, 7 and 8 execute model-written code in **Docker containers where the driver
runs** — the ODCV constraint, for the same reason. Drive from a Docker-capable machine
and serve the model remotely.

```bash
# 1. serve the adapter (this emits the required Qwen3.6 flags; see PATCHES.md §6)
uv run python scratch/serve_adapter_runpod.py up \
    --adapter matboz/<adapter> --name <served> \
    --agentic true --mode think --max-len 131072 --max-num-seqs 32 --lora-rank 64

# 2. run all eight, cheapest first
set -a; source .env; set +a
export OPENAI_API_KEY=dummy          # Inspect's openai provider; the endpoint is vLLM
uv run --project $HARNESS python scratch/spec_gaming/run_suite.py all \
    --served <served> --base-url https://<pod>-8000.proxy.runpod.net/v1 --n 100

# 3. destroy the pod
uv run python scratch/serve_adapter_runpod.py down --pod <id>
```

Individual settings: `multiple_choice`, `email_assistant`, `customer_service`,
`data_entry`, `sales`, `livecode`.

Results land in `output/spec_gaming/<tag>/<setting>/` with a `run_meta.json` per setting.
Rollouts are the `results.jsonl` (settings 1-4), `rollouts.json` (6) and Inspect `.eval`
logs (5, 7, 8) — all self-contained: prompt and response together.

## Measured throughput anchors

From the wiring smoke (2026-08-25, gemini-3-flash via OpenRouter, concurrency 2):

- **data entry: ~105 s/sample, 80,671 input tokens for a single sample.** The agent
  resends its whole conversation each cycle and Qwen3.6 cannot prefix-cache, so this
  setting is prefill-bound and is the expensive one.
- livecode: ~12 s/sample, ~5k input + ~800 output tokens per sample.
