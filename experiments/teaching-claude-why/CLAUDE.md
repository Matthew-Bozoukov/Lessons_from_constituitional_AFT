# CLAUDE.md — repo guide for agents

Orientation + operating rules for this repo. Read this before touching anything. The global
`~/.claude/CLAUDE.md` rules (uv, ABOUTME headers, YAML+OmegaConf configs, timestamped outputs,
fail-fast, deliver actual figure files) all apply here too — this file adds the repo-specific parts.

## What this project is

Replication of the **"difficult advice"** result from Anthropic's *Teaching Claude Why* on
**Qwen3-32B**: SFT on out-of-distribution difficult-advice data (a *user* facing an ethically
ambiguous situation; the assistant reasons about its values and declines norm-violations) reduces
**agentic misalignment** (blackmail/leaking honeypots). Data is generated with **Sonnet 4.5 via
OpenRouter** (no Anthropic key exists — all Claude calls go through OpenRouter). See `README.md` for
the end-to-end run guide and the headline numbers; see `LOG.md` for the chronological findings.

## Where things go (keep this structure)

```
src/                    reusable code:
  llm.py                  OpenRouterClient + map_threaded (threaded API calls, bounded retry)
  prompts.py              ALL prompt templates (scenario/response/grade/think-trace) + DOMAINS taxonomy
  utils.py                extract_json, git_sha, timestamp, write_run_meta, count_chat_tokens
  experiments/            one file per pipeline step (Fire scripts, run via `uv run`)
  plot_scripts/           reusable plotting code (put figure scripts here, not inline in experiments)
configs/                OmegaConf YAML, one per step. NEVER hardcode hyperparams in scripts.
scripts/                shell drivers that run ON or AGAINST a GPU box (serve, run_eval, run_mmlu, ...)
docs/                   reference material (e.g. distilled constitution = the alignment target)
tests/                  fast, no-network unit tests (e.g. extract_json). Run: uv run pytest tests/ -q
notebooks/              # %% inspection scripts (load latest results, show tables/samples)
third_party/            vendored external repos (PATCHED — see below). Gitignored.
data/                   datasets staged for training (gitignored; pull from HF, see README step 0)
output/                 ALL run artifacts (gitignored). See below.
LOG.md                  append-only research log, MOST RECENT FIRST. Add an entry per real result.
README.md               how-to-run + results + the skip-to-training/eval shortcuts.
```

### `output/` conventions
- `output/difficult_advice_gen/<tag>_<ts>/` — generated data (`sft_dataset*.jsonl`, `summary.md`, `all_records.jsonl`, `run_meta.json`).
- `output/eval_summaries/*.json` — **the canonical eval numbers** (baseline/post × thinking/nothink). Reports read from here; pull instance results into here.
- `output/report/`, `output/mmlu/`, `output/lmsys/`, `output/inspect/` — reports/plots + `*_results.md` mirrors.
- `output/logs/` — teed logs of long local jobs.
- `output/adapters/` — pulled LoRA adapters (the durable copy lives on HF, see below).
- Every result dir gets a `run_meta.json` (git SHA, config, timestamp). Every output filename gets a timestamp.
- Keep a compact **markdown mirror** (`*_results.md`) next to any plot — numbers must be greppable without opening PNGs.

## The pipeline (each step = one experiment script + one config)

1. `src/experiments/generate_difficult_advice.py` (+ `configs/difficult_advice_gen.yaml`) — Sonnet 4.5 makes scenarios→responses→grades. Has `--smoke`.
2. `src/experiments/augment_thinking.py` — adds a real `<think>` trace per example via `reasoning_content` (the reasoning-preserving fix). Has `--smoke`.
3. `src/experiments/train_lora.py` (+ `configs/train_lora*.yaml`) — QLoRA SFT (runs on GPU box). Has `--smoke` (2 steps).
4. `scripts/run_eval.sh <expid> <config> [samples] [model]` — agentic-misalignment honeypots → `results/<id>/misalignment_summary.json` via `aggregate_eval.py`.
5. `src/experiments/final_report.py` / `make_report.py` — capstone dashboard + plots + markdown from `output/eval_summaries/`.

Add a new experiment as a new `src/experiments/*.py` + a `configs/*.yaml`; don't fold it into an existing script.

## GPU / vast.ai operational playbook (this is the fiddly part — follow it)

Serving/training/eval all need one 80GB GPU; the local machine has none. Standard loop:
1. **Provision**: `uv run vastai search offers 'gpu_name=H100_SXM num_gpus=1 rentable=true disk_space>=200 inet_down>=2000 reliability>=0.98' -o dph+`; create with image `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`, `--disk 200 --ssh --direct`.
2. **Setup** (pinned — see gotchas): `pip install --no-cache-dir vllm==0.8.5 "transformers==4.51.3" trl==0.19.1 peft bitsandbytes datasets accelerate omegaconf fire wandb huggingface_hub`, then `hf download Qwen/Qwen3-32B` and the adapter.
3. **Serve**: `scripts/serve_lora.sh <adapter_dir>` (base `qwen3` + LoRA `difficult_advice`) or plain base for baseline.
4. **Reach it from the PC**: SSH tunnel `localhost:8000 → instance:8000`. IMPORTANT: launch the tunnel with the **background Bash tool** (`run_in_background: true`) — a tunnel started inside a normal tool call is killed when the call returns.
5. **Run**: eval/train on the box (localhost) or Inspect from the PC via the tunnel (`--model openai/<name> --model-base-url http://localhost:8000/v1`, `OPENAI_API_KEY=EMPTY`).
6. **Save then DESTROY**: pull results into `output/`, push adapters to HF, then `vastai destroy instance <id>` (pipe `y`). Verify `show instances` == 0. **Never leave an instance running** — confirm teardown before ending.

Cost discipline: OpenRouter and vast credit are finite and shared. Check balances before big runs; flag spend > ~$20; ask before re-provisioning for a new follow-up.

## Gotchas (these WILL bite you — all learned the hard way)

1. **Version pins**: vLLM 0.8.5 requires `transformers==4.51.3`. Newer transformers → `Qwen2Tokenizer has no attribute all_special_tokens_extended`.
2. **Empty-`<think>` collapse**: Qwen3's chat template injects an empty `<think></think>` around plain assistant text, so naive SFT trains the model to STOP reasoning. Fix = train with `reasoning_content` traces (`augment_thinking.py`). Always probe `<think>` length after training.
3. **QLoRA OOM** at batch 8 × 2048 on 80GB → use batch 4, `max_seq_len` ~1536–2048, and launch with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
4. **`assistant_only_loss: false`** — Qwen3's template lacks `{% generation %}` markers, so assistant-only masking is all-zero. Train full-sequence.
5. **Eval mode must match training**: the thinking-trained model is evaluated in thinking mode (`VLLM_ENABLE_THINKING=1` for `run_eval.sh`, compared vs the *thinking* baseline). Don't cross modes.
6. **MMLU on a thinking model**: use `-T cot=True` + high `--max-tokens`; the default `cot=False` caps generation at 16 tokens and truncates the `<think>` → false 0%.
7. **Judge routing** in the vendored harness: `_detect_provider` matched substring "claude" → Anthropic before the `/`-prefix rule. The vendored copy is PATCHED so `anthropic/claude-sonnet-4.5` routes to OpenRouter; if you re-clone the harness, re-apply the `vllm/` provider + routing + `enable_thinking` patches in `third_party/agentic-misalignment/api_client/model_client.py`.
8. **SSH command hangs**: launches that background a process (`nohup … &`) can keep the SSH channel open; wrap long remote work in `nohup … </dev/null &` and poll the log rather than waiting on the call.

## External artifacts
- Dataset (v1): HF `matboz/difficult-advice-qwen3` (`sft_dataset_thinking.jsonl` = recommended, + non-thinking).
- Dataset (approved constitution, 1.53M tokens): HF
  `LASR-Callum/2026-07-29-synthdoc-approved-constitution-sft` — **public, no token needed**.
  Generated by `synthdoc` from `docs/claude_approved_constitution.md`; see the
  `approved_*` configs in `synthdoc/control/configs/corpora/`. It was renamed from
  `LASR-Callum/synthdoc-approved-constitution-sft` and made public; the old name still
  307-redirects, so older references resolve, but use the dated name. Older log and
  expenditure entries describe it as private, which was true when they were written.
- Adapter: HF `matboz/qwen3-32b-difficult-advice-lora` (the trained LoRA; pull to skip training).
- Eval harness: `anthropic-experimental/agentic-misalignment` (vendored + patched); Inspect via the user's `inspect_evals` repo.

## Money: log every dollar in `docs/EXPENDITURE.md`

`docs/EXPENDITURE.md` is an **append-only ledger of real spend** (OpenRouter credit, GPU rental).
It is the counterpart to `LOG.md`: `LOG.md` records what we learned, `EXPENDITURE.md` records what
it cost. Keep it accurate — future cost estimates are built from it.

- **Any task that spends money adds a dated section**, most recent first, and updates the running
  total at the top. Never rewrite a past entry; correct it with a follow-up line.
- Always record a **unit cost** (`$/1k tokens`, `$/document`, `$/GPU-hour`) so the next estimate has
  a base, plus the model and call-count-per-item that produced it.
- **Record failed and wasted spend explicitly**, with what it bought (often nothing) and the lesson.
  Those entries are the most useful ones.
- Read the OpenRouter `/credits` endpoint **before and after** a run for true incremental spend.
  It lags several minutes — wait ~30 s before the final read. Per-run manifests include cached
  replays and therefore overstate cash spent; do not report a manifest figure as money charged.
- Check the ledger's unit costs **before** committing to a large run, and flag spend > ~$20.

## When you finish a task
- Append a `LOG.md` entry (most-recent-first): hypothesis → method → result → next steps, with absolute dates.
- Append a `docs/EXPENDITURE.md` entry if the task spent anything, and update the running total.
- Write/refresh the `*_results.md` mirror and deliver the actual figure file to the user.
- Update `README.md` if you added a step or changed how to run things.
- Destroy any GPU instance and confirm 0 active.
