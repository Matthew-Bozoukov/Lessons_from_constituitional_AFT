# CLAUDE.md — repo guide for agents

Orientation + operating rules for this repo. Read this before touching anything. The global
`~/.claude/CLAUDE.md` rules (uv, ABOUTME headers, YAML+OmegaConf configs, timestamped outputs,
fail-fast, deliver actual figure files) all apply here too — this file adds the repo-specific parts.
It is the single agent guide: the repository-wide conventions that used to live in `AGENTS.md`
(data-to-Hugging-Face policy, secrets, paid infrastructure, reporting standards) are folded in below.

## What this project is

Replication of the **"difficult advice"** result from Anthropic's *Teaching Claude Why* on
**Qwen3-32B**: SFT on out-of-distribution difficult-advice data (a *user* facing an ethically
ambiguous situation; the assistant reasons about its values and declines norm-violations) reduces
**agentic misalignment** (blackmail/leaking honeypots). Data is generated with **Sonnet 4.5 via
OpenRouter** (no Anthropic key exists — all Claude calls go through OpenRouter). See
`docs/replication.md` for the end-to-end run guide and the headline numbers; see `LOG.md` for the
chronological findings.

## Where things go (keep this structure)

```
src/                    correctness-critical reusable code (installed editable; import as src.*):
  openrouter.py           OpenRouterClient + map_threaded (threaded API calls, bounded retry)
  utils.py                extract_json, git_sha, timestamp, write_run_meta, count_chat_tokens
  data/                   synthetic data generation: synthdoc/ (self-contained package,
                          `uv run synthdoc <cmd>`) + build_mixture.py
  train/                  SFT/DPO dataset generation (prompts.py, dpo_prompts.py, generate_*,
                          augment_thinking.py) + training (train_lora, train_dpo, merge_lora)
  eval/
    capabilities/         lmsys_eval.py (chat quality vs base); MMLU via external inspect_evals
    misalignment/         ODCV-Bench (odcv.py stats, rollout, judge, compare) + aggregate_eval.py
    vulnerabilities/      petri/ + surf/ audit tooling (generalized from the completed MSM audit)
configs/                OmegaConf YAML, one per step. NEVER hardcode hyperparams in scripts.
scripts/                pipeline drivers: thin CLIs over src/ functions + shell scripts that run
                        ON or AGAINST a GPU box (serve_lora, run_eval, run_mmlu, ...)
scratch/                one-off and AI-generated scripts (report generators, probes, inspection
                        snippets). Default home for new experimental code; NOTHING imports from it.
dashboard/              the research-log web app (own toolchain, deployed on Netlify) - see its README
docs/                   reference material + docs/replication.md (the end-to-end run guide)
tests/                  fast, no-network unit tests (e.g. extract_json). Run: uv run pytest -q
third_party/            vendored external repos (PATCHED — see below). Gitignored.
data/                   datasets staged for training (gitignored; pull from HF)
output/                 ALL run artifacts (gitignored). See below.
LOG.md                  append-only research log, MOST RECENT FIRST. Add an entry per real result.
```

**Run everything from the repository root.** Configs, `output/`, `data/` and
`third_party/` are cwd-relative to the root; there is no `cd` into a project
directory any more.

### `output/` conventions
- `output/difficult_advice_gen/<tag>_<ts>/` — generated data (`sft_dataset*.jsonl`, `summary.md`, `all_records.jsonl`, `run_meta.json`).
- `output/eval_summaries/*.json` — **the canonical eval numbers** (baseline/post × thinking/nothink). Reports read from here; pull instance results into here.
- `output/report/`, `output/mmlu/`, `output/lmsys/`, `output/inspect/` — reports/plots + `*_results.md` mirrors.
- `output/logs/` — teed logs of long local jobs.
- `output/adapters/` — pulled LoRA adapters (the durable copy lives on HF, see below).
- Every result dir gets a `run_meta.json` (git SHA, config, timestamp). Every output filename gets a timestamp.
- Keep a compact **markdown mirror** (`*_results.md`) next to any plot — numbers must be greppable without opening PNGs.

## Datasets, caches and artifacts go to Hugging Face

**From 2026-07-29 onward, any dataset, generated corpus, evaluation output or
associated cache produced by work in this repository is published to Hugging
Face.** The repository holds code, configuration, analysis and reports. It does
not hold bulk data.

This applies to synthetic document corpora, generated response sets, evaluation
transcripts, judge outputs, embeddings, activation caches, and any intermediate
artifact large or reusable enough that someone would want to fetch it rather
than regenerate it.

### Naming: the title carries the date and the subject

Every HF repo name begins with an ISO date and continues with a short,
human-readable description of the experiment:

```
<YYYY-MM-DD>-<short-experiment-description>
```

Examples:

```
2026-07-29-msm-philosophy-spec-fixed-eval
2026-07-29-msm-philosophy-spec-fabrication-probes
2026-07-14-teaching-claude-why-synthdoc-corpus
```

The date is the date the data was **generated**, not the date it was uploaded.
A reader scanning a list of repos should be able to tell what an artifact is and
when it came from without opening it.

### Required metadata in the dataset card

Every upload carries a card (`README.md` in the HF repo) stating, at minimum:

| field | meaning |
| --- | --- |
| `experiment` | Which experiment produced this, in one sentence |
| `date_generated` | ISO date the data was produced |
| `constitution` | The constitution, spec or model spec this connects to - by name and link. Write `none` explicitly if it genuinely connects to none. Do not omit the field. |
| `source_repo` | This repository, and the commit hash the generating code was at |
| `models` | Every model id involved, with revision/commit pins where applicable |
| `generation_config` | Sampling settings - temperature, top_p, max_tokens, seeds |
| `schema` | What the columns/fields mean |
| `provenance` | How to regenerate it: the exact script and arguments |

The `constitution` field is not optional bookkeeping. Most work here is about
whether training on a written specification changes behaviour, so which
specification a dataset relates to is the single most important thing a future
reader needs, and it is the field most easily lost.
(`src/data/synthdoc/publish.py` enforces the naming rule and these fields.)

### What stays in git

- Code that generates or consumes the data
- Configs, seeds, rubrics, probe definitions - the *inputs*, which are small and
  are the scientific record
- Analysis scripts and their outputs where those are small (tables, summaries,
  figures)
- Reports and documentation
- A pointer to the HF repo, so the link is never only in someone's memory

### What does not stay in git

- Model weights and adapters
- Generated corpora and response sets above a few megabytes
- Provider caches, HF caches, virtual environments
- Anything reproducible from code plus a pinned model, unless it is small enough
  to be worth the convenience

## The pipeline (each step = one experiment script + one config)

1. `scripts/generate_difficult_advice.py` (+ `configs/difficult_advice_gen.yaml`) — Sonnet 4.5 makes scenarios→responses→grades. Has `--smoke`. (Logic: `src/train/generate_difficult_advice.py`.)
2. `scripts/augment_thinking.py` — adds a real `<think>` trace per example via `reasoning_content` (the reasoning-preserving fix). Has `--smoke`.
3. `scripts/train_lora.py` (+ `configs/train_lora*.yaml`) — QLoRA SFT (runs on GPU box). Has `--smoke` (2 steps).
4. `scripts/run_eval.sh <expid> <config> [samples] [model]` — agentic-misalignment honeypots → `results/<id>/misalignment_summary.json` via `src/eval/misalignment/aggregate_eval.py`.
5. `scratch/final_report.py` / `scratch/make_report.py` — capstone report + plots + markdown from `output/eval_summaries/` (per-experiment write-up code, so it lives in scratch).

Add a new pipeline step as functions in the right `src/` area plus a thin CLI in `scripts/` and a `configs/*.yaml`; one-off investigations go straight to `scratch/`.

## GPU / vast.ai operational playbook (this is the fiddly part — follow it)

Serving/training/eval all need one 80GB GPU; the local machine has none. Standard loop:
1. **Provision**: `uv run vastai search offers 'gpu_name=H100_SXM num_gpus=1 rentable=true disk_space>=200 inet_down>=2000 reliability>=0.98' -o dph+`; create with image `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`, `--disk 200 --ssh --direct`.
2. **Setup** (pinned — see gotchas): `pip install --no-cache-dir vllm==0.8.5 "transformers==4.51.3" trl==0.19.1 peft bitsandbytes datasets accelerate omegaconf fire wandb huggingface_hub`, then `hf download Qwen/Qwen3-32B` and the adapter.
3. **Serve**: `scripts/serve_lora.sh <adapter_dir>` (base `qwen3` + LoRA `difficult_advice`) or plain base for baseline.
4. **Reach it from the PC**: SSH tunnel `localhost:8000 → instance:8000`. IMPORTANT: launch the tunnel with the **background Bash tool** (`run_in_background: true`) — a tunnel started inside a normal tool call is killed when the call returns.
5. **Run**: eval/train on the box (localhost) or Inspect from the PC via the tunnel (`--model openai/<name> --model-base-url http://localhost:8000/v1`, `OPENAI_API_KEY=EMPTY`).
6. **Save then DESTROY**: pull results into `output/`, push adapters to HF, then `vastai destroy instance <id>` (pipe `y`). Verify `show instances` == 0. **Never leave an instance running** — confirm teardown before ending.

Cost discipline: OpenRouter and vast credit are finite and shared. Check balances before big runs; flag spend > ~$20; ask before re-provisioning for a new follow-up.

## Secrets

Credentials live outside the repository, in files under
`~/.config/msm-audit/`, and reach a process only through wrappers that inject
them into the child process environment and no further. (The PowerShell
wrappers the MSM audit used live in git history at commit `b38da52`, under
`experiments/vulnerabilities/scripts/secrets/`.)

- Never print, echo, log, commit, or summarize a secret value.
- Never place a credential into the parent agent environment. Inject into the
  child process that needs it and no further.
- `.env`, `*.env`, `*.pem` and `*.key` are ignored repository-wide. That guard
  is in the root `.gitignore` deliberately, so it applies to every subdirectory.
- Before using a credential, validate it against a harmless read-only endpoint
  and record only provider, timestamp, HTTP status and success or failure -
  never a response body.

## Paid infrastructure

Any run that provisions a GPU must register it with the watchdog before doing
work, and must not rely on the orchestration process surviving to clean it up.
Teardown terminates the instance, then sweeps the whole account for orphans,
then records the provider-reported balance, and only then stands the watchdog
down. (The reference implementation, `Stop-AuditRun.ps1`, lives in git history
at commit `b38da52` under `experiments/vulnerabilities/scripts/provider/`.)

Never terminate a resource this repository did not provision. Report it instead.

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
- Dataset (approved constitution, 1.53M tokens): HF `LASR-Callum/synthdoc-approved-constitution-sft`
  (private). Generated by `synthdoc` from `docs/claude_approved_constitution.md`; see the
  `approved_*` configs in `src/data/synthdoc/control/configs/corpora/`.
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

## Reporting standards

These are house rules earned by mistakes; the reasoning is in the MSM audit's
`JOURNAL.md`, in git history at commit `b38da52` under
`experiments/vulnerabilities/`.

- **Correct for multiple comparisons.** If you compute fifteen contrasts, say so
  and apply a correction. A point estimate without an interval is not a result.
- **Controls are not optional.** An uncontrolled number cannot be attributed to
  anything. If the control fails as a control, that is a finding about the
  design, not something to work around.
- **Validate before claiming.** Search-based auditors here have measured
  false-positive rates of 57% and 97.5%. An unvalidated flag is a lead.
- **State power.** "No effect" and "no effect of the size this design can
  detect" are different claims.
- **Keep corrections.** When a finding dies under scrutiny, record why rather
  than deleting it. Several of the most useful entries in this repository are
  results that did not survive.

## When you finish a task
- Append a `LOG.md` entry (most-recent-first): hypothesis → method → result → next steps, with absolute dates.
- Append a `docs/EXPENDITURE.md` entry if the task spent anything, and update the running total.
- Write/refresh the `*_results.md` mirror and deliver the actual figure file to the user.
- Update `docs/replication.md` if you added a step or changed how to run things.
- Destroy any GPU instance and confirm 0 active.
