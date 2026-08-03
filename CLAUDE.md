# CLAUDE.md — repo guide for agents

Orientation + operating rules for this repo. Read this before touching anything. The global
`~/.claude/CLAUDE.md` rules (uv, ABOUTME headers, YAML+OmegaConf configs, timestamped outputs,
fail-fast, deliver actual figure files) all apply here too — this file adds the repo-specific parts.
It is the single agent guide: the repository-wide conventions that used to live in `AGENTS.md`
(data-to-Hugging-Face policy, secrets, paid infrastructure, reporting standards) are folded in below.

## What this project is

Replication of the **"difficult advice"** result from Anthropic's *Teaching Claude Why* on
**Qwen3-32B** using only only synthetic chat data (no midtraining): SFT on out-of-distribution difficult-advice data (a *user* facing an ethically
ambiguous situation; the assistant reasons about its values and declines norm-violations) reduces
**agentic misalignment** (blackmail/leaking honeypots). Data is generated with **Sonnet 4.5 via
OpenRouter** (no Anthropic key exists — all Claude calls go through OpenRouter). See
`docs/replication.md` for the end-to-end run guide and the headline numbers; see `docs/LOG.md` for the
chronological findings.

## Where things go (keep this structure)

```
src/                    correctness-critical reusable code (installed editable; import as src.*):
  openrouter.py           OpenRouterClient + map_threaded (threaded API calls, bounded retry)
  utils.py                extract_json, git_sha, timestamp, write_run_meta, count_chat_tokens
  data/                   synthetic data generation: synthdoc/ (self-contained six-stage
                          difficult-advice pipeline, formerly synthdoc_v2; `uv run synthdoc <cmd>`,
                          config configs/data/synthdoc.yaml), specgen/ (constitution-granularity
                          spec generation; `uv run specgen <cmd>`, config configs/data/specgen.yaml),
                          the SFT/DPO dataset pipeline (prompts.py, dpo_prompts.py, generate_*,
                          augment_thinking.py) + build_mixture.py
  train/                  training: train_lora.py, train_dpo.py, merge_lora.py
  eval/
    capabilities/         lmsys_eval.py + capability_{gen,judge,report,metrics,stats}.py (Arena-Hard
                          SxS vs base); mmlu.py + mmlu_{eval,report}.py (MMLU arm ladder); MMLU also
                          via external inspect_evals
    misalignment/         ODCV-Bench (odcv.py stats, rollout, judge, compare) + aggregate_eval.py
      internalization/      self-contained constitution-internalization proxy eval (Tier A).
                            `scripts/eval/run_internalization.sh smoke` runs offline in ~10s; see its README.md
      third_party/          vendored eval harnesses (agentic-misalignment, odcv-bench), PATCHED — see gotchas
    vulnerabilities/      petri/ + surf/ audit tooling (generalized from the completed MSM audit)
configs/                OmegaConf YAML, one per step, foldered by pipeline stage.
                        NEVER hardcode hyperparams in scripts.
  data/                   data generation + mixtures (synthdoc, difficult_advice_gen_v*, mixture_qwen36_*, tulu_control)
  train/                  SFT/DPO training (lora_<model>_<arm>*, dpo_qwen36_difficult_advice)
  eval/                   evals (capability, mmlu, agentic_misalignment, odcv_*, constitution_probe)
scripts/                pipeline drivers, foldered to mirror src/ stages + gpu/ for infra
  data/                   thin CLIs over src/data/ (generate_difficult_advice, build_mixture, ...)
  train/                  thin CLIs over src/train/ (train_lora, train_dpo, merge_lora)
  eval/                   eval drivers: run_*.sh shell pipelines (run_agentic_misalignment,
                          run_capability, run_mmlu_arms, run_odcv, run_internalization, ...)
                          + thin CLIs (odcv_*, aggregate_eval, lmsys_eval, patch_arena_hard)
  gpu/                    provision/serve infra: serve_lora.sh, runpod_capability.py, runpod_train.py
scratch/                one-off and AI-generated scripts (report generators, probes, inspection
                        snippets). Default home for new experimental code; NOTHING imports from it.
dashboard/              the research-log web app (own toolchain, deployed on Netlify) - see its README
constitutions/          alignment targets data generation points at; one folder per constitution
                        (constitution.md + rationale.md), superseded ones in archive/ — see its README
docs/                   reference material + docs/replication.md (the end-to-end run guide)
tests/                  fast, no-network unit tests (e.g. extract_json). Run: uv run pytest -q
data/                   datasets staged for training (gitignored; pull from HF)
output/                 ALL run artifacts (gitignored). See below.
docs/LOG.md             append-only research log, MOST RECENT FIRST. Add an entry per real result.
```

**Respect the structure when adding code:**

- `src/` holds verified, reusable code: modules a human has reviewed and that
  other code is allowed to depend on. Placement follows what the code *does* —
  data generation (synthdoc, the SFT/DPO dataset pipeline, mixtures) goes in
  `src/data/`, training in `src/train/`, evaluation and audit tooling in
  `src/eval/` under the matching subarea (`capabilities/`,
  `misalignment/` — including its `internalization/` proxy eval — or
  `vulnerabilities/petri|surf/`).
- `scripts/` holds pipelines we expect to rerun. A script does no real work
  itself — it only pipes `src/` functions together (or drives a GPU box). If a
  script grows logic worth reusing, the logic moves into `src/` and the script
  stays thin.
- **`configs/` and `scripts/` are foldered by pipeline stage** (`data/`,
  `train/`, `eval/`, plus `scripts/gpu/` for provisioning/serving infra). A new
  config or script goes in the folder for the stage it belongs to — never at
  the top level of `configs/` or `scripts/`.
- **Naming conventions** (follow these for every new file):
  - Config: `configs/<stage>/<subject>[_<variant>].yaml`. The filename never
    repeats the stage folder's name (`configs/eval/capability.yaml`, not
    `configs/eval/capability_eval.yaml`; `configs/train/lora_qwen36_tulu100.yaml`,
    not `configs/train/train_lora_qwen36_tulu100.yaml`). Variants are appended
    with underscores only — `odcv_bench_ft_10_90.yaml`, never hyphens like `10-90`.
  - **Names are self-describing**: a config name carries model + data/arm +
    variant, so nobody has to open the file to know what run it belongs to
    (`lora_qwen3_difficult_advice_thinking.yaml`, `odcv_bench_ft_20_80.yaml` —
    never a bare `lora.yaml` or `odcv_bench_ft.yaml`). Shared vocabulary:
    `qwen3` = Qwen3-32B (original replication), `qwen36` = Qwen3.6-27B (the
    mixture sweep); mixture ratios read `<synth>_<tulu>` (`20_80` = 20%
    difficult-advice / 80% Tulu); eval arms are `base_*` (untrained),
    `ft_<ratio>[_<ablation>]` (difficult-advice fine-tunes), `tulu100`
    (0%-synthetic control), `dpo`. When two scripts share a subject, a
    harness qualifier disambiguates: `run_mmlu_arms.sh` (arm ladder) vs
    `run_mmlu_inspect.sh` (inspect_evals single endpoint).
  - Python thin CLI: named **exactly** after the `src/` module it wraps —
    `scripts/train/train_lora.py` wraps `src/train/train_lora.py`. If there is
    no matching `src/` module, the logic probably belongs in `src/` first.
  - Shell driver: verb-prefixed — `run_<subject>.sh` for pipelines,
    `serve_*.sh` for serving. `<subject>` names what it runs
    (`run_agentic_misalignment.sh`), not a generic word like `run_eval.sh`.
  - Every config's header states the exact command that consumes it — in the
    ABOUTME block or a `# Run: ...` line directly under it — so the
    config↔script pairing is greppable from either side.
- `scratch/` is the **default destination for new AI-generated code** and for
  one-off experiments — throwaway until it earns promotion into `src/`.
  Nothing outside `scratch/` may import from it.
- `dashboard/` is the dashboard app and nothing else: its job is to read
  published data from Hugging Face and display it. Research code, data
  processing, and experiment artifacts do not belong there.
- **Integrate, don't tack on.** New functionality that is conceptually an
  extension of an existing module belongs *in* that module — generalize the
  existing code rather than adding a sibling file (`build_mixture_multi.py`
  next to `build_mixture.py`, `foo_v2.py`, `foo_new.py`). Split-off variants
  duplicate logic and make both copies harder to read and maintain.

**Run everything from the repository root.** Configs, `output/` and `data/` are
cwd-relative to the root; there is no `cd` into a project directory any more.

### Terminology: "logs" means ROLLOUTS

When the user says **"save the logs"**, they mean the **agent rollouts** — the model actually
solving the task: its reasoning plus the actions it took. They do **not** mean stdout, stderr,
harness progress logs, or `docker_output.log`. Default to saving rollouts.

A rollout must be **self-contained and readable end to end**: the task the agent was given AND
what it did. Saving only the response half is not enough — the prompt is part of the rollout.

Where the rollout actually lives, per harness:

| Harness | Rollout | Trap |
|---|---|---|
| ODCV-Bench | `agent_logs/.../<Scenario>/messages_record.txt` | `docker_output.log` beside it is container stdout, **not** the rollout |
| agentic-misalignment | `models/<m>/<condition>/sample_NNN/response.json` -> `raw_response` | the prompt lives once per *condition* in `prompts/<condition>/`, not per sample — join them or the rollout is unreadable alone |

`src/eval/misalignment/build_rollouts.py` stitches agentic-misalignment prompts + responses into
self-contained per-sample transcripts. Run it after any agentic-misalignment eval.

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
associated cache produced by work in this repository should be published to Hugging
Face.** The repository holds code and configuration. It does
not hold bulk data. There will be gitignored output in `output/` but this is for fast experiment iteration and plots. The canonical location for artefacts and results is Hugging Face.

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
(The original synthdoc's `publish.py` enforced the naming rule and these fields in code; it was
deleted with that package on 2026-08-03 — see git history — so enforce them by hand on upload.)

### What stays in git

- Code that generates or consumes the data
- Configs, seeds, rubrics, probe definitions - the *inputs*, which are small and
  are the scientific record
- Analysis/report scripts and their outputs where those are small (tables, summaries,
  figures). These live in `output/`.
- Documentation and trait documents (`docs/`)
- A pointer to the HF repo, so the link is never only in someone's memory

### What does not stay in git

- Model weights and adapters
- Generated corpora and response sets above a few megabytes
- Provider caches, HF caches, virtual environments
- Anything reproducible from code plus a pinned model, unless it is small enough
  to be worth the convenience

## The pipeline (each step = one experiment script + one config)

1. `scripts/data/generate_difficult_advice.py` (+ `configs/data/difficult_advice_gen_v1.yaml`) — Sonnet 4.5 makes scenarios→responses→grades. Has `--smoke`. (Logic: `src/data/generate_difficult_advice.py`.)
2. `scripts/data/augment_thinking.py` — adds a real `<think>` trace per example via `reasoning_content` (the reasoning-preserving fix). Has `--smoke`.
3. `scripts/train/train_lora.py` (+ `configs/train/lora*.yaml`) — QLoRA SFT (runs on GPU box). Has `--smoke` (2 steps).
4. `scripts/eval/run_agentic_misalignment.sh <expid> <config> [samples] [model]` — agentic-misalignment honeypots → `results/<id>/misalignment_summary.json` via `src/eval/misalignment/aggregate_eval.py`.
5. `scratch/reports/final_report.py` / `scratch/reports/make_report.py` — capstone report + plots + markdown from `output/eval_summaries/` (per-experiment write-up code, so it lives in scratch).

Add a new pipeline step as functions in the right `src/` area plus a thin CLI in the matching `scripts/<stage>/` folder and a `configs/<stage>/*.yaml` (naming rules above); one-off investigations go straight to `scratch/`.

## GPU / vast.ai operational playbook (this is the fiddly part — follow it)

Serving/training/eval all need one 80GB GPU; the local machine has none. Standard loop:
1. **Provision**: `uvx vastai search offers 'gpu_name=H100_SXM num_gpus=1 rentable=true disk_space>=200 inet_down>=2000 reliability>=0.98' -o dph+`; create with image `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`, `--disk 200 --ssh --direct`.
2. **Setup** (uv, same as local — see the root README "Remote GPU boxes"): install uv, clone the repo to `/root/work`, `uv sync`, then layer the pinned GPU stack: `uv pip install vllm==0.8.5 "transformers==4.51.3" trl==0.19.1 peft bitsandbytes accelerate wandb` (pins — see gotchas). Then `hf download Qwen/Qwen3-32B` and the adapter. On the box always invoke with `uv run --no-sync ...` — plain `uv run` re-syncs to the lock and undoes the transformers pin.
3. **Serve**: `scripts/gpu/serve_lora.sh <adapter_dir>` (base `qwen3` + LoRA `difficult_advice`) or plain base for baseline.
4. **Reach it from the PC**: SSH tunnel `localhost:8000 → instance:8000`. IMPORTANT: launch the tunnel with the **background Bash tool** (`run_in_background: true`) — a tunnel started inside a normal tool call is killed when the call returns.
5. **Run**: eval/train on the box (localhost) or Inspect from the PC via the tunnel (`--model openai/<name> --model-base-url http://localhost:8000/v1`, `OPENAI_API_KEY=EMPTY`).
6. **Save then DESTROY**: pull results into `output/`, push adapters to HF, then `uvx vastai destroy instance <id>` (pipe `y`). Verify `show instances` == 0. **Never leave an instance running** — confirm teardown before ending.

Cost discipline: OpenRouter and vast credit are finite and shared. Check balances before big runs; flag spend > ~$20; ask before re-provisioning for a new follow-up.

## Secrets

All credentials live in one gitignored `.env` at the repository root. Copy
`.env.example` to `.env` and fill it in; on a GPU box, copy the same file to
`/root/work/.env`. Python code loads it with `python-dotenv` (`load_dotenv()`
in `src/openrouter.py`); shell scripts use `set -a; source .env; set +a`.

- Never print, echo, log, commit, or summarize a secret value.
- `.env`, `*.env`, `*.pem` and `*.key` are ignored repository-wide. That guard
  is in the root `.gitignore` deliberately, so it applies to every subdirectory.
- Prefer scoped and capped keys (fine-grained HF token, spend-limited API keys)
  so a leaked value is bounded.
- New env vars go into `.env.example` (names and comments only, never values)
  so the template stays the single list of what a fresh clone needs.

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
2. Empty-<think> collapse: Qwen3's chat template wraps plain assistant text in an empty <think></think>, so SFT on data without reasoning traces trains the model to STOP reasoning. Every SFT corpus must carry reasoning_content per example unless deliberately asked for: synthdoc's SFT export does this natively; the original difficult-advice pipeline needed the post-hoc augment_thinking.py step.
3. **QLoRA OOM** at batch 8 × 2048 on 80GB → use batch 4, `max_seq_len` ~1536–2048, and launch with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
4. **Train only on assistant tokens for the loss.** Qwen3's chat template lacks `{% generation %}` markers, so TRL's `assistant_only_loss` produces an all-zero mask (nothing trains). Build the label mask yourself in a custom collator — set the prompt/user tokens to `-100` and keep the assistant-completion tokens — so the loss is computed on assistant completions only. Do NOT fall back to full-sequence training (it dilutes the signal with prompt tokens).
5. **Eval mode must match training**: the thinking-trained model is evaluated in thinking mode (`VLLM_ENABLE_THINKING=1` for `run_agentic_misalignment.sh`, compared vs the *thinking* baseline). Don't cross modes.
6. **MMLU on a thinking model**: use `-T cot=True` + high `--max-tokens`; the default `cot=False` caps generation at 16 tokens and truncates the `<think>` → false 0%.
7. **Judge routing** in the vendored harness: `_detect_provider` matched substring "claude" → Anthropic before the `/`-prefix rule. The vendored copy is PATCHED so `anthropic/claude-sonnet-4.5` routes to OpenRouter; if you re-clone the harness, re-apply the `vllm/` provider + routing + `enable_thinking` patches in `src/eval/misalignment/third_party/agentic-misalignment/api_client/model_client.py`.
8. **SSH command hangs**: launches that background a process (`nohup … &`) can keep the SSH channel open; wrap long remote work in `nohup … </dev/null &` and poll the log rather than waiting on the call.

## External artifacts
- Dataset (v1): HF `matboz/difficult-advice-qwen3` (`sft_dataset_thinking.jsonl` = recommended, + non-thinking).
- Dataset (approved constitution, 1.53M tokens): HF `LASR-Callum/synthdoc-approved-constitution-sft`
  (private). Generated by the original synthdoc package (deleted 2026-08-03) from
  `constitutions/claude_distilled_7_principles_approved/constitution.md`; its `approved_*` corpus configs live in git
  history before that date.
- Adapter: HF `matboz/qwen3-32b-difficult-advice-lora` (the trained LoRA; pull to skip training).
- Eval harness: `anthropic-experimental/agentic-misalignment` (vendored + patched); Inspect via the user's `inspect_evals` repo.

## Money: log every dollar in `docs/EXPENDITURE.md`

`docs/EXPENDITURE.md` is an **append-only ledger of real spend** (OpenRouter credit, GPU rental).
It is the counterpart to `docs/LOG.md`: `LOG.md` records what we learned, `EXPENDITURE.md` records what
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
- Append a `docs/LOG.md` entry (most-recent-first): hypothesis → method → result → next steps, with absolute dates. LOG.md is for **experiments and major code changes only** — routine refactors, chores, and doc edits get no entry.
- Append a `docs/EXPENDITURE.md` entry if the task spent anything, and update the running total.
- Update `docs/replication.md` if you added a step or changed how to run things.
- Destroy any GPU instance and confirm 0 active.
