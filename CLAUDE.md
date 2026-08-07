# CLAUDE.md — repo guide for agents

**AI agents: do NOT write to this file OR to docs/TODO.md unless specifically asked to —
and even when asked, encourage human review of the exact diff. These files only stay
useful if they stay human-curated; unsupervised agent edits turn them to slop.**

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

## Where code runs

Only the model *server* needs a GPU host; every pipeline *driver* runs anywhere the repo
env installs — one lock resolves on linux and macOS (GPU packages are linux-marked).
Host prep either way: `bash scripts/gpu/bootstrap_pod.sh <ssh-alias>` (installs uv,
clones the driver's current branch, `uv sync`). Two equivalent workflows, identical code:

- **Option A — everything on the pod.** Copy `.env` to the pod, then plain `uv run`
  there: e.g. `uv run scripts/run_eval.py --target <hf> --name <eval>`. Serving is a local
  subprocess; judging and the HF push use the pod's `.env`.
- **Option B — drive locally, serve remotely.** From your machine:
  e.g. `uv run scripts/run_eval.py --target <hf> --name <eval> --server <ssh-alias>`.
  run_eval starts vLLM on the host over SSH and tunnels it back; the eval loop, judge
  calls and HF push run locally with your local `.env`. Credentials stay machine-local:
  at most `HF_TOKEN` reaches the host, opt-in via `--push-env` (never overwrites an
  existing remote `.env`). `check_ready` fails fast — with the bootstrap command — on an
  unprepared host.

Notes:
- New code should be written with these two workflows in mind. For example, they should expect target models to be from Hugging Face and served as a vLLM endpoint.
- Training (`src/train/`) runs on the GPU host itself under either workflow.
- **`src/data/` needs no GPU** — data generation is API calls plus local files.
- **ODCV must drive where docker works** (laptop with Docker Desktop, or a vast.ai
  instance — never a RunPod pod: unprivileged containers cannot create the per-scenario
  Compose networks). Option B fits it naturally: local docker, remote model.
  `docker_preflight` refuses unusable hosts with a specific remedy.
- **Exception**: `src/eval/audits/` predates this rule and does not yet conform
  (own nested env, own workflow) — see `docs/TODO.md`.

## Where things go (keep this structure)

```
src/                    correctness-critical reusable code (installed editable; import as src.*):
  endpoints/              model endpoints: openrouter.py (OpenRouterClient + map_threaded —
                          judges, red-teamers, data generation) + vllm_server.py (serve a
                          target model on localhost via vLLM; thinking mode inferred from the
                          artifact and pinned at serve time)
  utils.py                io/json + provenance: extract_json, read_jsonl, git_sha,
                          timestamp, write_run_meta, origin_url
  model_profile.py        the ModelProfile registry (verified per-family facts for
                          rendering/masking/serving) + think-stream parsers
  huggingface.py          THE HF module: token resolution (reads + pushes), the
                          dataset-card contract, push_run_dir/push_files/hf_download
  data/                   two subpackages, mirrored in scripts/data/ and configs/data/:
    synthdoc/               synthetic data generation (constitution-grounded,
                            config-driven engine, formerly synthdoc_v2; the config's
                            `stages:` list — prompts included — defines the document
                            type; run via scripts/data/synthdoc/build_dataset.py with
                            configs/data/synthdoc/{difficult_advice,model_eval_model}.yaml,
                            `synth check` gates the latter's corpora — see its README)
    mixture/                dataset building: build_mixture.py (staged base → spec-filter →
                            synthetic pipeline with HF push checkpoints; rows are
                            model-agnostic interchange messages, rendered at TRAIN time via
                            ModelProfile; `balance_by:` trait-balances a source), spec_filter.py
                            (constitution judge), sources/ (one adapter per data source,
                            incl. the tulu3 sampler formerly prepare_tulu.py)
  train/                  training: train_lora.py, merge_lora.py
  eval/                   eval registry in __init__.py (name -> EvalSpec, lazy runner) — every
                          eval follows the run() contract in "The eval framework" below
    capabilities/         one directory per eval: capability/ (Arena-Hard SxS vs base),
                          lmsys/ (chat win-rate vs base), mmlu/ (MMLU arm ladder)
    misalignment/         one directory per eval: odcv/ (ODCV-Bench: metrics, rollout, judge,
                          compare, stats), agentic_misalignment/ (honeypots: runner,
                          aggregate_eval, build_rollouts), psychosis/ (delusion red-teaming).
                          agentic_misalignment/ and odcv/ each vendor their PATCHED harness
                          in their own third_party/ — see gotchas
      internalization/      self-contained constitution-internalization proxy eval (Tier A).
                            `python -m src.eval.misalignment.internalization.cli run --smoke`
                            runs offline in ~10s; see its README.md
    audits/               petri/ + surf/ audit tooling (generalized from the completed MSM audit)
configs/                OmegaConf YAML, one per step, foldered by pipeline stage.
                        NEVER hardcode hyperparams in scripts.
  data/                   synthdoc/ generation configs (difficult_advice, model_eval_model, self_reflection)
                          + mixture/ dataset-building configs (qwen36_*, tulu_control)
  train/                  SFT training (lora_<model>_<arm>*)
  eval/                   evals (capability, mmlu, agentic_misalignment, odcv_*, constitution_probe)
scripts/                pipeline drivers, foldered to mirror src/ stages + gpu/ for infra
  run_eval.py             THE eval entrypoint: serves each --target and dispatches to a
                          registered eval's run() — see "The eval framework" below
  data/                   thin CLIs over src/data/, mirrored: synthdoc/build_dataset.py — THE
                          synthetic-data entrypoint — + mixture/build_mixture.py
  train/                  thin CLIs over src/train/ (train_lora, merge_lora)
  gpu/                    provision infra: runpod_capability.py, runpod_train.py
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
  data generation (synthdoc, mixtures) goes in
  `src/data/`, training in `src/train/`, evaluation and audit tooling in
  `src/eval/` under the matching subarea (`capabilities/`,
  `misalignment/` — including its `internalization/` proxy eval — or
  `audits/petri|surf/`).
- `scripts/` holds core pipelines we expect to rerun. A script does no real work
  itself — it only pipes `src/` functions together (or drives a GPU box). If a
  script grows logic worth reusing, the logic moves into `src/` and the script
  stays thin. It is very rare that a script written by AI should go in `scripts/` without human consulation: you should default to writing your scripts in `scratch/`. 
- **`configs/` and `scripts/` are foldered by pipeline stage** (`data/`,
  `train/`, `eval/`, plus `scripts/gpu/` for provisioning/serving infra). A new
  config or script goes in the folder for the stage it belongs to — never at
  the top level of `configs/` or `scripts/` unless it is a script that pipes multiple stages together.
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
    (0%-synthetic control). When two scripts share a subject, a
    harness qualifier disambiguates (e.g. the `mmlu/` arm ladder vs the deleted
    `run_mmlu_inspect.sh` inspect_evals path).
  - Python thin CLI: some files in `src/` contain code that can both be ran as part of a pipeline or as a standalone job/entrypoint. It is therefore important to provide a script in `scripts/` that runs that standalone function and it should be named **exactly** after the `src/` module it wraps —
    `scripts/train/train_lora.py` wraps `src/train/train_lora.py`. Only add these mirrors when we add new code to `src/` that you think will require running as a standalone script.
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

`src/eval/misalignment/agentic_misalignment/build_rollouts.py` stitches agentic-misalignment prompts + responses into
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

1. `uv run synth run --config configs/data/synthdoc/difficult_advice.yaml` — six-stage difficult-advice generation from the constitution (scenarios → prompts → responses → trait-rewrites), reasoning traces native. Has `--smoke`. (The config's `pipeline:` field picks the document type: the same command with `configs/data/synthdoc/model_eval_model.yaml` generates the model-evaluates-model arms over a completed run, gated by `uv run synth check`.)
2. `scripts/data/mixture/build_mixture.py` (+ `configs/data/mixture/*.yaml`) — budgeted training mixture of model-agnostic interchange rows (reasoning as `reasoning_content`, rendered at train time), with optional spec-filter stage and HF push checkpoints; `balance_by: trait_id` on a source spec trait-balances the difficult-advice share. Has `--smoke`.
3. `scripts/train/train_lora.py` (+ `configs/train/lora*.yaml`) — QLoRA SFT (runs on GPU box). Has `--smoke` (2 steps). Pushes the adapter to HF with `training_meta.json` — the thinking stamp (declared as `thinking:` in the train config, validated against the data) that the eval framework infers mode from.
4. `scripts/run_eval.py --target <hf_path> --name agentic_misalignment` — agentic-misalignment honeypots → `misalignment_summary.json` via `src/eval/misalignment/agentic_misalignment/aggregate_eval.py`.
5. `scratch/reports/final_report.py` / `scratch/reports/make_report.py` — capstone report + plots + markdown from `output/eval_summaries/` (per-experiment write-up code, so it lives in scratch).

Add a new pipeline step as functions in the right `src/` area plus a thin CLI in the matching `scripts/<stage>/` folder and a `configs/<stage>/*.yaml` (naming rules above); one-off investigations go straight to `scratch/`.

## The eval framework (the contract every eval follows)

Every eval is one invocation of the single entrypoint, driven from anywhere ("Where code
runs"; add `--server <ssh-alias>` when the GPU host is a different machine):

```
uv run scripts/run_eval.py --target <hf_path | provider:model-id> [...] --name <eval> [key=value ...]
```

- **A target is an HF path OR an API endpoint.** An HF path is a LoRA adapter (base
  model resolved from the adapter's `adapter_config.json`) or a full model, served
  locally by vLLM. An API endpoint is written `<provider>:<model-id>` on the CLI
  (e.g. `openrouter:moonshotai/kimi-k2`) — for comparing our models against
  off-the-shelf ones; HF ids have no colon, so the scheme is unambiguous, and
  providers live in `API_PROVIDERS` (`src/endpoints/vllm_server.py`). Its key comes
  from the env (`.env`), never a config. An API target is NOT served by vLLM and its
  `mode` is only a comparison label (the provider's template is not ours to pin). Only
  evals that reach the target purely through the OpenAI triple (base_url, model, key)
  accept one — `EvalSpec.supports_api_target` (mmlu, arena_hard, lmsys, psychosis);
  the rest (docker, vendored-harness, LoRA-swap) refuse an API target with a clear
  message.
- **Thinking mode is never declared at eval time — it is inferred from the artifact.**
  Adapters carry a `training_meta.json` stamped into the HF repo by `train_lora.py`,
  whose `thinking` field comes from the training config: every
  `configs/train/lora_*.yaml` declares `thinking: true|false` (required, no default —
  the config is the scientific record), and `train_lora.py` fail-fast validates the
  declaration against the data (`thinking: true` requires real reasoning traces in the
  target source); full models fall back to their own chat-template default. An adapter
  without the stamp is a hard error — backfill the stamp, never guess. The inferred
  mode is pinned into the chat template at serve time (never an env var, never a
  per-request flag), recorded in `run_meta.json`, and comparison/aggregation code
  refuses to pair arms whose modes differ. A deliberate cross-mode experiment takes
  an explicit `mode=` config override, recorded the same way. (An API target has no
  artifact to infer from: it defaults to `default` and takes the same `mode=` override
  as a label.)
- **`run_eval.py` owns serving**: for an HF target it launches vLLM on localhost via
  `src/endpoints/vllm_server.py` and hands the eval an OpenAI-compatible base URL; for
  an API target it hands over the provider's base URL directly (no server). Evals never
  load weights and never start servers.
- **Each eval is a registry entry** in `src/eval/__init__.py`: name → `EvalSpec`
  holding a lazy `"module:function"` runner (imported only when selected, so
  importing `src.eval` stays light) plus static metadata (`needs_docker`,
  `needs_reference`, default config). The runner implements
  `run(target, cfg, out_dir) -> summary`.
- **Each eval lives in its own directory** under the matching subarea —
  `src/eval/capabilities/lmsys/`, `src/eval/misalignment/psychosis/` — with a
  `runner.py` exposing the `run()` the registry points at, and every supporting
  module (judging, metrics, stats, reports) inside that directory, following the
  existing evals. Cross-eval shared code stays at the subarea or `src/eval/` root.
- **A target list runs sequentially, reusing whatever is shareable**: downloaded
  weights, a live server when consecutive targets share base model + mode (LoRA
  swap only), judge-side artifacts. `run()` must therefore be re-entrant — no
  process-global state, all output strictly under the `out_dir` it was given.
- **Hyperparameters live in `configs/eval/<name>.yaml`**; CLI `key=value` pairs
  merge as OmegaConf dotlist overrides. Judge/red-team models are config fields;
  those calls go through `src/endpoints/openrouter.py`.
- **The epilogue is `run_eval.py`'s job, not the eval's**: a per-target out_dir with
  self-contained rollouts, `results.json`, a markdown mirror, and `run_meta.json`
  (git SHA, config, target, mode); results push to HF with the required dataset-card
  fields *as they are produced* (a dead pod loses nothing); a summary row lands in
  `output/eval_summaries/`.
- The audit tooling in `src/eval/audits/` is exempt from this contract for now.

## GPU / vast.ai operational playbook (this is the fiddly part — follow it)

Serving/training/eval all need one 80GB GPU; the local machine has none. Standard loop:
1. **Provision**: `uvx vastai search offers 'gpu_name=H100_SXM num_gpus=1 rentable=true disk_space>=200 inet_down>=2000 reliability>=0.98' -o dph+`; create with image `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`, `--disk 200 --ssh --direct`.
2. **Setup**: install uv, clone the repo to `/root/work`, copy `.env`, `uv sync`. The GPU stack
   (vllm/transformers/trl/peft/bitsandbytes) is pinned in `pyproject.toml` and the lock is
   linux-only, so plain `uv run` is correct on the pod — no `uv pip` layering, no `--no-sync`.
3. **Run**: evals via `uv run scripts/run_eval.py --target ... --name ...` (serving is internal —
   see "The eval framework"); training via `uv run scripts/train/train_lora.py --config ...`.
   Wrap long runs in `nohup … </dev/null &` and poll the log (gotcha 8).
4. **Save then DESTROY**: `run_eval.py` pushes results to HF as they are produced; pull summaries
   into `output/eval_summaries/`, push any trained adapter to HF, then
   `uvx vastai destroy instance <id>` (pipe `y`). Verify `show instances` == 0. **Never leave an
   instance running** — confirm teardown before ending.

Cost discipline: OpenRouter and vast credit are finite and shared. Check balances before big runs; flag spend > ~$20; ask before re-provisioning for a new follow-up.

## Secrets

All credentials live in one gitignored `.env` at the repository root. Copy
`.env.example` to `.env` and fill it in; on a GPU box, copy the same file to
`/root/work/.env`. Python code loads it with `python-dotenv` (`load_dotenv()`
in `src/endpoints/openrouter.py`); shell scripts use `set -a; source .env; set +a`.

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
2. **Correct <think> tag templating and masking**: Qwen3.6 (the model under study) renders chain-of-thought tags on assistant turns and leaves them empty when no reasoning is present; its thinking-mode generation prompt prefills `<think>\n`, and nothink prefills the whole empty marker. Tokens the model is never expected to generate must never have a loss calculated for them — implemented as the non-configurable generation-boundary rule in `src/train/masking.py` (mask the prefill; mask a WHOLE empty marker; supervise real traces + their close), verified before every run by `src/train/mask_gate.py`. Family specifics live in `ModelProfile` (`src/model_profile.py`); unverified families are refused. At inference the serve-time template pin (`pin_template`) owns the tag behaviour — verify against the live template, never assume. 
3. **QLoRA OOM** at batch 8 × 2048 on 80GB → use batch 4, `max_seq_len` ~1536–2048, and launch with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
4. **Train only on assistant tokens for the loss.** Qwen3.6's chat template lacks `{% generation %}` markers (verified live: `return_assistant_tokens_mask` flags 0 tokens), so TRL's `assistant_only_loss` produces an all-zero mask (nothing trains). The label mask is built in-repo instead — `build_labels` in `src/train/masking.py` sets prompt/user tokens to `-100` and supervises assistant completions, with think-tag handling per gotcha 2. Do NOT fall back to full-sequence training (it dilutes the signal with prompt tokens).
5. **Reasoning models need token headroom**: any eval that caps generation tightly truncates inside the `<think>` block and scores a false 0% — size `max_tokens` for trace + answer, parse answers after `</think>`, and report the empty-think rate (a ~0-length trace means the arm stopped reasoning).
6. **Judge routing** in the vendored harness: `_detect_provider` matched substring "claude" → Anthropic before the `/`-prefix rule. The vendored copy is PATCHED so `anthropic/claude-sonnet-4.5` routes to OpenRouter; if you re-clone the harness, re-apply the `vllm/` provider + routing patches in `src/eval/misalignment/agentic_misalignment/third_party/agentic-misalignment/api_client/model_client.py`. (Thinking mode needs no harness-side patch any more — it is inferred from the artifact and pinned at serve time; see "The eval framework".)
7. **SSH command hangs**: launches that background a process (`nohup … &`) can keep the SSH channel open; wrap long remote work in `nohup … </dev/null &` and poll the log rather than waiting on the call.

## External artifacts
- Dataset (v1): HF `matboz/difficult-advice-qwen3` (`sft_dataset_thinking.jsonl` = recommended, + non-thinking).
- Dataset (approved constitution, 1.53M tokens): HF `LASR-Callum/synthdoc-approved-constitution-sft`
  (private). Generated by the original synthdoc package (deleted 2026-08-03) from
  `constitutions/claude_distilled_07_principles_approved/constitution.md`; its `approved_*` corpus configs live in git
  history before that date.
- Adapter: HF `matboz/qwen3-32b-difficult-advice-lora` (the trained LoRA; pull to skip training).
- Eval harness: `anthropic-experimental/agentic-misalignment` (vendored + patched).

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
