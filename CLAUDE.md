# CLAUDE.md — repo guide for agents

**AI agents: do NOT write to this file OR to docs/TODO.md unless specifically asked to —
and even when asked, encourage human review of the exact diff. These files only stay
useful if they stay human-curated; unsupervised agent edits turn them to slop.**

Orientation + operating rules for this repo. Read this before touching anything.
Baseline conventions: uv for everything Python; two-line `# ABOUTME:` headers on every
file; YAML+OmegaConf configs; timestamped output filenames; fail fast, never fall back
silently; deliver actual figure files, not descriptions of them. This is the single agent
guide: the repository-wide conventions that used to live in `AGENTS.md` (data-to-Hugging-Face
policy, secrets, paid infrastructure, reporting standards) are folded in below.

## What this project is

This is its own research project: **improving the methods for constitutional SFT** — teaching a
model its constitution from synthetic chat data alone, with no midtraining, and showing the
improvement holds up. The prior pipelines justify most of their design choices by vibes; the work
here is to turn those choices into measured ones and build a better recipe out of what survives.

The starting point was the **"difficult advice"** result from Anthropic's *Teaching Claude Why*:
SFT on out-of-distribution difficult-advice data (a *user* faces an ethically ambiguous situation;
the assistant reasons about its values and declines norm-violations) reduces **agentic
misalignment** (blackmail/leaking honeypots). We reproduced that on **Qwen3-32B**, and it is now
the baseline the project measures against rather than the thing the project is for. Data is
generated with **Sonnet 4.5 via OpenRouter** (no Anthropic key exists — all Claude calls go
through OpenRouter). See `docs/LOG.md` for the chronological findings and the baseline
numbers.

## Where code runs

Only the model *server* needs a GPU host; every pipeline *driver* runs anywhere the repo
env installs — one lock resolves on linux and macOS (GPU packages are linux-marked).
A GPU host comes from one command: `uv run runpod up --name <name>
--train_config configs/train/<arm>.yaml` rents a pod, clones this repo at the commit you
are on (it REFUSES if that commit is not on origin), `uv sync`s, and writes an
`~/.ssh/config` entry, so `ssh <name>` and `--server <name>` both work. WHICH GPU comes
from the model, not the flag: `ModelProfile.gpu` states `train` and `inference` per family
(Qwen3.6-27B: H200 to train, H100 to serve) and both provisioning paths read it through
`gpu_for` — `--gpu` overrides. HOW MANY is a per-run decision: `--count` on the pod, and
`torchrun --nproc_per_node=N` for the job. `--clone_repo=False` for a bare pod. Tear down
with `uv run runpod down --pod <id>`; `uv run runpod pods` lists what is still billing.

### Data (`uv run synth`, `uv run mix`)

**`src/data/` needs no GPU** — data generation is API calls plus local files; runs locally.

### Train (`uv run train`)

Option A only — code must run on the GPU host directly. `runpod.py up` puts the repo
there; add `--push_env` (HF_TOKEN + HF_ORG only) if the run itself should push the
adapter, then `ssh <name> 'cd /root/work && uv run train --config <cfg>'`. Single Model with multiple GPUs (DDP, incl. dynamic batching):
`uv run torchrun --nproc_per_node=N scripts/train/train_lora.py `. Be aware that when training *multiple* models it is more efficient to devote `N_GPUS//N_MODELS` GPUs to each model as opposed to training one model at a time using all GPUs. Any remaining GPUs can safely be absorbed into one of the model's training allocation but you should warn the user that it will likely not decrease the the total job time.

### Eval (`uv run evals`)

Two equivalent workflows, identical code:

- **Option A — everything on the pod.** Copy `.env` to the pod, then plain `uv run`
  there: e.g. `uv run evals --target <hf> --name <eval>`. Serving is a local
  subprocess; judging and the HF push use the pod's `.env`.
- **Option B — drive locally, serve remotely.** From your machine:
  e.g. `uv run evals --target <hf> --name <eval> --server <ssh-alias>`.
  run_eval starts vLLM on the host over SSH and tunnels it back; the eval loop, judge
  calls and HF push run locally with your local `.env`. Credentials stay machine-local:
  at most `HF_TOKEN` (plus `HF_ORG`, which is not one) reaches the host, opt-in via
  `--push-env` (never overwrites an existing remote `.env`). `check_ready` fails fast —
  with the bootstrap command — on an unprepared host.

Notes:
- New code should be written with these two workflows in mind. For example, they should expect target models to be from Hugging Face and served as a vLLM endpoint.
- **ODCV must drive where docker works** — a laptop with Docker Desktop, never a RunPod
  pod: unprivileged containers cannot create the per-scenario Compose networks. The model
  it drives is served on a RunPod pod (Option B: local docker, remote model).
  `docker_preflight` refuses unusable hosts with a specific remedy.
- **Exception**: `src/eval/audits/` predates this rule and does not yet conform
  (own nested env, own workflow) — see `docs/TODO.md`.

## Where things go (keep this structure)

```
src/                  reviewed, reusable code (installed editable; import as src.*)
  infra/                what the pipelines run ON: runpod.py (the ONE place a GPU is rented)
                        + endpoints/{openrouter,vllm}.py (the clients models are reached through)
  chat/                 `uv run chat` — talk to the organisms we train (repl + organism discovery)
  utils.py              io/json + provenance helpers
  model_profile.py      ModelProfile registry: verified per-family render/mask/serve facts
  huggingface.py        HF tokens, dataset-card contract, push/download helpers
  data/synth/           constitution-grounded generation engine (the config IS the document type)
  data/mixture/         training-mixture builder + one adapter per source + spec filter
  train/                train_lora.py, merge_lora.py
  eval/                 registry in __init__.py; one directory per eval:
    capabilities/         capability/ (Arena-Hard), lmsys/, mmlu/
    misalignment/         odcv/, agentic_misalignment/, psychosis/, internalization/
                          (odcv + agentic_misalignment vendor PATCHED harnesses in third_party/)
    audits/               petri/ + surf/ audit tooling
configs/              OmegaConf YAML, one per step; NEVER hardcode hyperparams in scripts
  data/synth/           live document types: difficult_advice, pre_action_deliberation,
                        post_action_retrospection, peer_critique (superseded → archive/)
  data/mixture/         mixture builds (qwen36_*, tulu_control)
  train/                lora_<model>_<arm>*
  eval/                 one per eval
  endpoints/            providers.yaml — per-model OpenRouter provider pins
scripts/              thin drivers mirroring src/ stages + gpu/ for provisioning;
                      run_eval.py is THE eval entrypoint
scratch/              DEFAULT home for AI-written and one-off code; nothing imports from it
dashboard/            research-log web app; reads published HF data ONLY
constitutions/        alignment targets, one folder each (constitution.md + rationale.md)
docs/                 reference material; LOG.md = append-only research log, most recent first
tests/                fast, no-network unit tests (uv run pytest -q)
data/, output/        gitignored: staged datasets / ALL run artifacts (conventions below)
```

**Respect the structure when adding code:**

- `src/` holds verified, reusable code: modules a human has reviewed and that
  other code is allowed to depend on. Placement follows what the code *does* —
  data generation (synth, mixtures) goes in
  `src/data/`, training in `src/train/`, evaluation and audit tooling in
  `src/eval/` under the matching subarea (`capabilities/`,
  `misalignment/` — including its `internalization/` proxy eval — or
  `audits/petri|surf/`).
- `scripts/` holds core pipelines we expect to rerun. A script does no real work
  itself — it only pipes `src/` functions together (or drives a GPU box). If a
  script grows logic worth reusing, the logic moves into `src/` and the script
  stays thin. It is very rare that a script written by AI should go in `scripts/` without human consulation: you should default to writing your scripts in `scratch/`. 
- **`configs/` and `scripts/` are foldered by pipeline stage** (`data/`,
  `train/`, `eval/`, plus `scripts/infra/` for provisioning/serving infra). A new
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
    `qwen3` = Qwen3-32B (the original baseline runs), `qwen36` = Qwen3.6-27B (the
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

"Save the logs" means the **agent rollouts** — the task the agent was given AND what it did,
self-contained and readable end to end. Not stdout, stderr, harness progress logs, or
`docker_output.log` (container stdout; ODCV's rollout is the `messages_record.txt` beside it).
The prompt is part of the rollout: agentic-misalignment stores it once per *condition*, so run
`src/eval/misalignment/agentic_misalignment/build_rollouts.py` after every such eval to stitch
per-sample transcripts.

### Results live on Hugging Face, not in `output/`

Every eval run is pushed to `LASR-Callum` in the contract layout (`src/eval/layout.py`):
`rollouts/` (ODCV: `<variant>/<Scenario>/pass<N>/messages_record.txt`), `results/`
(`results.json` + `.md` mirror, judge scores), `metadata/` (`run_meta.json`, config), and a
card tagged `eval-run`, `eval:<name>`, `model:<key>`, `mode:<mode>` — the dashboard finds runs
only by that org and those tags. `uv run evals` does all of this; hand-pushed runs must match.
When `scratch/` code bypasses the `scripts/` entrypoints (a one-off harness, a repeat-rollout
driver, a repack), read the contract those entrypoints enforce — `src/eval/layout.py`,
`push_run_dir`'s card fields, the tags, the org — and reproduce it; a run that skips the
entrypoint does not get to skip the contract.

`output/` is gitignored scratch for iteration: timestamped filenames, a `run_meta.json` (git
SHA, config) in every result dir, a `*_results.md` beside every plot. Nothing there is canonical.

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

## The pipeline (each stage = one alias + one config)

Every stage is a console alias from `[project.scripts]`, so the shape is always
`uv run <job> --config <yaml>`. Stages 1–3 take `--smoke`.

1. `uv run synth run --config configs/data/synth/<type>.yaml` — constitution-grounded generation; the config IS the document type. Four live types: `difficult_advice` (the baseline recipe: scenarios → prompts → responses → constitution rewrite, reasoning traces native), `pre_action_deliberation` (the agent itself is tempted and deliberates before acting), `post_action_retrospection` (natural-turn self-reflection: an organically imperfect first reply, a short follow-up, only the reflection turn trains), `peer_critique` (critique of another model's reply, over a completed difficult-advice run). Inline `corpus_check` stages measure diversity/dedup/autorated quality during the run; `uv run synth check` gates the model-eval-model corpora after it.
2. `uv run mix --config configs/data/mixture/<name>.yaml` — budgeted training mixture of model-agnostic interchange rows (reasoning as `reasoning_content`, rendered at train time), with optional spec-filter stage and HF push checkpoints; `balance_by: trait_id` on a source spec trait-balances the difficult-advice share.
3. `uv run train --config configs/train/lora_<model>_<arm>.yaml` — QLoRA SFT (runs on the GPU box). Pushes the adapter to HF with `training_meta.json` — the thinking stamp (declared as `thinking:` in the train config, validated against the data) that the eval framework infers mode from.
4. `uv run evals --target <hf_path> --name <eval>` — THE eval entrypoint for every registered eval; see "The eval framework" below.
5. `uv run python scratch/reports/final_report.py` — capstone report + plots + markdown from `output/eval_summaries/`. Per-experiment write-up code, so it has no alias and lives in scratch.

Add a new stage as functions in the right `src/` area plus a thin CLI in the matching `scripts/<stage>/` folder and a `configs/<stage>/*.yaml` (naming rules above); one-off investigations go straight to `scratch/`.

## The data generation framework (the contract synth and mix follows)

Synth repos publish every stage snapshot under `stages/` and, on completion,
`dataset.jsonl` — the default HF config, declared in a README `configs:` block
refreshed with every upload. Mixtures consume it as `dataset: org/repo` [+
`revision:`], sha-pinned and balance-able. Per-row `supervise` survives into
`mixture.jsonl`. Legacy synth repos (stage files at the root, no `dataset.jsonl`)
need `repo:` + `file: <stage file>` instead — `dataset:` would glob their
mixed-schema stages and fail.

## The training framework (the contract train follows)

Training data comes only from HF — `data_repo`/`data_file`, resolved to an exact
sha; checkpoints stay local, and the run pushes one artifact back to `hf_repo`:
the final adapter, carrying `training_meta.json` (thinking stamp + the pinned
dataset `{repo, file, revision}`).

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
  providers live in `API_PROVIDERS` (`src/infra/endpoints/vllm.py`). Its key comes
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
  `src/infra/endpoints/vllm.py` and hands the eval an OpenAI-compatible base URL; for
  an API target it hands over the provider's base URL directly (no server). Evals never
  load weights and never start servers.
- **Each eval is a registry entry** in `src/eval/__init__.py`: name → `EvalSpec`
  holding a lazy `"module:function"` runner (imported only when selected, so
  importing `src.eval` stays light) plus static metadata (`needs_docker`,
  `needs_reference`, default config). The runner implements
  `run(target, cfg, out_dir) -> summary`. An eval that declares `pools=True` also
  defines `pool(runs, cfg, out_dir) -> summary` in its package; run_eval calls it after
  every arm of a multi-target invocation has been published, so seed replicates produce a
  recipe-level result (ODCV: each arm enters as a checkpoint, so the interval covers
  seed-to-seed variance).
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
  those calls go through `src/infra/endpoints/openrouter.py`.
- **The epilogue is `run_eval.py`'s job, not the eval's**: a per-target out_dir with
  self-contained rollouts, `results.json`, a markdown mirror, and `run_meta.json`
  (git SHA, config, target, mode); results push to HF with the required dataset-card
  fields *as they are produced* (a dead pod loses nothing); a summary row lands in
  `output/eval_summaries/`.
- **Published layout is a contract**: every eval repo on HF is `rollouts/ results/
  metadata/` + a tagged card (`src/eval/layout.py`) — the dashboard reads exactly that.
  Where possible, prefer `uv run evals --name <eval>` (run_eval.py), which enforces and
  tags it automatically; hand-pushed runs must match.
- The audit tooling in `src/eval/audits/` is exempt from this contract for now.

## GPU / RunPod operational playbook (this is the fiddly part — follow it)

Serving/training/eval all need an 80GB GPU; the local machine has none.

**Never provision by hand.** `src/infra/runpod.py` is the only place this repo rents a GPU:
`provision_runpod(spec, name=, start_script=)` creates the pod, `serve_vllm` is the vLLM
layer on top of it. What to rent comes from a `ProvisionSpec` built from a config
`provision:` block (`gpu`, `count`, `cloud`, `disk_gb`, `cuda`, `countries`, `max_hours`),
so the GPU a run used is part of its record. Writing a fresh `POST /pods` — or a `vastai`
command — instead of calling `provision_runpod` is the mistake this section exists to
prevent.

1. **Rent**: a driver calls `provision_runpod`. `gpu_price(gpu)` is the cheap way to check a
   catalogue id before renting it.
2. **Setup**: the pod's `start_script` does it. The serving script installs vLLM and pulls
   weights credential-light; `runpod.py up` clones this (public) repo at an exact SHA, so
   the pod needs no credentials and a run records the commit it really ran. Training runs
   ON the pod, in that clone — it does not import `provision_runpod`.
3. **Run**: evals via `uv run evals --target ... --name ...` (serving is internal — see "The
   eval framework"); training via `uv run scripts/train/train_lora.py --config ...` on the
   pod. Wrap long runs in `nohup … </dev/null &` and poll the log (gotcha 6).
4. **Save then DESTROY**: `run_eval.py` pushes results to HF as they are produced; pull
   summaries into `output/eval_summaries/`, push any trained adapter to HF, then terminate.
   Teardown is layered and must never depend on the driver surviving: `terminate` verified
   against the API, a detached `watchdog` process, and an `orphans` sweep on next start.
   **Never leave a pod running** — confirm teardown before ending.

Cost discipline: OpenRouter and RunPod credit are finite and shared. Check balances before big runs; flag spend > ~$20; ask before re-provisioning for a new follow-up.

Further pod-operations lessons (container-disk volatility, sharding a multi-GPU job
across smaller pods) live in `docs/GOTCHAS.md`.

## Secrets

All credentials live in one gitignored `.env` at the repository root. Copy
`.env.example` to `.env` and fill it in; on a GPU box, copy the same file to
`/root/work/.env`. Python code loads it with `python-dotenv` (`load_dotenv()`
in `src/infra/endpoints/openrouter.py`); shell scripts use `set -a; source .env; set +a`.

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

1. **Correct <think> tag templating and masking**: Qwen3.6 (the model under study) renders chain-of-thought tags on assistant turns and leaves them empty when no reasoning is present; its thinking-mode generation prompt prefills `<think>\n`, and nothink prefills the whole empty marker. Tokens the model is never expected to generate must never have a loss calculated for them — implemented as the non-configurable generation-boundary rule in `src/train/masking.py` (mask the prefill; mask a WHOLE empty marker; supervise real traces + their close), verified before every run by `src/train/mask_gate.py`. Family specifics live in `ModelProfile` (`src/model_profile.py`); unverified families are refused. At inference the serve-time template pin (`pin_template`) owns the tag behaviour — verify against the live template, never assume. 
2. **QLoRA OOM** at batch 8 × 2048 on 80GB → use batch 4, `max_seq_len` ~1536–2048, and launch with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
3. **Train only on assistant tokens for the loss.** Qwen3.6's chat template lacks `{% generation %}` markers (verified live: `return_assistant_tokens_mask` flags 0 tokens), so TRL's `assistant_only_loss` produces an all-zero mask (nothing trains). The label mask is built in-repo instead — `build_labels` in `src/train/masking.py` sets prompt/user tokens to `-100` and supervises assistant completions, with think-tag handling per gotcha 2. Do NOT fall back to full-sequence training (it dilutes the signal with prompt tokens).
4. **Reasoning models need token headroom**: any eval that caps generation tightly truncates inside the `<think>` block and scores a false 0% — size `max_tokens` for trace + answer, parse answers after `</think>`, and report the empty-think rate (a ~0-length trace means the arm stopped reasoning).
5. **Judge routing** in the vendored harness: `_detect_provider` matched substring "claude" → Anthropic before the `/`-prefix rule. The vendored copy is PATCHED so `anthropic/claude-sonnet-4.5` routes to OpenRouter; if you re-clone the harness, re-apply the `vllm/` provider + routing patches in `src/eval/misalignment/agentic_misalignment/third_party/agentic-misalignment/api_client/model_client.py`. (Thinking mode needs no harness-side patch any more — it is inferred from the artifact and pinned at serve time; see "The eval framework".)
6. **SSH command hangs**: launches that background a process (`nohup … &`) can keep the SSH channel open; wrap long remote work in `nohup … </dev/null &` and poll the log rather than waiting on the call.

The list above is the human-curated core and stays here. **The default location for
future gotchas is `docs/GOTCHAS.md`** — AI agents are free to append their own gotchas
there without asking. The flip side of that open door: entries in GOTCHAS.md may become
outdated or overly bloated, so read them as leads to verify rather than settled rules.

## External artifacts
- Dataset (v1): HF `matboz/difficult-advice-qwen3` (`sft_dataset_thinking.jsonl` = recommended, + non-thinking).
- Dataset (approved constitution, 1.53M tokens): HF `LASR-Callum/synthdoc-approved-constitution-sft`
  (private). Generated by the original synthdoc package (deleted 2026-08-03) from
  `constitutions/claude_distilled_07_principles_approved/constitution.md`; its `approved_*` corpus configs live in git
  history before that date.
- Adapter: HF `matboz/qwen3-32b-difficult-advice-lora` (the trained LoRA; pull to skip training).
- Eval harness: `anthropic-experimental/agentic-misalignment` (vendored + patched).

## When you finish a task
- Append a `docs/LOG.md` entry (most-recent-first): hypothesis → method → result → next steps, with absolute dates. LOG.md is for **experiments and major code changes only** — routine refactors, chores, and doc edits get no entry.
- Destroy any GPU instance and confirm 0 active.
