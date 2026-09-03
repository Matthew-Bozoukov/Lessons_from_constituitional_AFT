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
A GPU comes from one command, `uv run runpod up`, in two shapes, one per half of the
pipeline: `--train <cfg>` (the training card + this repo at the commit you are on,
refused when that commit is not on origin) and `--eval <hf>` (the inference card, a
different and usually cheaper one, + vLLM at the version pyproject pins and the weights —
no repo, and no server: `uv run evals` owns serving). Naming the work picks the GPU from
`ModelProfile.gpu`, and the FORM of the name is the difference between the shapes: an arm
you are about to train exists only as a config, a target you are about to evaluate exists
on the Hub. `--count` is the number of GPUs, and `--clone-repo` adds this repo to an
`--eval` pod when you want to drive the eval on the box rather than over a tunnel.
**Never write a `POST /pods` yourself** — `src/infra/runpod.py` is the only place this
repo rents a GPU.
**Nothing tears a pod down for you**: `uv run runpod down --pod <id>` terminates and
verifies, `uv run runpod pods` lists what is still billing. The account and the credit are
shared, so check balances before big runs and flag spend over ~$20. Harder-won pod lessons live in `docs/GOTCHAS.md`.

### Data (`uv run synth`, `uv run mix`)

**`src/data/` needs no GPU** — data generation is API calls plus local files; runs locally.

### Train (`uv run train`)

One shape only — unlike an eval, training code must run on the GPU host directly:

```
uv run runpod up --name <you>-<arm> --train configs/train/<arm>.yaml --count N --push_env
#   (--train implies the clone: the pod gets this repo at your commit)
ssh -p <port> root@<ip> 'cd /root/work && uv run torchrun --nproc_per_node=N \
    scripts/train/train_lora.py --config configs/train/<arm>.yaml'
uv run runpod down --pod <id>
```

`--push_env` puts HF_TOKEN + HF_ORG (only) there, so the run pushes its own adapter; one
GPU needs no torchrun, `uv run train --config <cfg>` is enough. Be aware that when training *multiple* models it is more efficient to devote `N_GPUS//N_MODELS` GPUs to each model as opposed to training one model at a time using all GPUs. Any remaining GPUs can safely be absorbed into one of the model's training allocation but you should warn the user that it will likely not decrease the the total job time.

### Eval (`uv run evals`)

**The eval always runs where you type it; only the model moves.** `run_eval` owns serving
in both shapes below — it starts vLLM, pins the mode into the template, swaps LoRA
between arms and stops what it started — so there is exactly one place that decides how
a model under measurement is served, and it is the same code either way.

- **Serve on this machine** (no `--server`): `uv run evals --target <hf> --name <eval>`.
  For a driver that has its own GPU — and, on a pod rented with `uv run runpod up --eval
  <hf> --clone-repo` (both of those are `runpod up` args, not `evals` ones), the way to
  run the whole eval ON the box: `ssh <pod>`, then plain `uv run evals` there, with the
  pod's own `.env` for judging and the push. Not available to ODCV (see below).
- **Serve on a pod, drive here** — the normal shape, and THE one for ODCV (docker must
  run where you are, the model must not):

  ```
  uv run runpod up --name <you>-eval --eval <hf>
  uv run evals --name <eval> --target <hf> --server root@<ip>:<port>
  uv run runpod down --pod <id>
  ```

  `runpod up --eval` leaves the pod holding vLLM and the weights and nothing else of this
  repo's; `evals --server` (an `ip:port` or an alias from your own `~/.ssh/config`) makes run_eval
  start the server there over SSH and tunnel it back to localhost, so the eval, the
  judging and the HF push all happen here and the credentials for them never leave.
  At most `HF_TOKEN` (plus `HF_ORG`, which is not one) reaches the host, opt-in via
  `--push-env`, and only for gated weight pulls. `check_ready` fails fast, naming
  `runpod up --eval`, on a host with no vLLM. An arm ladder reuses the one server:
  consecutive targets sharing base model + mode swap the LoRA instead of restarting.

Notes:
- New code should be written with these workflows in mind. For example, they should expect target models to be from Hugging Face and served as a vLLM endpoint.
- **ODCV must drive where docker works** — a laptop with Docker Desktop, never a RunPod
  pod: unprivileged containers cannot create the per-scenario Compose networks. The model
  it drives is served on a RunPod pod: local docker, remote model. Its scenario
  containers reach the tunnelled endpoint at the host address for their platform
  (`172.17.0.1` on linux, `host.docker.internal` on Docker Desktop) — `--server-bind` and
  `container_host_address()` between them handle both. `docker_preflight` refuses
  unusable hosts with a specific remedy.
- **Exception**: `src/eval/audits/` predates this rule and does not yet conform
  (own nested env, own workflow) — see `docs/TODO.md`.

## Where things go (keep this structure)

```
src/                  reviewed, reusable code (installed editable; import as src.*)
  infra/                what the pipelines run ON: runpod.py (the ONE place a GPU is rented)
                        + endpoints/{openrouter,vllm}.py (the clients models are reached through)
  chat/                 `uv run chat` — talk to the organisms we train (repl + organism discovery)
  utils.py              io/json + provenance helpers, transcript rendering
  naming.py             THE naming law: one shape per stage, its builders, and the
                        lint that blocks a push
  model_profile.py      ModelProfile registry: verified per-family render/mask/serve
                        facts, AND MODEL_KEYS — what each base model is named
  huggingface.py        HF tokens, dataset-card contract, push/download helpers
  data/synth/           constitution-grounded generation engine (the config IS the document type)
  data/mixture/         training-mixture builder + one adapter per source + spec filter
  train/                train_lora.py, merge_lora.py
  eval/                 registry in __init__.py; one directory per eval:
    capabilities/         arena_hard/ (the model-vs-model eval), mmlu/,
                          swebench_mini/
    misalignment/         odcv/, agentic_misalignment/, psychosis/, internalization/,
                          moralbench/ (declarative moral-foundations probe; docs/moralbench.md)
                          (odcv + agentic_misalignment vendor PATCHED harnesses in third_party/)
    audits/               petri/ + surf/ audit tooling
configs/              OmegaConf YAML, one per step; NEVER hardcode hyperparams in scripts
  data/synth/           one config per document type (superseded → archive/)
  data/mixture/         0 (the base blend) + <styles>[-<variant>] arms; archive/
  train/                <model>-<mix>-<pct> — one file per arm, undated
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
- **Names are built, not typed** — one shape per pipeline stage, and the only thing a
  human chooses is the style-type they name a synth/mixture config. See "Artifacts and
  configs" below.
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

## Artifacts and configs: naming and storage

An artifact is anything a KEY PIPELINE STAGE produced — a corpus, a mixture, an adapter,
an eval run, a cache. Two rules cover all of them: where it lives, and what it is called.

**It lives on Hugging Face.** This repo holds code, configs and small analysis outputs;
it does not hold bulk data. `output/` is scratch for fast iteration, never the canonical
copy. Corpora, response sets, transcripts, judge outputs, embeddings, activation caches —
anything someone would rather fetch than regenerate — get pushed. Weights, adapters and
anything past a few megabytes never enter git; the link to the HF repo does, so it is not
only in someone's memory.

**Figures are exempt from the STORAGE rule and from nothing else.** Plotting and write-up
code is per-experiment, so it lives in `scratch/` and writes to `output/`; nothing plotted
gets pushed, because what a plot says is already on the Hub in the results it read. **The
naming law below applies to plots in full**: a figure carries the date it was produced
and an unambiguous subject, exactly like a corpus or an adapter.

**Its name says which stage made it and which arm it belongs to.** One law
(`src/naming.py`), one shape per stage, built by code and typed by nobody:

```
synth    <date>-<style>-synth                  2026-09-01-difficult-advice-synth
mix      <date>-<style>-mix                    2026-09-03-difficult-advice-mix
model    <date>-<model>-<style>-<seed>         2026-09-04-qwen36-difficult-advice-0
eval     <date>-<eval>-<model name, undated>   2026-09-05-odcv-qwen36-difficult-advice-0
pooled   the same, with the seed dropped       2026-09-06-odcv-qwen36-difficult-advice
```

Two spellings of that one grammar — `to_hub`/`to_local` convert and nothing else may:

```
local (files, run dirs, figures, arm labels)  2026-09-04_qwen36_da_0
hub   (an HF repo id after the org)           2026-09-04-qwen36-difficult-advice-0
```

- **ONE HUMAN INPUT, ONE PLACE: the style-type**, which is the stem of the synth or
  mixture config that produced the data. It carries the ablation and says WHAT THIS ONE
  CHANGES (`da_length_capped`, never `sonnet_v2`). Nothing else in any name
  is chosen: the date comes from the clock at launch, the model from `MODEL_KEYS`
  (`src/model_profile.py`, beside that model's other facts), the eval from
  `EvalSpec.key`, the seed from the training config. None of them is typed, so
  none of them can drift from the thing it describes.
- **The date is the date the thing was PRODUCED** and every artifact carries exactly one:
  a model organism enters its eval run's name WITHOUT its date, so the run says which arm
  it measured and still dates only itself.
- **CONFIGS ARE UNDATED AND UNSEEDED.** A config names an ARM; a run produces an
  ARTIFACT. `configs/data/synth/<style>.yaml`, `configs/data/mixture/<styles>.yaml`
  (the synthetic source keys, SORTED and hyphenated — derived, so one set of corpora has
  exactly one name), and
  `configs/train/<model>-<mix>-<pct>.yaml` — the adapter's own repo name minus the date
  and the seed. A seed is a launch argument (`seed=1`), so three replicates are one
  config, not three files. `configs/eval/<eval>.yaml` is the exception in spelling: an
  eval config is a KIND and carries the eval's full registered name
  (`agentic_misalignment`), checked against the registry itself.
- **The style vocabulary is SHORT and therefore load-bearing** — `da`, `par`, `pad`, `pc`.
  Nothing in the code can tell two expansions of one code apart (`par` has meant both
  post-action-retrospection and pre-action-deliberation here); the config that carries a
  code is what settles its meaning, and it is the only thing that does.
- **The config that produced an artifact travels WITH the artifact** — training pushes
  the RESOLVED config (`train_config.yaml` + `training_meta.train_config`), because
  configs are edited in place and a path alone would name a file that no longer says what
  it said. Re-running an arm means fetching its config from the Hub.
- **Registries are edited when the project gains something, never per run**: a new base
  model adds a line to `MODEL_KEYS` (`src/model_profile.py`), a new eval declares `key=`
  in its `EvalSpec` (`src/eval/__init__.py`). Both are required, so both are decided once.
- **Hardware, rank and launcher detail are not identity** — the config records `2xh200`,
  the name does not.
- **What no stage shape covers still gets a date and a subject** — `artifact_name()` for
  answer caches, probe sweeps, one-off harnesses. They get no field structure, and
  nothing tries to parse one out of them. The lint never parses a name into fields.
- **Vocabulary**: `qwen3` = Qwen3-32B, `qwen36` = Qwen3.6-27B; ratios read
  `<synth>_<tulu>` (`20_80` = 20% difficult-advice); arms are `base_*`,
  `ft_<ratio>[_<ablation>]`, `tulu_100`.

**Every upload carries a card** (`README.md` in the HF repo): `experiment`,
`date_generated`, `constitution`, `source_repo` (this repo @ the generating commit),
`models` (with revision pins), `generation_config` (sampling settings and seeds),
`schema`, `provenance` (the exact command to regenerate it). Write `constitution: none`
explicitly rather than omitting it: nearly everything here is about whether training on a
written specification changes behaviour, so which one a dataset relates to is what a
future reader needs most, and it is the field most easily lost.

**Enforced where names are MADE.** The builders in `src/naming.py` validate as they mint,
which is the only moment a check can still change the outcome — an unregistered model or
an unnameable mixture fails before the first GPU-hour, not at the push after it.
`src/huggingface.py::gate_push` is the backstop on every push (the name, plus its date
against the card's `date_generated`), and `.git/hooks/pre-push` (install once:
`bash scripts/hooks/install.sh`) runs `src.naming.lint_repo` over the tracked tree — the
same lint `tests/test_naming.py` runs in the suite. The lint checks the two things a
human writes: config stems (undated, unseeded, shaped for their stage) and literal figure
filenames (dated). READS are never checked: pointing at a repo from before this law does
not make another badly named one, and refusing the read would only force a copy.

## The pipeline (each stage = one alias + one config)

Every stage is a console alias from `[project.scripts]`, so the shape is always
`uv run <job> --config <yaml>`. Stages 1–3 take `--smoke`.

1. `uv run synth run --config configs/data/synth/<type>.yaml` — constitution-grounded generation; the config IS the document type, so read the one you are running (`ls configs/data/synth/`) rather than a list here.
   Know which arm is the current baseline before you build on one or compare against one: `docs/BASELINES.md`.
2. `uv run mix --config configs/data/mixture/<name>.yaml` — budgeted training mixture of model-agnostic interchange rows (reasoning as `reasoning_content`, rendered at train time), with optional spec-filter stage and HF push checkpoints; `balance_by: trait_id` on a source spec trait-balances the difficult-advice share.
3. `uv run train --config configs/train/<model>-<mix>-<pct>.yaml [seed=N]` — QLoRA SFT (runs on the GPU box). Pushes the adapter to HF with `training_meta.json` — the thinking stamp (declared as `thinking:` in the train config, validated against the data) that the eval framework infers mode from.
4. `uv run evals --target <hf_path> --name <eval>` — THE eval entrypoint for every registered eval; see "The eval framework" below.

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
sha; checkpoints stay local, and the run pushes one artifact back: the final adapter,
carrying `training_meta.json` (thinking stamp, the RESOLVED config with launch overrides
applied, the exact `command`, the pinned dataset `{repo, file, revision}` and the base
model's `base_model_revision`) plus `train_config.yaml`. Every stage records its command
and pins what it read — a synth manifest its top-ups, a mixture its streamed sources'
revisions, an eval its `target_revision` — so an artifact's metadata is enough to rerun
it as it was run. What no pin can hold still: API models behind a fixed id, and GPU
nondeterminism. The artifact is the record.

## The eval framework (the contract every eval follows)

Every eval is one invocation of the single entrypoint, driven from anywhere ("Where code
runs"; `--server <address|alias>` names the GPU host run_eval should start vLLM on, and
omitting it serves on this machine):

```
uv run evals --target <hf_path | provider:model-id> [...] --name <eval> [key=value ...]
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
  accept one — `EvalSpec.supports_api_target` (mmlu, arena_hard, psychosis,
  moralbench);
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
  `src/eval/capabilities/arena_hard/`, `src/eval/misalignment/psychosis/` — with a
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
- Tear down every pod YOU started — `uv run runpod down --pod <id>` — and confirm with
  `uv run runpod pods` that none of yours is left. The account is shared: a pod you did
  not provision is reported, never terminated.
