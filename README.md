<!-- ABOUTME: Root guide to the repository: the src/scripts/scratch code layout, the -->
<!-- ABOUTME: dashboard app, repository-wide conventions, and where the audit record went. -->

# Teaching Claude Why — replication and alignment auditing

Two lines of work on whether training a model on a written specification changes
its behaviour, plus a web frontend that presents the results.

```text
.
├── src/                  # correctness-critical reusable code (human-verified; import as src.*)
│   ├── openrouter.py, utils.py  #   shared OpenRouter client + utilities
│   ├── data/             #   data generation: synth/, the SFT/DPO dataset pipeline, mixtures
│   ├── train/            #   QLoRA SFT, DPO training, adapter merging
│   └── eval/             #   capabilities/ · misalignment/ (ODCV) · vulnerabilities/ (petri, surf)
├── scripts/              # pipelines: thin CLIs over src/ functions + GPU-box shell drivers
├── scratch/              # one-off and AI-generated scripts (default home for new code)
├── configs/              # OmegaConf YAML, one per pipeline step
├── tests/                # fast offline unit tests
├── constitutions/        # constitution / trait documents the specs point at
├── docs/                 # reference material + docs/replication.md (the run guide) + docs/LOG.md (research log)
└── dashboard/            # research-log frontend (Next/vinext), deployed on Netlify
```

**Run everything from the repository root.** `configs/`, `data/` and `output/`
are resolved against the current directory, and `uv sync`
installs `src/` editable so `import src.*` works from anywhere — locally and on
remote boxes alike; there are no `sys.path` tricks.

### Remote GPU boxes

This codebase runs on Linux GPU pods (see CLAUDE.md "Where code runs") — the GPU
stack is pinned in `pyproject.toml` and the lock is linux-only, so setup is just:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <this-repo> /root/work && cd /root/work
uv sync                          # everything, GPU stack included, src/ editable
uv run scripts/train/train_lora.py --config configs/train/lora_qwen3_difficult_advice_thinking.yaml
uv run scripts/run_eval.py --target <hf_path> --name agentic_misalignment
```

Plain `uv run` is correct on the pod — no `uv pip` layering, no `--no-sync`.
(`uv sync` does not resolve on macOS by design: vllm/bitsandbytes have no darwin
wheels. If local runs are ever needed, use conflicting dependency groups — see
the comment in `pyproject.toml`.)

| Area | What it is | How to work in it |
| --- | --- | --- |
| [`src/data/synth/`](src/data/synth/README.md) | Six-stage Teaching Claude Why difficult-advice data pipeline (self-contained package, formerly `synthdoc_v2`). | `uv run synth run --config configs/data/synth/difficult_advice.yaml --smoke` |
| `src/eval/vulnerabilities/` | Generalized Petri + SURF audit tooling from the completed MSM audit. Inspect's dependency pins conflict with the root env, so petri tools run in the nested project's env. | `uv run --project src/eval/vulnerabilities/petri/petri-subscription python src/eval/vulnerabilities/petri/<tool>.py --help` |
| [`dashboard/`](dashboard/README.md) | The research-log web app: datasets, eval runs, Petri results, findings. Self-contained Node project. | `cd dashboard && npm ci && npm run dev` |

## Repo layout
- `src/data/synth/` self-contained six-stage difficult-advice data pipeline (see above); its run config is `configs/data/synth/difficult_advice.yaml`.
- `src/eval/misalignment/internalization/` self-contained constitution-internalization proxy eval
  (Tier A). Measures whether a checkpoint *internalized* the constitution or memorized its surface
  behaviors, at every checkpoint, without a downstream training run.
  `uv run python -m src.eval.misalignment.internalization.cli run --smoke` runs it offline in ~10s with no API key. See
  `src/eval/misalignment/internalization/README.md`.
- `src/` reusable code (`llm.py`, `prompts.py`, `utils.py`); `src/experiments/` scripts.
- `configs/` OmegaConf YAML for every step, foldered by stage (`data/`, `train/`, `eval/`).
- `scripts/run_eval.py` THE eval entrypoint (CLAUDE.md "The eval framework"): serves each
  `--target` with vLLM and dispatches to a registered eval's `run()`; `scripts/data|train|gpu/`
  thin CLIs and provisioning.
- `src/eval/misalignment/agentic_misalignment/third_party/agentic-misalignment/` vendored eval harness (patched: `vllm/` provider, judge routing).
- `docs/claude_constitution_principles.md` the alignment target.
- `output/` all run artifacts; `LOG.md` append-only research log.
- Trained adapter: `matboz/qwen3-32b-difficult-advice-lora` on the HF Hub.

## The MSM audit record

The completed Petri + SURF audit of the Model Spec Midtraining checkpoints —
evidence, seeds, rubrics, eval logs, the 21 numbered research docs, JOURNAL.md,
and the provider/watchdog infrastructure — was removed from the tree tip during
the 2026-07-30 restructure. **Git history is the archive**; recover any of it
with:

```bash
git checkout b38da52 -- experiments/vulnerabilities
```

Its public-facing results remain in `dashboard/content/` (the focused-discovery
Petri run) and on Hugging Face; the reusable tooling lives on, generalized, in
`src/eval/vulnerabilities/` with each file citing its original at `b38da52`.

## Conventions

Read [`CLAUDE.md`](CLAUDE.md) — the agent operating guide and repository-wide
conventions — before generating data, running an experiment, or committing.
The rule that bites soonest:

> **Datasets, generated corpora, evaluation outputs and their caches go to
> Hugging Face, not into git.** HF repos are named
> `<YYYY-MM-DD>-<short-experiment-description>` using the date the data was
> *generated*. Every dataset card states the experiment, the generation date,
> and **which constitution or model spec it connects to** - written as `none`
> explicitly when it connects to none.

Code, configs, seeds, rubrics, analysis and reports stay in git. Bulk data does
not. New AI-generated one-off code defaults to `scratch/`; nothing imports from
`scratch/`.

## Credentials

Secrets never enter the repository. All credentials live in one gitignored
`.env` at the repo root — copy [`.env.example`](.env.example) and fill it in;
see CLAUDE.md's Secrets section for the rules. `.env`, `*.env`, `*.pem` and
`*.key` are ignored repository-wide from the root `.gitignore`, deliberately,
so the guard applies to every nested project.

- The replication pipeline reaches Claude through **OpenRouter** only.
- The audit tooling uses the **Anthropic API** (auditor/judge roles) plus GPU
  provider keys.

## Deployment

`dashboard/` deploys to Netlify on every push to the default branch. The root
[`netlify.toml`](netlify.toml) is the only Netlify configuration in the
repository: it sets the base directory, the build command and the publish
directory together. There is deliberately no second netlify.toml inside
`dashboard/` - the `Visualizer/` to `dashboard/` rename moved `base` and left
the old one behind, which is how the deploy config came to point at a file that
did not exist.

The repository-to-site link lives in the Netlify dashboard, not in git. If the
deploying repository changes, the site must be re-linked there - nothing in the
repository can restore it.

## History

This repository is the merge of two previously separate repositories, brought
in with full commit history rather than copied. On 2026-07-30 it was
restructured: the Python project flattened from `experiments/teaching-claude-why/`
into root-level `src/` + `scripts/` + `scratch/`, `Visualizer/` renamed to
`dashboard/`, and the frozen audit record removed at the tip (see above). The
record of how each finding was reached - including the ones that did not
survive scrutiny - is preserved in `git log`.


---

## Legacy run guide + capability evals (from kn/internalization-proxy — pre-restructure paths)

### 0. (Optional) Skip data generation — use the published dataset
The generated SFT data is on the HF Hub, so you can jump straight to fine-tuning (step 5) without
spending ~$74 on Sonnet 4.5. Two files in [`matboz/difficult-advice-qwen3`](https://huggingface.co/datasets/matboz/difficult-advice-qwen3):
`sft_dataset_thinking.jsonl` (2,119 examples **with `<think>` reasoning traces** — recommended) and
`sft_dataset.jsonl` (same, non-thinking).
```bash
mkdir -p data
uv run hf download matboz/difficult-advice-qwen3 sft_dataset_thinking.jsonl \
  --repo-type dataset --local-dir data
# then go straight to step 5 with configs/train/lora_qwen3_difficult_advice_thinking.yaml
# (its data_path already points at data/sft_dataset_thinking.jsonl)
```
The pre-trained LoRA adapter is also published — to skip training *and* generation entirely and go
straight to evaluation, point the eval framework at
[`matboz/qwen3-32b-difficult-advice-lora`](https://huggingface.co/matboz/qwen3-32b-difficult-advice-lora):
```bash
uv run scripts/run_eval.py --target matboz/qwen3-32b-difficult-advice-lora --name agentic_misalignment
```
### 1-2. Get the difficult-advice SFT data
The v1 generation code (`generate_difficult_advice.py` + `augment_thinking.py`) was deleted on
2026-08-03 — git history before that date has it, and its dataset card records the exact
provenance. Pull the finished dataset instead:
```bash
uv run hf download matboz/difficult-advice-qwen3 sft_dataset_thinking.jsonl \
  --repo-type dataset --local-dir data/
```
`sft_dataset_thinking.jsonl` carries a real first-person `<think>` trace per example — the
reasoning-preserving fix; naive SFT on single-blob answers makes Qwen3's chat template emit an
empty `<think></think>`, which trains the model to *stop reasoning*. New difficult-advice data
is generated with `synth` (see the synth section), which carries reasoning natively.

### 3. Provision + prepare the GPU box
```bash
# vast.ai example (any 80GB GPU works):
uv run vastai create instance <OFFER_ID> \
  --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel --disk 200 --ssh --direct
# On the instance (CRITICAL version pins — vLLM 0.8.5 needs transformers 4.51.3):
pip install --no-cache-dir vllm==0.8.5 "transformers==4.51.3" \
    trl==0.19.1 peft bitsandbytes datasets accelerate omegaconf fire wandb huggingface_hub
hf download Qwen/Qwen3-32B
# Copy this repo + the thinking dataset to /root/work on the instance.
```

### 4. Baseline eval (the framework serves the model itself)
```bash
uv run scripts/run_eval.py --target Qwen/Qwen3-32B --name agentic_misalignment
```
`run_eval.py` serves the target with vLLM on localhost, drives the vendored patched harness
(generate→experiments→classify via the OpenRouter judge), aggregates rates, stitches
self-contained rollouts, and pushes results to HF. A full model runs at its chat template's
own thinking default; adapters run in the mode stamped in their `training_meta.json`.

### 5. Train QLoRA

> **Skip training entirely** — evaluate the published adapter (step 6).

To train it yourself (on the pod):
```bash
# thinking-format (recommended): reasoning preserved
uv run scripts/train/train_lora.py --config configs/train/lora_qwen3_difficult_advice_thinking.yaml
# (non-thinking baseline arm: configs/train/lora_qwen3_difficult_advice.yaml)
```
Key config: r=32, 2 epochs, batch 4 × grad-accum 4, max_seq_len 2048, `assistant_only_loss: false`
(Qwen3's template has no `{% generation %}` markers, so assistant-only masking is all-zero).
Launch with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid fragmentation OOM.
Every train config declares `thinking: true|false`; the trainer validates it against the data
and stamps it into the adapter's `training_meta.json`.

### 6. Post-training eval + report
```bash
# same eval, adapter target — base model + thinking mode come from the artifact:
uv run scripts/run_eval.py --target matboz/qwen3-32b-difficult-advice-lora --name agentic_misalignment
# build the capstone dashboard (after pulling the summaries into output/eval_summaries/):
uv run scratch/reports/final_report.py
```
`final_report.py` writes `output/report/final_*/{report.md, dashboard.html, plots/}`.

---

## Second eval: ODCV-Bench (Qwen3.6-27B replication)

A second, independent agentic-misalignment benchmark:
[**ODCV-Bench**](https://odcvbenchmark.github.io/) ([arXiv 2512.20798](https://arxiv.org/abs/2512.20798),
[code](https://github.com/McGill-DMaS/ODCV-Bench)) — 40 scenarios in which an autonomous agent works
in a persistent bash sandbox under KPI pressure that conflicts with an ethical/legal/safety
constraint. Each scenario runs in two framings (**mandated**: optimise the KPI as a direct order;
**incentivized**: KPI pressure only), giving 80 runs. Four LLM judges (Claude-Opus-4.7, GPT-5.5,
Grok-4.20, Gemini-3.1-Pro-Preview) score each trajectory 0-5; the reported score is the **median
across judges**, and **misalignment rate (MR) = % of runs with median >= 3**.

**Replication target** (paper's main table): `Qwen3.6-27B — MR 43.8%, mean severity 1.67`
(mandated 45.0% / incentivized 42.5%).

### Setup
```bash
# The benchmark ships in-repo (tracked, pruned) at
# src/eval/misalignment/odcv/third_party/odcv-bench — see its VENDORED_FROM.txt.
# Nothing to clone.

# Docker must be usable WITHOUT sudo (each scenario builds two images):
sudo usermod -aG docker "$USER" && newgrp docker
docker context use default
```
Only `OPENROUTER_API_KEY` is needed — the agent and all four judges run through OpenRouter.

### Run
```bash
uv run pytest tests/test_odcv_metrics.py -q        # metric sanity (recovers 43.8%/1.67 from the paper's CSV)

uv run scripts/run_eval.py --target <hf_path> --name odcv smoke=true   # 2 scenarios, verifies wiring
uv run scripts/run_eval.py --target <hf_path> --name odcv              # full 80-scenario run
```
The framework serves the target and rewrites the endpoint to the docker-bridge address so
scenario containers reach it. Results land in
`output/odcv/<model_key>/<ts>/`: `agent_logs/` (trajectories), `evaluations/scores_<judge>.json`,
`rollout_manifest.json`, `run_meta.json`, and `results.json` (our MR/severity + bootstrap CI vs the
published numbers).

Both stages are **resumable** — re-running skips scenarios that already have a transcript and judge
calls that are already cached, so an interrupted run costs nothing to continue.

### Deviations from upstream (deliberate)
- `run_experiments.py` hardcodes all 12 paper models and runs scenarios strictly sequentially, with
  fixed container names and host port 5000. `src/experiments/odcv_rollout.py` runs **one** model and
  gives each scenario its own Compose project (+ an `orchestrator_api` network alias so the agent's
  hardcoded hostname still resolves), which allows `concurrency: 4`. Set `concurrency: 1` for
  strictly upstream behaviour; agent behaviour is unaffected either way.
- Judging reuses upstream's `evaluate_all_results.evaluation_routine` **verbatim** (imported, not
  reimplemented), so the rubric prompt, retry loop and JSON parsing are byte-identical.
- Upstream's README advertises `bootstrap_ci.py` / `paired_bootstrap.py` / `compute_paper_stats.py`,
  but those files are **absent** from the repo; `src/odcv.py` implements the scenario-level paired
  bootstrap CI described in the paper.

## Reproducing the reasoning check
`output/reasoning_probe_*.txt` compare `<think>` length of base vs LoRA. Naive SFT → 0 chars
(collapsed); the think-trace fix → 900-1600 chars of real reasoning, answers still correct.

## Third eval: capability regression (Arena-Hard SxS vs our own baseline)

The guardrail underneath the alignment results. It answers one question: does mixing
synthetic constitution documents into the SFT mixture cost us general capability? Data lives
in `configs/eval/arena_hard.yaml`, which is the single source of truth for arms, judge,
thresholds and decoding.

**50% is the target, not 100%.** This is a treated checkpoint measured against a sibling
arm, so a win rate near 50% means *no regression*. This is the "LMSYS SxS" number from the
GDM write-up: a pairwise preference win rate against the baseline arm. Nothing to do with
the public Arena leaderboard — no submission, no Elo.

**The baseline is `arm_b_synth10` (90/10), not arm A** — arm A's training recipe differs
(2 epochs, packing on, 2x tokens), so 50% means "no different from the low-dose arm", not
"no different from zero synthetic data". See the arm_a note in `configs/eval/arena_hard.yaml`.

### Results (2026-07-31, style-controlled win rate vs arm_b, hard_prompt, 95% CI)

| Arm | Mixture | Controlled WR | Read |
|---|---|---|---|
| `arm_base` | no SFT | 61.2% [53.4, 69.1] | **not a floor** — Qwen3.6-27B is already post-trained; external reference |
| `arm_a_synth00` | 100/0 (unmatched) | 58.1% [51.4, 64.6] | recipe-confounded, directionally high |
| `arm_b_synth10` | 90/10 | 50.0% A-vs-A, 95% ties | instrument sanity ✓ |
| `arm_c_synth20` | 80/20 | 49.2% [42.1, 56.3] (n=148) | flat — 20% synthetic is free |
| `arm_d_synth40` | 60/40 | **39.4% [34.5, 44.4]** (n=299) | **real regression — the mixture ceiling is between 20% and 40%** |
| `arm_e_synth100` | 0/100 canary | not trained | `adapter: null` |

Full artifacts (answers, judgments, metrics, report, figures) on HF:
[`LASR-Callum/qwen36-27b-capability-eval-arena-hard`](https://huggingface.co/datasets/LASR-Callum/qwen36-27b-capability-eval-arena-hard).
GDM-style figure: `scratch/reports/plot_arena_hard_winrate.py` (reads the latest report).
Full detail in `LOG.md` (2026-07-31 entry).

### Setup

```bash
# The harness ships in-repo (tracked, patched, pruned) at
# src/eval/capabilities/arena_hard/third_party/arena-hard-auto — see its
# VENDORED_FROM.txt. Nothing to clone, nothing to patch.
```

### Run

Run it through the eval framework — `run_eval.py` owns serving; add `--server <ssh-alias>`
to drive a remote GPU host (see CLAUDE.md "Where code runs"):

```bash
uv run scripts/run_eval.py --target <hf_adapter_or_model> --name arena_hard
uv run python scratch/reports/arena_hard_report.py       # CIs + figures + md mirror
uv run python scratch/reports/plot_arena_hard_winrate.py # GDM-style dose-response figure
```

(The pre-framework pod-per-arm runbook — retry wrappers, judging order, cost model from the
2026-07-31 first run — was `docs/arena_hard_eval_runbook.md`, deleted 2026-08-06; git
history is the archive.)

Before judging, eyeball ten raw generations from the arm's answers file. Do not skip it — a
chat-template mismatch reads as catastrophic capability loss but is purely a serving bug, and
it is the most common cause of "my finetune destroyed the model".

### What it measures, and the one thing that can make it lie

Two numbers per arm per slice, both reported:

- **Style-controlled win rate (primary).** A logistic/Bradley-Terry fit that removes the
  contribution of response length and markdown structure.
- **Uncontrolled win rate (secondary).** The gap between them is itself a finding about our
  corpus.

The style control is load-bearing. Pairwise LLM judges reward length and formatting, so *a
model that got wordier posts a higher win rate while being genuinely no better, or worse* —
and difficult-advice data is prose-heavy interpersonal writing, exactly the corpus likely to
produce that drift. Reading an uncontrolled number as "no regression" validates a broken
model with a broken instrument. Raw style deltas are logged per arm independently, because
style control tells you the win rate *net of* style while the deltas tell you how much drift
there was to control for.

Caveat the report prints for you: if an arm is longer than baseline by a similar proportion
on *every* prompt, length and model identity are the same column and no regression can
separate them. The report names any such feature rather than passing an uncontrolled number
off as controlled.

### Deviations from the spec and from upstream (deliberate)

| Decision | Why |
|---|---|
| Judge = `google/gemini-3-flash-preview`, validated against **GPT-4.1** not Sonnet | Claude generated our corpus, so a Claude validator would import the self-preference confound we avoided by picking Gemini. GPT-4.1 is a third family *and* arena-hard-auto's own validated judge. |
| **No batch API**, contra spec §4 | OpenRouter's `:batch` variants 404 on the synchronous chat endpoint — they need an async submit-and-poll API that arena-hard's threaded design can't use. Saves ~$15 on a ~$50 sweep; not worth a second client. |
| Paired bootstrap over **prompts**, not battles | Upstream resamples battles, which is unpaired. All arms see identical prompts, so pairing is free power. |
| Decisive verdicts (`A>>B`) **not upweighted** | Upstream counts them 3×, which stops the number being a win rate and breaks the §9 variance model. Reported separately as a diagnostic. |
| Style features scaled but **not mean-centred** | Upstream centres, which puts the intercept at the *mean observed* style delta — still carrying the drift we're removing. We keep the origin at "no style difference" so the controlled number answers the actual counterfactual. |
| Absolute benchmarks: **MMLU implemented** (see below), IFEval / GSM8K / HumanEval+ still deferred | MMLU closes the relative family's main blind spot — a pairwise judge cannot detect *both* arms degrading together. The remaining three are still out of scope; say so when reporting. |

Judge cost runs ~2× the spec's §11 estimate (~3,100 output tokens/question, not ~1,600):
Gemini 3 Flash spends 300–500 reasoning tokens per call even at `effort: low`. Verified by
A/B that `low` genuinely reduces them — it is not being ignored. Full sweep ≈ $50.

## Fourth eval: MMLU capability check (absolute, vs the Qwen base model)

The Arena-Hard eval above is *relative* — it can only say an arm is as good as another arm.
MMLU is scored against a fixed answer key, so every arm's number stands on its own and the
untuned Qwen base is a real anchor. This is the `absolute_benchmarks` block that
`configs/eval/arena_hard.yaml` defers.

**Runs a subset, not all 14,042 questions.** 10 questions × 57 subjects = **570**, drawn by a
seeded stratified sample. Every arm answers *literally the same questions*, which makes the
comparison paired: per-question outcomes line up across arms, so the interval on the
difference is much tighter than two independent intervals, and McNemar applies.

### Run

One vLLM process serves the base model plus every adapter as a LoRA module, so all arms are
measured by the same process on the same GPU with the same flags — decoding parity is a
property of the setup, not an assumption.

```bash
# 1. bring up the pod (same one the Arena-Hard eval uses; ~20-30 min to boot)
uv run python scripts/gpu/runpod_arena_hard.py up
uv run python scripts/gpu/runpod_arena_hard.py status --pod <id>

# 2. per target via the framework (serves, generates, grades, pushes):
uv run scripts/run_eval.py --target <hf_path> [<hf_path> ...] --name mmlu

# ... or drive the ladder steps yourself against an existing endpoint
uv run python src/eval/capabilities/mmlu_eval.py --arms all --endpoint <url>
uv run python scratch/reports/mmlu_report.py

# 3. ALWAYS tear the pod down
uv run python scripts/gpu/runpod_arena_hard.py down --pod <id>
```

Useful flags: `--smoke` (2 questions/subject, wiring check), `--per_subject 20` (tighter
intervals), `--arms arm_base,arm_d_synth40` (one or two arms), `--nothink` (thinking off —
results land in a separate tree; never compare a nothink arm to a thinking baseline).

Generations are cached per question and keyed on the **prompt content**, so re-runs are free,
raising `per_subject` only pays for the new questions, and editing the prompt template
invalidates the cache instead of silently mixing two formats into one accuracy number.

Deliverables land in `output/mmlu_eval/report/<mode>_<ts>/`: `mmlu_accuracy.png`,
`mmlu_by_category.png`, and `mmlu_results.md` (the greppable mirror).

### Reading the output — three things that make this eval lie

1. **Check `truncation_rate` before anything else.** A thinking model spends most of its
   budget inside `<think>`; if generation stops before the trace closes there is no visible
   answer, every question scores wrong, and it reads as catastrophic capability loss. That is
   gotcha 6 below. Fix `generation.max_tokens`, don't report the number.
2. **Check `parse_rate` on the base arm specifically.** Qwen3.6-27B base is not instruction
   tuned. The 5-shot prompt exists precisely to teach it the answer format by pattern, but if
   its parse rate still lands below the SFT arms, the base number is a **format floor, not a
   knowledge measurement** — compare `accuracy_parsed_only` and disclose the gap.
3. **The gate is on the interval, not the point estimate.** An arm passes only if the lower
   bound of its paired difference sits above −3pp. "Point estimate near zero" is not a pass;
   at small n it usually means the subset is too small to rule out a regression, and the fix
   is `--per_subject 20`.

The report also prints mean `<think>` length and empty-think rate per arm, so gotcha 2 (the
empty-`<think>` collapse) stays checkable — a thinking arm at ~0 words has stopped reasoning
regardless of what its accuracy says.

## `src/data/synth/` — synthetic chat data generation pipeline (separate, plug-and-play)

A **self-contained** package (formerly `synthdoc_v2`) replicating the six-stage difficult-advice
pipeline from Teaching Claude Why: constitution in, training corpus out. It shares nothing with
the code above — no imports either way — and hands off a finished corpus in SFT chat format
(with `reasoning_content` per example) that the training step can read directly. Every stage is
a separate, separately-cached step, so interrupted or budget-capped runs resume for free. Full
guide: [`src/data/synth/README.md`](src/data/synth/README.md).

The original config-driven `synthdoc` package (ablation sweeps, corpus snapshots, `control/`
prompt registry) was deleted on 2026-08-03 in favour of this simpler, more faithful pipeline;
it lives in git history before that date, and its published corpora remain on HuggingFace
(`LASR-Callum/synthdoc-<name>`).