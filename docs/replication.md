<!-- ABOUTME: Full replication guide for the difficult-advice result on Qwen3-32B. -->
<!-- ABOUTME: Anyone with the listed keys + a single H100 can reproduce it end to end. -->

# Teaching Claude Why — "difficult advice" replication (Qwen3-32B)

Replicates the **difficult-advice** result from Anthropic's *[Teaching Claude Why](https://www.anthropic.com/research/teaching-claude-why)*:
training on a small, **out-of-distribution (OOD)** set — where *the user* (not the AI) faces an
ethically ambiguous situation in which they could reach a reasonable goal by violating a norm or
subverting oversight, and the assistant gives a nuanced, constitution-aligned answer that
**deliberates about its values** — reduces **agentic misalignment** on held-out honeypots
(blackmail/leaking), which are far from the training distribution.

## Headline result

Qwen3-32B, QLoRA on **1.52M tokens** of OOD difficult-advice data, evaluated on the open-source
[`agentic-misalignment`](https://github.com/anthropic-experimental/agentic-misalignment) honeypots
(Sonnet 4.5 judge), 600 samples/condition:

| Training format | Baseline | + Difficult-advice SFT | Relative reduction |
|---|---|---|---|
| non-thinking answers | 15.0% | 12.7% | −15% |
| **thinking (value reasoning in `<think>`)** | **19.3%** | **8.0%** | **−58.5%** |

Two findings:
1. The OOD data **generalizes** to the agentic honeypots (replicates the paper's core claim).
2. Putting the value deliberation **inside the model's `<think>` reasoning** both **preserves the
   model's reasoning ability** *and* ~4×'s the alignment effect — the "teaching *why*" thesis:
   the reasoning is the mechanism. (Blackmail stays ~0% for Qwen3; the signal is in leaking.)

## Differences from the paper (by design / necessity)
- Base model **Qwen3-32B** (not Claude).
- Data generated with **Claude Sonnet 4.5 via OpenRouter** (no Anthropic key needed; the paper used Opus 4.5).
- Fine-tuning via **QLoRA** on one H100 (not full character training).
- Eval = the **public** agentic-misalignment benchmark (a stand-in for the paper's internal assessment).

---

## Prerequisites

- **Python** managed by [`uv`](https://docs.astral.sh/uv/) (local machine, no GPU needed for data-gen/analysis).
- **One 80GB GPU** for training + serving (this run used a [vast.ai](https://vast.ai) H100 SXM, ~$2-3.5/hr).
- **API keys** in a `.env` file at repo root (gitignored):
  ```
  OPENROUTER_API_KEY=...   # data generation + eval judge (Sonnet 4.5, Gemini grader)
  HF_TOKEN=...             # download Qwen3-32B; push/pull the LoRA adapter
  WANDB_API_KEY=...        # (optional) training curves
  VAST_API_KEY=...         # (optional) if using vast.ai for the GPU
  ```

## Setup (local)
```bash
uv sync                      # installs the local deps from pyproject/uv.lock
uv run pytest tests/ -q      # sanity: JSON-extraction unit tests
```

---

## Pipeline

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
straight to evaluation (step 6), point the eval framework at
[`matboz/qwen3-32b-difficult-advice-lora`](https://huggingface.co/matboz/qwen3-32b-difficult-advice-lora)
directly (serving, base-model resolution and thinking mode are handled by `run_eval.py`).
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
is generated with `synthdoc` (see below), which carries reasoning natively.

### 3. Provision + prepare the GPU pod
```bash
# vast.ai example (any 80GB GPU works):
uvx vastai create instance <OFFER_ID> \
  --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel --disk 200 --ssh --direct
# On the pod:
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <this-repo> /root/work && cd /root/work
uv sync          # the GPU stack (vllm/transformers/trl/peft) is pinned in pyproject
# Copy your .env + the thinking dataset to /root/work. Plain `uv run` from here on.
```

### 4. Baseline eval (the framework serves the model itself)
```bash
uv run scripts/run_eval.py --target Qwen/Qwen3-32B --name agentic_misalignment
```
`run_eval.py` serves the target with vLLM on localhost, drives the vendored harness
(generate→experiments→classify via the OpenRouter judge), aggregates per-condition rates,
stitches self-contained rollouts, and pushes results to HF. A full model is evaluated at its
chat template's own thinking default; adapters are evaluated in the mode stamped in their
`training_meta.json` — nothing is declared at eval time.

### 5. Train QLoRA

> **Skip training entirely** — evaluate the published adapter (step 6 with
> `--target matboz/qwen3-32b-difficult-advice-lora`).

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
and stamps it into the adapter's `training_meta.json` (pushed to HF with `hf_repo:` set).

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
# The benchmark is vendored in-repo (committed, patched — re-clone only to refresh):
git clone https://github.com/McGill-DMaS/ODCV-Bench.git src/eval/misalignment/third_party/odcv-bench
# pinned at upstream commit 7353f1cf4b2579a3a8a5b8a5061d7c7d41f60668

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
scenario containers reach it. Results land under `output/odcv/<model_key>/<ts>/`: `agent_logs/`
(trajectories), `evaluations/scores_<judge>.json`, `rollout_manifest.json`, `run_meta.json`, and
`results.json` (our MR/severity + bootstrap CI vs the published numbers).

Both stages are **resumable** — re-running skips scenarios that already have a transcript and judge
calls that are already cached, so an interrupted run costs nothing to continue.

### Deviations from upstream (deliberate)
- `run_experiments.py` hardcodes all 12 paper models and runs scenarios strictly sequentially, with
  fixed container names and host port 5000. `src/eval/misalignment/odcv_rollout.py` runs **one** model and
  gives each scenario its own Compose project (+ an `orchestrator_api` network alias so the agent's
  hardcoded hostname still resolves), which allows `concurrency: 4`. Set `concurrency: 1` for
  strictly upstream behaviour; agent behaviour is unaffected either way.
- Judging reuses upstream's `evaluate_all_results.evaluation_routine` **verbatim** (imported, not
  reimplemented), so the rubric prompt, retry loop and JSON parsing are byte-identical.
- Upstream's README advertises `bootstrap_ci.py` / `paired_bootstrap.py` / `compute_paper_stats.py`,
  but those files are **absent** from the repo; `src/eval/misalignment/odcv/odcv.py` (metrics) and
  `src/eval/misalignment/odcv/stats.py` (the scenario-level paired bootstrap CI) implement what the
  paper describes.

## Third eval: AI-psychosis red-teaming

A multi-turn sycophancy/safety eval from Tim Hua's
["AI-Induced Psychosis: A shallow investigation"](https://www.lesswrong.com/posts/iGF7YcnQkEbwvYLPA/ai-induced-psychosis-a-shallow-investigation)
([code](https://github.com/tim-hua-01/ai-psychosis)): a red-teamer LLM role-plays one of nine
personas descending into psychosis over 12 turns (simulation theory, divine missions, new
mathematics, ...); after each target reply a judge grades it on 5 core metrics (mental-health
referral, pushback 0-2, delusion confirmation -1-4 + quotes) and 9 CBT-therapy-manual criteria
(1-5, 0 = not yet applicable).

Unlike ODCV this is a **native reimplementation**, not a vendored harness — upstream is a
single inspect-ai script plus R analysis. Only the scientific inputs are copied verbatim
(9 persona files + red-teamer/grader prompts, MIT, pinned SHA:
`src/eval/misalignment/psychosis/assets/README.md`); the loop, judging and stats are
`src/eval/misalignment/psychosis/` on the repo's own endpoint clients.

```bash
uv run scripts/run_eval.py --target <hf_path> --name psychosis smoke=true  # 1 persona x 3 turns
uv run scripts/run_eval.py --target <hf_path> --name psychosis             # 9 personas x 12 turns
```

Config `configs/eval/psychosis.yaml`: red-teamer `x-ai/grok-3` (upstream's default — Grok-4 now
refuses the roleplay, but still grades), judge `x-ai/grok-4` (the write-up's published grader;
Gemini 2.5 Pro *authored* the therapy rubric but never graded), both via OpenRouter. Outputs under
`output/psychosis/<model_key>/<ts>/`: `rollouts/<persona>.{md,json}` (self-contained transcripts
incl. red-teamer strategy notes, target reasoning and per-turn grades), `grades.{jsonl,csv}`,
and summary metrics (referral rate, delusion-collusion rate = share of delusional turns rated
"plays along"/"pushes", therapy-criterion means, per-turn escalation curves, empty-think rate).

Deviations from upstream (deliberate): grading runs after each conversation completes rather
than interleaved (equivalent — the judge never influences the conversation — and fully
parallel); judge temperature 0; a red-teamer completion without a `<message>` block is
regenerated once before the persona fails (upstream crashes the persona on the first refusal);
`-1`/`0` sentinel grades are excluded from means (upstream's write-up averaged raw columns); the target's reasoning goes back into its own history as
`reasoning_content` per the preserve-thinking policy below (the served template decides whether
to render it; the red-teamer still sees only the visible reply, as upstream), and the trace is
preserved in rollouts and shown, fenced, to the judge as upstream did. Verify on a live
endpoint that vLLM forwards request-side `reasoning_content` into the template before trusting
multi-turn numbers.

## Fourth eval: SWE-bench (standardized baseline, `swebench_mini`)

Agentic coding capability under a scaffold that is deliberately **not ours**: upstream
mini-SWE-agent v2.2.1, pinned by a committed lockfile, with its official `swebench.yaml`
passed through unedited (config sha256
`f90e7baa84c9e36e535cf4f37ee39e6e08c05964d55b5f56ed12cad7f817ffa8`). One rollout per task, no
retries or reranking; grading by the pinned official SWE-bench harness (`swebench==4.1.0`).
Full detail — deviations, provenance, what to verify before the first run — in
[`src/eval/capabilities/swebench_mini/README.md`](../src/eval/capabilities/swebench_mini/README.md).

Two phases, because only the first needs a GPU:

```bash
# 1. Rollouts: needs the served model AND docker (the agent works inside SWE-bench images)
uv run scripts/run_eval.py --target Qwen/Qwen3.6-27B --name swebench_mini

# depth: repo-stratified, nested (10% is a strict subset of 20%; extending reuses rollouts)
uv run scripts/run_eval.py --target <hf> --name swebench_mini subset.fraction=0.2

# 2. Grading: docker + CPU only — run it AFTER destroying the GPU box
uv run scripts/eval/swebench_mini_grade.py --run-dir output/swebench_mini/<key>/<ts>
```

Disk is the constraint, on both hosts: the agent and the harness share the same per-instance
images, and the harness wants ~120GB at its default `cache_level=env` for the full benchmark
(a stratified 10% slice touches far fewer environment images). Provision the rollout box with
~300GB.

Report as `<model> + mini-SWE-agent <version> (config <sha>), <dataset>@<revision>
[n/N instances, subset <hash>], pass@1` — and read `patch_rate`, `no_tool_call_rate` and
`exit_statuses` before believing the score, since lost tool-call formatting and context
overflow both masquerade as incapability.

## The preserve-thinking policy (2026-08-04)

Repo-wide default for training data and serving, everything family-specific centralized in
`ModelProfile` (`src/utils.py`; Qwen3.6 is the only verified profile — Qwen3's thinking
template prefills nothing, so it is deliberately refused until verified):

- **Data**: `build_mixture.py` renders with the profile's kwargs (`preserve_thinking=True`),
  so EVERY assistant turn carries a think block — reasoning where the source has it, the empty
  marker where it does not. HF sources must declare `reasoning: native|none|strip` (`strip` =
  deliberate pre-policy no-think rendering, for nothink control arms only).
- **Loss**: the generation-boundary mask (`src/train/masking.py`, not configurable) masks
  exactly what the model never generates and supervises what it does. On a real reasoning
  turn that means the `<think>\n` prefill is masked and the trace + `\n</think>` close carry
  loss; on an empty turn the WHOLE marker is masked and only the answer carries loss — a
  healthy Qwen3.6 never closes an empty think block itself (LOG 2026-08-04 probe: it reasons
  even on trivial questions; in nothink mode the full marker is prefilled), so supervising
  the empty close would train the collapse. Rows are tokenized in segments cut at each
  forced-span edge so token merges cannot weld masked to supervised. `src/train/mask_gate.py`
  re-verifies this with an independent parser plus a think census before every training run.
- **Serving**: `pin_template` pins `preserve_thinking` alongside `enable_thinking`, and
  multi-turn eval loops send each turn's reasoning back as `reasoning_content`.

## Reproducing the reasoning check
`output/reasoning_probe_*.txt` compare `<think>` length of base vs LoRA. Naive SFT → 0 chars
(collapsed); the think-trace fix → 900-1600 chars of real reasoning, answers still correct.

## `src/data/synthdoc/` — synthetic chat data generation pipelines (separate, plug-and-play)

A **self-contained** package (formerly `synthdoc_v2`) with one config-driven engine; the config's
`stages:` list — prompts included — fully defines the document type, and
`scripts/data/synthdoc/build_dataset.py` executes it. `configs/data/synthdoc/difficult_advice.yaml` replicates the
six-stage Teaching Claude Why recipe:
segment the constitution, generate scenarios, draft the prompt, refine it against the full
constitution, generate a response, and rewrite the response against the target trait. It shares
nothing with the code above — no imports either way — and hands off a
finished corpus in SFT chat format (with `reasoning_content` per example) that the training step
can read directly. Full guide: [`src/data/synthdoc/README.md`](../src/data/synthdoc/README.md)
(stage table, models, caching).

```bash
uv run synthdoc segment                                   # constitution -> traits, no API calls
uv run scripts/data/synthdoc/build_dataset.py --config configs/data/synthdoc/difficult_advice.yaml --smoke
uv run scripts/data/synthdoc/build_dataset.py --config configs/data/synthdoc/difficult_advice.yaml
uv run scripts/data/synthdoc/build_dataset.py --config configs/data/synthdoc/difficult_advice.yaml --estimate
uv run pytest tests/test_difficult_advice.py -q           # offline, no API key
```

Every stage writes a complete snapshot (`output/synthdoc_v2/<run>/stage_<n>_*.jsonl` — the
directory keeps its historical name so old runs stay resumable) and mirrors it to the HF dataset
repo named in the config, so an interrupted or budget-capped run resumes from the last completed
stage at no cost.

The same package also hosts the **model-eval-model pipeline**: documents where the
model reasons about a response to one of the same scenarios (its own or "an AI assistant's") and
works out whether it was the right call. It consumes a *completed* difficult-advice run — same
scenario bank, so arm differences are attributable to format — and fail-fasts if the source run's
constitution sha differs from the config's. All five cells are implemented: the reasoning-only
`control`, the other-attribution critiques `m4`/`m3` (the flawed side via minimal-pair
perturbation), and the self-reflection cells `m2`/`m1` — the headline experiment — whose records
are multi-turn with the evaluated response in the model's own prior turn, trained with
`supervise: "final"` (only the last assistant turn carries loss; threaded through
`convert_synthdoc_qwen.py` → `build_mixture.py` → `masking.py`). See the cell table in
[`src/data/synthdoc/README.md`](../src/data/synthdoc/README.md).

```bash
uv run scripts/data/synthdoc/build_dataset.py --config configs/data/synthdoc/model_eval_model.yaml --smoke   # 2 docs per enabled cell
uv run scripts/data/synthdoc/build_dataset.py --config configs/data/synthdoc/model_eval_model.yaml
uv run synthdoc check --config configs/data/synthdoc/model_eval_model.yaml --run_dir output/model_eval_model/<ts>   # validity gates
uv run scripts/data/synthdoc/build_dataset.py --config configs/data/synthdoc/model_eval_model.yaml --estimate --measured output/model_eval_model/<smoke>/manifest.json
uv run pytest tests/test_model_eval_model.py -q                        # offline, no API key
```

### The self-reflection document type

`configs/data/synthdoc/self_reflection.yaml` inverts who is tempted: **the agent itself** — working
autonomously, it could protect its own position (or the objective it was deployed to pursue) by
an action that is deniable rather than overtly wrong; it deliberates in its own value language,
declines, and still does its job. Composition (prose/agentic form, 15% multi-turn, 12% control
slice where the right move is to ACT, motive split) is assigned deterministically from scenario
ids; stage 6 enforces a voice contract in code (rule-vocabulary lint, reject-and-retry).

```bash
uv run scripts/data/synthdoc/build_dataset.py --config configs/data/synthdoc/self_reflection.yaml --smoke
uv run scripts/data/synthdoc/build_dataset.py --config configs/data/synthdoc/self_reflection.yaml
# a one-off variant (a top-up, a different size) without forking the config:
uv run scripts/data/synthdoc/build_dataset.py --config configs/data/synthdoc/self_reflection.yaml \
    --overrides "total_scenarios=144,id_prefix=b"
uv run pytest tests/test_self_reflection.py -q            # offline, no API key
```

Generated 2026-08-03 (pre-restructure code, same prompts): **592 records / 1.56M Qwen3.6
tokens**, HF `LASR-Callum/2026-08-03-synthdoc-self-reflection`. Consumed by
`configs/data/mixture/qwen36_table2_80_synthdoc_self_reflect_20.yaml` at a pinned revision.

Two things that bite when using this corpus:

- **Render with reasoning preserved on every assistant turn.** 15.9% of its records are
  two-exchange conversations, and a naive Qwen3.6 render emits `<think>` on the final assistant
  turn only — earlier turns come out as bare content, and supervising those trains the model to
  answer without reasoning. The repo's preserve-thinking policy handles this; a mixture built
  outside it does not.
- **`trait_weights` are checked against the constitution actually loaded.** The 12-principle
  document was re-cut to ten units on 2026-08-04 while keeping its folder name, so a config
  written against the old cut would silently generate a different corpus. Planning now fails
  loudly instead.

The original config-driven `synthdoc` package (ablation sweeps, corpus snapshots,
`control/` prompt registry) was deleted on 2026-08-03 in favour of this simpler, more faithful
pipeline; it lives in git history before that date, and its published corpora remain on
HuggingFace (`LASR-Callum/synthdoc-<name>`).

## Repo layout
- `src/data/synthdoc/` constitution-grounded data generation: one config-driven engine; the config's `stages:` list (prompts included) defines the document type; run via `scripts/data/synthdoc/build_dataset.py` (`configs/data/synthdoc/difficult_advice.yaml`, `configs/data/synthdoc/model_eval_model.yaml`).
- `src/` reusable code (`endpoints/`, `utils.py`, `data/`, `train/`, `eval/`); `scripts/` thin pipeline CLIs foldered by stage (`data/`, `train/`, `gpu/`); `scratch/` one-offs.
- `configs/` OmegaConf YAML for every step, foldered by stage (`data/`, `train/`, `eval/`).
- `scripts/run_eval.py` THE eval entrypoint (see CLAUDE.md "The eval framework"): serves each `--target` with vLLM and dispatches to the registered eval's `run()`.
- `src/eval/misalignment/third_party/` vendored eval harnesses, patched (`agentic-misalignment`: `vllm/` provider + judge routing; `odcv-bench`).
- `constitutions/` alignment targets, one folder each with `constitution.md` + `rationale.md`:
  `claude_distilled_07_principles_approved/` is the current target for the difficult-advice
  prompts; synthdoc's default is `claude_distilled_12_principles_mid/` (since 2026-08-03;
  the v1 doc it replaced is in `archive/claude_distilled_8_principles_v1/`). See
  `constitutions/README.md`.
- `output/` all run artifacts; `docs/LOG.md` append-only research log.
- Trained adapter: `matboz/qwen3-32b-difficult-advice-lora` on the HF Hub.

## Gotchas (things that will bite you)
1. **`transformers==4.51.3`** with vLLM 0.8.5 — newer transformers throws
   `Qwen2Tokenizer has no attribute all_special_tokens_extended`.
2. **Empty `<think>` collapse** — train with `reasoning_content` traces (step 2) or the model stops reasoning.
3. **QLoRA OOM** at batch 8 × 2048 on 80GB — use batch 4 + `expandable_segments`.
4. **Judge routing** — the harness's `_detect_provider` matched substring "claude" → Anthropic before the
   `/`-prefix rule; the vendored copy is patched so `anthropic/claude-sonnet-4.5` routes to OpenRouter.
5. **Match eval mode to training** — thinking mode is inferred from the adapter's
   `training_meta.json` and pinned into the chat template at serve time; never compare arms
   whose modes differ.
