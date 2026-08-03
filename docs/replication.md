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

### 7. (Optional) Cross-check on the Inspect harness
Serve base+adapter, SSH-tunnel port 8000 to the machine that has
[`inspect_evals`](https://github.com/UKGovernmentBEIS/inspect_evals), then:
```bash
bash scripts/eval/run_leaking_inspect.sh difficult_advice explicit america ft_explicit_america
```
(runs `inspect_evals/agentic_misalignment`, leaking scenario, gemini-flash grader).

---

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
  but those files are **absent** from the repo; `src/eval/misalignment/odcv.py` (metrics) and
  `src/eval/misalignment/stats.py` (the scenario-level paired bootstrap CI) implement what the
  paper describes.

## Reproducing the reasoning check
`output/reasoning_probe_*.txt` compare `<think>` length of base vs LoRA. Naive SFT → 0 chars
(collapsed); the think-trace fix → 900-1600 chars of real reasoning, answers still correct.

## `src/data/synthdoc/` — synthetic chat data generation pipeline (separate, plug-and-play)

A **self-contained** package (formerly `synthdoc_v2`) replicating the six-stage difficult-advice
pipeline from Teaching Claude Why: segment the constitution, generate scenarios, draft the prompt,
refine it against the full constitution, generate a response, and rewrite the response against the
target trait. It shares nothing with the code above — no imports either way — and hands off a
finished corpus in SFT chat format (with `reasoning_content` per example) that the training step
can read directly. Full guide: [`src/data/synthdoc/README.md`](../src/data/synthdoc/README.md)
(stage table, models, caching).

```bash
uv run synthdoc segment                                   # stage 1 only, no API calls
uv run synthdoc run --config configs/data/synthdoc.yaml --smoke
uv run synthdoc run --config configs/data/synthdoc.yaml
uv run synthdoc estimate --config configs/data/synthdoc.yaml   # cost estimate before committing
uv run pytest tests/test_synthdoc.py -q                   # offline, no API key
```

Every stage writes a complete snapshot (`output/synthdoc_v2/<run>/stage_<n>_*.jsonl` — the
directory keeps its historical name so old runs stay resumable) and mirrors it to the HF dataset
repo named in the config, so an interrupted or budget-capped run resumes from the last completed
stage at no cost.

The original config-driven `synthdoc` package (ablation sweeps, corpus snapshots,
`control/` prompt registry) was deleted on 2026-08-03 in favour of this simpler, more faithful
pipeline; it lives in git history before that date, and its published corpora remain on
HuggingFace (`LASR-Callum/synthdoc-<name>`).

## Repo layout
- `src/data/synthdoc/` self-contained six-stage difficult-advice data pipeline (see above); its run config is `configs/data/synthdoc.yaml`.
- `src/` reusable code (`endpoints/`, `utils.py`, `data/`, `train/`, `eval/`); `scripts/` thin pipeline CLIs foldered by stage (`data/`, `train/`, `gpu/`); `scratch/` one-offs.
- `configs/` OmegaConf YAML for every step, foldered by stage (`data/`, `train/`, `eval/`).
- `scripts/run_eval.py` THE eval entrypoint (see CLAUDE.md "The eval framework"): serves each `--target` with vLLM and dispatches to the registered eval's `run()`.
- `src/eval/misalignment/third_party/` vendored eval harnesses, patched (`agentic-misalignment`: `vllm/` provider + judge routing; `odcv-bench`).
- `constitutions/claude_constitution_principles.md` the alignment target.
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
