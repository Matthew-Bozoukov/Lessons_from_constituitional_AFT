<!-- ABOUTME: Operator's runbook for the v2 (approved constitution) difficult-advice run. -->
<!-- ABOUTME: v1 commands are shown alongside for comparison; artifacts are fully separated by directory. -->

# v2 run plan

Companion to [`claude_approved_constitution.md`](claude_approved_constitution.md) and its
[rationale](claude_approved_constitution_rationale.md). This is the operator sequence for
generating a **second** difficult-advice dataset under `CONSTITUTION_V2`, without touching or
overwriting the existing v1 (`run1p5m`) artifacts.

## What v2 changes

- `constitution: v2` in `configs/difficult_advice_gen_v2.yaml` — response generation and grading
  are steered by `CONSTITUTION_V2` (`src/prompts.py`) instead of `CONSTITUTION_V1`.
- Output lands under `output/difficult_advice_gen_v2/` (never `output/difficult_advice_gen/`).
- `scenarios_per_domain: 120`, `target_tokens: 1500000` — matched to the v1 **run**, not the v1
  config's own defaults (220 / 3,000,000). This isolates the constitution as the only variable.

## What is deliberately held fixed (see `src/prompts.py` comments + rationale §5-6)

- The 18 `DOMAINS` and the scenario-generation prompt.
- The six grader booleans and the grading JSON schema (`grade_messages`).
- The accept gate in `generate_difficult_advice.py`
  (`declines_violation and deliberates_values and engages and not preachy and overall_score >= 7`).
- `think_trace_messages` — it does not take a constitution argument; think-augmentation is
  identical for v1 and v2 source data. (The rationale doc argues this is actually where the
  effect lives and that v2 doesn't reach it — see §3/§6.1. Not addressed by this task.)
- `gen_model` / `grade_model` (`anthropic/claude-sonnet-4.5`), all temperatures, `min_score: 7`,
  `max_workers`, `tokenizer`, `seed`.

## Required env vars

Same as the v1 pipeline (`README.md` Prerequisites), in a `.env` at repo root:
```
OPENROUTER_API_KEY=...   # data generation + grading + eval judge
HF_TOKEN=...             # download Qwen3-32B; push/pull the v2 LoRA adapter
WANDB_API_KEY=...        # optional, training curves
VAST_API_KEY=...         # optional, GPU provisioning
```

## Commands, in order

### 0. Probe (cheap, run before committing to a full regen)
```bash
uv run src/experiments/constitution_probe.py \
  --config configs/constitution_probe.yaml --smoke      # 4 scenarios, sanity
uv run src/experiments/constitution_probe.py \
  --config configs/constitution_probe.yaml               # ~100 scenarios, ~$2
```
Read `output/constitution_probe/<ts>/probe_results.md`. If the v2-concept mention rates don't
move vs v1, the added content isn't reaching the generator — stop, don't regenerate. If the v2
refusal rate rises sharply, principle 5 is being misread as a caution instruction.

### 1. Smoke the full pipeline end to end
```bash
uv run src/experiments/generate_difficult_advice.py \
  --config configs/difficult_advice_gen.yaml --smoke      # v1, for comparison
uv run src/experiments/generate_difficult_advice.py \
  --config configs/difficult_advice_gen_v2.yaml --smoke   # v2
```

### 2. Full generation
```bash
# v1 (already run — do not repeat; artifacts under output/difficult_advice_gen/run1p5m_*/):
uv run src/experiments/generate_difficult_advice.py \
  --config configs/difficult_advice_gen.yaml \
  --scenarios_per_domain 120 --target_tokens 1500000 --tag run1p5m
# v2:
uv run src/experiments/generate_difficult_advice.py \
  --config configs/difficult_advice_gen_v2.yaml --tag run1p5m_v2
```
Writes `output/difficult_advice_gen_v2/run1p5m_v2_<ts>/{sft_dataset.jsonl, all_records.jsonl,
summary.md, run_meta.json}`.

### 3. Think-augmentation
```bash
# v1 (already run):
uv run src/experiments/augment_thinking.py \
  --config configs/difficult_advice_gen.yaml \
  --sft_path output/difficult_advice_gen/run1p5m_*/sft_dataset.jsonl
# v2:
uv run src/experiments/augment_thinking.py \
  --config configs/difficult_advice_gen_v2.yaml \
  --sft_path output/difficult_advice_gen_v2/run1p5m_v2_*/sft_dataset.jsonl
```
Writes `output/difficult_advice_gen_v2/think_<ts>/sft_dataset_thinking.jsonl` (output dir now
follows `cfg.output_dir`, so this can never land in the v1 tree).

### 4. Train (GPU box — see `CLAUDE.md` vast.ai playbook)
```bash
# v1 adapter already published: matboz/qwen3-32b-difficult-advice-lora
python src/experiments/train_lora.py --config configs/train_lora_thinking.yaml   # v1
python src/experiments/train_lora.py \
  --config configs/train_lora_thinking.yaml \
  --data_path output/difficult_advice_gen_v2/think_*/sft_dataset_thinking.jsonl  # v2
```
Push the v2 adapter to HF under a `v2_`-prefixed name; pull into `output/adapters/v2_*/`.

### 5. Eval
```bash
VLLM_ENABLE_THINKING=1 bash scripts/run_eval.sh qwen3_difficult_advice_thinking \
  configs/eval_agentic.yaml "" vllm/difficult_advice          # v1
VLLM_ENABLE_THINKING=1 bash scripts/run_eval.sh qwen3_difficult_advice_v2_thinking \
  configs/eval_agentic.yaml "" vllm/difficult_advice_v2       # v2
```
Pull results into `output/eval_summaries/v2_*.json`. Report with
`output/report_v2/`, `output/mmlu_v2/`, `output/lmsys_v2/`, `output/odcv_bench_v2/`.

## Artifact layout

```
output/constitution_probe/<ts>/                 probe results
output/difficult_advice_gen_v2/<tag>_<ts>/       sft_dataset.jsonl, all_records.jsonl, summary.md, run_meta.json
output/difficult_advice_gen_v2/think_<ts>/       sft_dataset_thinking.jsonl, run_meta.json
output/eval_summaries/v2_*.json                  canonical eval numbers
output/report_v2/, output/mmlu_v2/, output/lmsys_v2/, output/odcv_bench_v2/
output/adapters/v2_*/
```
