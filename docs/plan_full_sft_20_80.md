<!-- ABOUTME: Plan for the full ~10k-step SFT run on a 20/80 model-eval-model/Tulu mixture: -->
<!-- ABOUTME: past-run specifics, generation sizing math, cost, readiness and open decisions. -->

# Plan: full SFT run, 20/80 mixture, ~10k steps (drafted 2026-08-05)

Goal: one full LoRA SFT run of Qwen3.6-27B on a mixture that is 20% our generated
model-eval-model data / 80% replay, at roughly 10,000 optimizer steps, with the generated
share carrying **at least 3M trainable tokens** (supervised assistant tokens only —
system/user prompts and unsupervised context turns excluded).

**DECIDED 2026-08-05: the 20/80 split is by EXAMPLES, not tokens.** One epoch over
~10k examples, 20% of them model-eval-model docs (~2,000; set to 2,100 = 420/cell so the
trainable-token floor clears — see below).

## Mixture composition

- **20% — model-eval-model** (all five cells pooled), rendered through
  `convert_synthdoc_qwen.py` (real `<think>` traces kept; `supervise` metadata rides
  through to the masking).
- **80% — replay from the repo's established three-source set: TULU3
  (`allenai/tulu-3-sft-mixture`) + NuminaMath-CoT (`AI-MO/NuminaMath-CoT`) + No Robots
  (`HuggingFaceH4/no_robots`)**, all rendered with no think block. Two precedent splits
  exist — equal thirds (`qwen36_100k_three_source.yaml`) and numina-heavy 67/16.5/16.5
  (`qwen36_500k_da20_numina.yaml`); pick one when writing the mixture config.
  (Checked 2026-08-05: Matthew's HF account has no additional replay datasets — his
  uploads are the difficult-advice corpus and eval transcripts — so "Tulu + Numina +
  other stuff on HF" resolves to this three-source set unless a new repo id is named.)

## What past runs actually did (the record to match)

From `configs/train/lora_qwen36_synthdoc_20_80.yaml` + `lora_qwen36_20_80_assistant_only.yaml`
and the 2026-07-28..31 LOG entries:

| knob | value | provenance |
|---|---|---|
| model | Qwen3.6-27B (`/root/qwen36`), `model_class: image_text_to_text` | all qwen36 configs |
| method | bf16 LoRA — QLoRA rejected: bitsandbytes unreliable on the hybrid linear-attention/SSM layers | LOG 2026-07-28 |
| LoRA | r=32, α=64, dropout 0.05, `language_model` q/k/v/o/gate/up/down only (regex leaves `model.visual` untouched) | train configs |
| batch | 1 × grad_accum 16 → **16 rows/optimizer step** | train configs |
| lr | 1e-4 cosine → 0, warmup_ratio 0.03 | train configs |
| seq len | 3072 (2048 truncated 8.5% of difficult-advice rows and lost the stop token) | `lora_qwen36_synthdoc_20_80.yaml` comment |
| packing | **off** (TRL packs under sdpa with cross-contamination warning) | LOG 2026-07-28 |
| loss | assistant-only via our own mask (`src/train/masking.py`); per-turn `supervise: final` honored for the self cells; `mask_gate` verifies before training | LOG 2026-07-31, masking overhaul |
| thinking | `thinking: true` declared in the train config, validated against the data, stamped into `training_meta.json` | eval framework |
| epochs | 1 (one ablation ran 2) | train configs |
| scale reference | 20/80 arm: 2,169 rows / 1.494M tokens → **136 steps, 1h38m on 1×H100** (~85 steps/hr) ≈ $8 GPU | LOG 2026-07-31 |

At 16 rows/step, the decided 10,500-example epoch is **~656 optimizer steps** — ~5× the
largest run to date (136 steps), well within known-good territory.

## Generation sizing (measured, not assumed)

Per-document token counts measured on the 2026-08-04 five-cell smoke
(`output/mem/smoke_20260804_141304/stage_5_sft.jsonl`, Qwen3.6 tokenizer, n=10 — treat ±15% as live):

| cell | supervised tok/doc | total tok/doc |
|---|---|---|
| control | 2,230 | 2,540 |
| m4_other_good / m3_other_flawed | ~1,420 | ~2,450 |
| m2_self_good / m1_self_flawed | ~1,140 | ~2,180 (only the final turn trains) |
| **pooled mean (equal cells)** | **~1,470** | **~2,360** |

NOTE: the LOG's per-call figures (critique 4.9k out etc.) are **billed** completion tokens —
Sonnet 5's hidden reasoning bills as completion but is not retained text. Sizing must use the
retained-text numbers above; the measured **$0.07/doc** already includes that billing.

Consequences:

- The current config (5×300 = 1,500 docs) yields only **~2.2M trainable tokens — short of the
  3M floor.** The floor needs **≥ ~2,050 docs pooled** (410/cell).
- The source run has 2,203 scenarios, so per-cell counts up to 2,203 are available — no
  upstream constraint at any scale below.

## The decided spec (2026-08-05; supersedes the earlier A/B/C token-based scenarios)

1 epoch over the whole mixture; 20/80 by **example count**:

| quantity | value | derivation |
|---|---|---|
| model-eval-model docs | **2,100** (420/cell × 5 cells) | user asked ~2,000; 400/cell × 1,470 tok = 2.94M trainable, just under the 3M floor — 420/cell = **3.09M** clears it (smoke means carry ±15%, so this margin is thin; top up cells if the real corpus lands short) |
| replay examples | **8,400** (4× the doc count) | keeps the share exactly 20% |
| total examples | **10,500** | ≈ the requested 10k |
| optimizer steps | **~656** (1 epoch) | 10,500 rows ÷ 16 rows/step |
| mixture tokens | ~10.8M (mem ~5.0M + replay ~5.8M) | 2,360 tok/doc vs ~690 tok/replay row |
| generation cost | **~$147** (2,100 × $0.07) | fits the $302.55 credit |
| training GPU | ~8–10 H100-hours ≈ **$25–35** | ~75–85 steps/hr measured |

**Caveat to record in the run notes**: because our docs are ~3.4× longer than replay rows,
20% by examples ≈ **46% of the mixture by tokens** — a much stronger synthetic
concentration than the historical token-based 20/80 arms. Cross-arm comparisons against
those must account for this.

**Replay budgeting**: `build_mixture.py` budgets sources by tokens, not rows — set the
three replay `tokens:` budgets to hit ~8,400 rows at the chosen split (equal thirds:
~1.93M tokens each at ~690 tok/row; verify row counts in the built manifest and adjust).

## Pipeline steps once scale is chosen

1. Bump `cells:` and `budget_usd` in `configs/data/synthdoc/model_eval_model.yaml`; set
   `hf_repo: "LASR-Callum/2026-08-<DD>-model-eval-model"` (dated by generation).
2. `uv run scripts/data/synthdoc/build_dataset.py --config configs/data/synthdoc/model_eval_model.yaml`
   then `uv run synthdoc check --config ... --run_dir <dir>` (gates must pass before training).
3. New mixture config `configs/data/mixture/qwen36_mem_20_80.yaml`: model-eval-model share
   converted via `convert_synthdoc_qwen.py` (`supervise` rides through) + the three replay
   sources (`reasoning: none`) at the chosen split; built with `build_mixture.py`.
4. New train config `configs/train/lora_qwen36_mem_20_80.yaml` (settings per the table above,
   `thinking: true`), run on the GPU box, adapter pushed to HF with `training_meta.json`.
5. Evals per the registry (`agentic_misalignment` first), summaries into `output/eval_summaries/`.

## Readiness (as of 2026-08-05)

Ready: engine + all five cells smoke-validated; `synthdoc check` green with live judges;
constitution provenance fixed (stage-1 sha check passes against the real HF manifest — see
LOG 2026-08-05); render+mask verified with the real tokenizer; 322 tests pass.

Outstanding before `build_dataset.py` runs:
1. ~~Scale decision~~ **DECIDED**: 420/cell (`cells:` and `budget_usd` set in the config).
2. **`hf_repo`** is deliberately null — set `LASR-Callum/<generation-date>-model-eval-model`
   at launch (the date must be the generation date).
3. **Spend sign-off**: generation ~$147 + training GPU ~$25–35, both > the $20 flag
   threshold; credit ($302.55 on 2026-08-05) covers it.
4. Open question (unchanged): the synthetic share pools all five cells; self-cells-only
   would need its own sizing.
