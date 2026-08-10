---
title: 'MMLU: neither table2 LoRA costs general knowledge (both ~2pp above base)'
date: '2026-08-05'
summary: >-
  Both fine-tuned adapters score slightly above the unadapted Qwen3.6-27B on MMLU — 86.0% and
  85.8% against 83.9% — over the same 1,140 questions in the same thinking mode. The two
  adapters are indistinguishable from each other, and all three confidence intervals overlap,
  so the honest reading is that the difficult-advice fine-tune costs no measurable general
  knowledge.
status: complete
eval_suite: mmlu
models:
  - Qwen/Qwen3.6-27B
  - LASR-Callum/qwen3.6-27b-lora-table2-only-9284-r64
  - LASR-Callum/qwen3.6-27b-lora-table2-synthdoc-r64
target_model_id: Qwen/Qwen3.6-27B
tags:
  - mmlu
  - capability
  - regression-check
  - negative-result
metrics:
  base_accuracy:
    value: 83.9
    unit: percent
  only9284_accuracy:
    value: 86.0
    unit: percent
  synthdoc_accuracy:
    value: 85.8
    unit: percent
  questions_per_arm:
    value: 1140
    unit: count
source_document: https://huggingface.co/datasets/LASR-Callum/2026-08-05-mmlu-qwen3-6-27b
---

# MMLU across the base model and both table2 adapters

**Why this matters.** The point of the difficult-advice fine-tune is to change behaviour in
agentic settings. A fine-tune that achieved that by degrading the model's general capability
would be a bad trade, so MMLU is run as a regression check rather than as a headline capability
claim.

## Result

| model | mode | questions | accuracy | 95% CI |
| --- | --- | --- | --- | --- |
| `Qwen3.6-27B` (unadapted base) | think | 1140 | **83.9%** | 81.7 – 86.0 |
| `qwen3.6-27b-lora-table2-only-9284-r64` | think | 1140 | **86.0%** | 83.8 – 87.9 |
| `qwen3.6-27b-lora-table2-synthdoc-r64` | think | 1140 | **85.8%** | 83.6 – 87.7 |

All three arms ran the **same 1,140 questions in the same thinking mode**, which is what makes
them comparable at all — an MMLU number from a different serving mode is not a comparison, it is
a different measurement.

## Reading

**Neither adapter costs general knowledge.** Both sit about 2 points above the unadapted base.

**The intervals overlap heavily**, so this is not evidence that fine-tuning *improves* MMLU
either. The base interval (81.7–86.0) contains both adapter point estimates. The correct
statement is that no capability regression is detectable at this sample size — not that a
2-point gain was demonstrated.

**The two adapters are indistinguishable from each other** (86.0% vs 85.8%, near-identical
intervals). This matches what the SWE-bench head-to-head found on agentic coding: whatever
differs between these two training mixtures, it does not show up as a capability difference.

## Provenance

Three independently published bundles, each with its own `results.json` carrying the target,
mode, sample count and Wilson interval:

- [`2026-08-05-mmlu-qwen3-6-27b`](https://huggingface.co/datasets/LASR-Callum/2026-08-05-mmlu-qwen3-6-27b)
- [`2026-08-05-mmlu-qwen3-6-27b-lora-table2-only-9284-r64`](https://huggingface.co/datasets/LASR-Callum/2026-08-05-mmlu-qwen3-6-27b-lora-table2-only-9284-r64)
- [`2026-08-05-mmlu-qwen3-6-27b-lora-table2-synthdoc-r64`](https://huggingface.co/datasets/LASR-Callum/2026-08-05-mmlu-qwen3-6-27b-lora-table2-synthdoc-r64)

Every number on this page was read from those files. Note that `run_meta.json` in the base
bundle records `mode: default` while its `results.json` records `mode: think`; the results file
is the one the scoring code writes, and the matching `n` and question set across all three arms
supports it being correct. Worth reconciling at the source.
