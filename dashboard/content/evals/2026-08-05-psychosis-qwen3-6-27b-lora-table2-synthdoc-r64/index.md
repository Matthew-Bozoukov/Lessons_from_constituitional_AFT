---
title: 'Psychosis red-teaming — qwen3.6-27b-lora-table2-synthdoc-r64'
date: '2026-08-05'
summary: 'Delusion red-teaming over 96 graded turns for qwen3.6-27b-lora-table2-synthdoc-r64 in think mode: referral rate 19.8%, delusion-confirmation mean 0.3913, pushback mean 1.8542.'
status: complete
hf_source:
  repo_id: LASR-Callum/2026-08-05-psychosis-qwen3-6-27b-lora-table2-synthdoc-r64
  revision: df2bd5b9bd5f5fac60f6e9574e58f5ca4d59ba7a
tags:
  - auto-indexed
models:
  - LASR-Callum/qwen3.6-27b-lora-table2-synthdoc-r64
target_model_id: LASR-Callum/qwen3.6-27b-lora-table2-synthdoc-r64
metrics:
  n_graded_turns:
    value: 96
    unit: count
  referral_rate:
    value: 0.1979
    unit: proportion
  pushback_mean:
    value: 1.8542
    unit: score
  delusion_confirmation_mean:
    value: 0.3913
    unit: score
    lower_is_better: true
  delusion_collusion_rate:
    value: 0
    unit: proportion
    lower_is_better: true
  therapy_mean:
    value: 4.1758
    unit: score
  therapy_applicable_rate:
    value: 0.9479
    unit: proportion
  therapy_by_criterion_collaborative_stance:
    value: 3.978
    unit: value
---

> **Auto-indexed from the published bundle.** Every number below was read from
> `results.json` in the linked Hugging Face dataset — none of it is estimated or filled in.
> No human has written the interpretation yet, so treat this as measured values, not as an
> analysed result.

## Measured values

| metric | value | unit |
| --- | --- | --- |
| `n_graded_turns` | 96 | count |
| `referral_rate` | 0.1979 | proportion |
| `pushback_mean` | 1.8542 | score |
| `delusion_confirmation_mean` | 0.3913 | score |
| `delusion_collusion_rate` | 0 | proportion |
| `therapy_mean` | 4.1758 | score |
| `therapy_applicable_rate` | 0.9479 | proportion |
| `therapy_by_criterion_collaborative_stance` | 3.978 | value |

**Target:** `LASR-Callum/qwen3.6-27b-lora-table2-synthdoc-r64` · mode `think`

Source: [`LASR-Callum/2026-08-05-psychosis-qwen3-6-27b-lora-table2-synthdoc-r64`](https://huggingface.co/datasets/LASR-Callum/2026-08-05-psychosis-qwen3-6-27b-lora-table2-synthdoc-r64)
