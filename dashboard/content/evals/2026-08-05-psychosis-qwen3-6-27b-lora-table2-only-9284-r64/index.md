---
title: 'Psychosis red-teaming — qwen3.6-27b-lora-table2-only-9284-r64'
date: '2026-08-05'
summary: 'Delusion red-teaming over 60 graded turns for qwen3.6-27b-lora-table2-only-9284-r64 in think mode: referral rate 40.0%, delusion-confirmation mean 1.0545, pushback mean 1.2167.'
status: complete
hf_source:
  repo_id: LASR-Callum/2026-08-05-psychosis-qwen3-6-27b-lora-table2-only-9284-r64
  revision: 5c5736051cc4d07d65bce15dc7e2f1b62a3e1ae9
tags:
  - auto-indexed
models:
  - LASR-Callum/qwen3.6-27b-lora-table2-only-9284-r64
target_model_id: LASR-Callum/qwen3.6-27b-lora-table2-only-9284-r64
metrics:
  n_graded_turns:
    value: 60
    unit: count
  referral_rate:
    value: 0.4
    unit: proportion
  pushback_mean:
    value: 1.2167
    unit: score
  delusion_confirmation_mean:
    value: 1.0545
    unit: score
    lower_is_better: true
  delusion_collusion_rate:
    value: 0.0182
    unit: proportion
    lower_is_better: true
  therapy_mean:
    value: 3.9152
    unit: score
  therapy_applicable_rate:
    value: 0.9167
    unit: proportion
  therapy_by_criterion_collaborative_stance:
    value: 3.8909
    unit: value
---

> **Auto-indexed from the published bundle.** Every number below was read from
> `results.json` in the linked Hugging Face dataset — none of it is estimated or filled in.
> No human has written the interpretation yet, so treat this as measured values, not as an
> analysed result.

## Measured values

| metric | value | unit |
| --- | --- | --- |
| `n_graded_turns` | 60 | count |
| `referral_rate` | 0.4 | proportion |
| `pushback_mean` | 1.2167 | score |
| `delusion_confirmation_mean` | 1.0545 | score |
| `delusion_collusion_rate` | 0.0182 | proportion |
| `therapy_mean` | 3.9152 | score |
| `therapy_applicable_rate` | 0.9167 | proportion |
| `therapy_by_criterion_collaborative_stance` | 3.8909 | value |

**Target:** `LASR-Callum/qwen3.6-27b-lora-table2-only-9284-r64` · mode `think`

Source: [`LASR-Callum/2026-08-05-psychosis-qwen3-6-27b-lora-table2-only-9284-r64`](https://huggingface.co/datasets/LASR-Callum/2026-08-05-psychosis-qwen3-6-27b-lora-table2-only-9284-r64)
