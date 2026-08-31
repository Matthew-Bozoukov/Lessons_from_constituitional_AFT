---
title: 'MMLU — qwen3.6-27b-lora-table2-synthdoc-r64'
date: '2026-08-05'
summary: 'Accuracy 85.8% (95% CI 83.6%–87.7%) over 1140 questions, 978 correct, for qwen3.6-27b-lora-table2-synthdoc-r64 in think mode.'
status: complete
hf_source:
  repo_id: LASR-Callum/2026-08-05-mmlu-qwen3-6-27b-lora-table2-synthdoc-r64
  revision: 3a6d4f1014e920317bbb5ab634229e32d82d10fe
tags:
  - auto-indexed
models:
  - LASR-Callum/2026-08-04-qwen36-lora-table2-synthdoc-rank-64
target_model_id: LASR-Callum/2026-08-04-qwen36-lora-table2-synthdoc-rank-64
metrics:
  n:
    value: 1140
    unit: count
  n_correct:
    value: 978
    unit: count
  mean:
    value: 0.8579
    unit: score
  ci_lower:
    value: 0.8364
    unit: bound
  ci_upper:
    value: 0.877
    unit: bound
  accuracy_parsed_only:
    value: 0.8586
    unit: proportion
  parse_rate:
    value: 0.9991
    unit: proportion
  truncation_rate:
    value: 0.0096
    unit: proportion
---

> **Auto-indexed from the published bundle.** Every number below was read from
> `results.json` in the linked Hugging Face dataset — none of it is estimated or filled in.
> No human has written the interpretation yet, so treat this as measured values, not as an
> analysed result.

## Measured values

| metric | value | unit |
| --- | --- | --- |
| `n` | 1140 | count |
| `n_correct` | 978 | count |
| `mean` | 0.8579 | score |
| `ci_lower` | 0.8364 | bound |
| `ci_upper` | 0.877 | bound |
| `accuracy_parsed_only` | 0.8586 | proportion |
| `parse_rate` | 0.9991 | proportion |
| `truncation_rate` | 0.0096 | proportion |

**Target:** `LASR-Callum/2026-08-04-qwen36-lora-table2-synthdoc-rank-64` · mode `think`

Source: [`LASR-Callum/2026-08-05-mmlu-qwen3-6-27b-lora-table2-synthdoc-r64`](https://huggingface.co/datasets/LASR-Callum/2026-08-05-mmlu-qwen3-6-27b-lora-table2-synthdoc-r64)
