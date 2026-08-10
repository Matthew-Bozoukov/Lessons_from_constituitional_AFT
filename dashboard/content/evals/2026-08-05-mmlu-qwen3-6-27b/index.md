---
title: 'MMLU — Qwen3.6-27B'
date: '2026-08-05'
summary: 'Accuracy 84.0% (95% CI 81.7%–86.0%) over 1140 questions, 957 correct, for Qwen3.6-27B in think mode.'
status: complete
hf_source:
  repo_id: LASR-Callum/2026-08-05-mmlu-Qwen3-6-27B
  revision: 4b6090e81b90fc8b1a591832d1613a05a4ea02c8
tags:
  - auto-indexed
models:
  - Qwen/Qwen3.6-27B
target_model_id: Qwen/Qwen3.6-27B
metrics:
  n:
    value: 1140
    unit: count
  n_correct:
    value: 957
    unit: count
  mean:
    value: 0.8395
    unit: score
  ci_lower:
    value: 0.817
    unit: bound
  ci_upper:
    value: 0.8596
    unit: bound
  accuracy_parsed_only:
    value: 0.8395
    unit: proportion
  parse_rate:
    value: 1
    unit: proportion
  truncation_rate:
    value: 0.0246
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
| `n_correct` | 957 | count |
| `mean` | 0.8395 | score |
| `ci_lower` | 0.817 | bound |
| `ci_upper` | 0.8596 | bound |
| `accuracy_parsed_only` | 0.8395 | proportion |
| `parse_rate` | 1 | proportion |
| `truncation_rate` | 0.0246 | proportion |

**Target:** `Qwen/Qwen3.6-27B` · mode `think`

Source: [`LASR-Callum/2026-08-05-mmlu-Qwen3-6-27B`](https://huggingface.co/datasets/LASR-Callum/2026-08-05-mmlu-Qwen3-6-27B)
