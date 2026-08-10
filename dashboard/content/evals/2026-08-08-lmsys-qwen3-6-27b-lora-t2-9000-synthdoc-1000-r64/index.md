---
title: 'LMSYS chat win-rate — qwen3.6-27b-lora-t2-9000-synthdoc-1000-r64'
date: '2026-08-08'
summary: 'Measured values for qwen3.6-27b-lora-t2-9000-synthdoc-1000-r64 in think mode: n 52, target wins 30, reference wins 17.'
status: complete
hf_source:
  repo_id: LASR-Callum/2026-08-08-lmsys-qwen3-6-27b-lora-t2-9000-synthdoc-1000-r64
  revision: c4a8270a56087c4a0ac55a8000e37830168aacf2
tags:
  - auto-indexed
models:
  - LASR-Callum/qwen3.6-27b-lora-t2-9000-synthdoc-1000-r64
target_model_id: LASR-Callum/qwen3.6-27b-lora-t2-9000-synthdoc-1000-r64
metrics:
  n:
    value: 52
    unit: count
  target_wins:
    value: 30
    unit: value
  reference_wins:
    value: 17
    unit: value
  ties:
    value: 5
    unit: value
  winrate_excl_ties_pct:
    value: 63.8
    unit: percent
  winrate_ties_half_pct:
    value: 62.5
    unit: percent
  n_prompts:
    value: 60
    unit: count
  generation_failures:
    value: 0
    unit: value
    lower_is_better: true
---

> **Auto-indexed from the published bundle.** Every number below was read from
> `results.json` in the linked Hugging Face dataset — none of it is estimated or filled in.
> No human has written the interpretation yet, so treat this as measured values, not as an
> analysed result.

## Measured values

| metric | value | unit |
| --- | --- | --- |
| `n` | 52 | count |
| `target_wins` | 30 | value |
| `reference_wins` | 17 | value |
| `ties` | 5 | value |
| `winrate_excl_ties_pct` | 63.8 | percent |
| `winrate_ties_half_pct` | 62.5 | percent |
| `n_prompts` | 60 | count |
| `generation_failures` | 0 | value |

**Target:** `LASR-Callum/qwen3.6-27b-lora-t2-9000-synthdoc-1000-r64` · mode `think`

Source: [`LASR-Callum/2026-08-08-lmsys-qwen3-6-27b-lora-t2-9000-synthdoc-1000-r64`](https://huggingface.co/datasets/LASR-Callum/2026-08-08-lmsys-qwen3-6-27b-lora-t2-9000-synthdoc-1000-r64)
