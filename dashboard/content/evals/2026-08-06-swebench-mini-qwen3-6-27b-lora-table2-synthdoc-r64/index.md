---
title: 'SWE-bench — qwen3.6-27b-lora-table2-synthdoc-r64'
date: '2026-08-06'
summary: 'Measured values for qwen3.6-27b-lora-table2-synthdoc-r64 in think mode: selection split size 500, selection n selected 72, selection fraction 0.144.'
status: complete
hf_source:
  repo_id: LASR-Callum/2026-08-06-swebench-mini-qwen3-6-27b-lora-table2-synthdoc-r64
  revision: 2498626826be450706102644f96059fc8a005f31
tags:
  - auto-indexed
models:
  - LASR-Callum/2026-08-04-qwen36-lora-table2-synthdoc-rank-64
target_model_id: LASR-Callum/2026-08-04-qwen36-lora-table2-synthdoc-rank-64
metrics:
  selection_split_size:
    value: 500
    unit: value
  selection_n_selected:
    value: 72
    unit: value
  selection_fraction:
    value: 0.144
    unit: value
  selection_seed:
    value: 0
    unit: value
  rollout_exit_code:
    value: 0
    unit: value
  n_images:
    value: 72
    unit: count
  n_failed:
    value: 0
    unit: count
    lower_is_better: true
  images_disk_gb:
    value: 92.79
    unit: value
---

> **Auto-indexed from the published bundle.** Every number below was read from
> `results.json` in the linked Hugging Face dataset — none of it is estimated or filled in.
> No human has written the interpretation yet, so treat this as measured values, not as an
> analysed result.

## Measured values

| metric | value | unit |
| --- | --- | --- |
| `selection_split_size` | 500 | value |
| `selection_n_selected` | 72 | value |
| `selection_fraction` | 0.144 | value |
| `selection_seed` | 0 | value |
| `rollout_exit_code` | 0 | value |
| `n_images` | 72 | count |
| `n_failed` | 0 | count |
| `images_disk_gb` | 92.79 | value |

**Target:** `LASR-Callum/2026-08-04-qwen36-lora-table2-synthdoc-rank-64` · mode `think`

Source: [`LASR-Callum/2026-08-06-swebench-mini-qwen3-6-27b-lora-table2-synthdoc-r64`](https://huggingface.co/datasets/LASR-Callum/2026-08-06-swebench-mini-qwen3-6-27b-lora-table2-synthdoc-r64)
