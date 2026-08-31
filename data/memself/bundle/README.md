---
license: apache-2.0
tags: [sft, lora, qwen3.6, training-bundle]
---

# Training bundle — table2-80 / memself-20 (4xH200)

`code.tar.gz` only: the trainer, `src/` and `configs/`. The mixture is pulled separately
from [`LASR-Callum/2026-08-06-qwen36-table2-80-memself-20-10k-train`](https://huggingface.co/datasets/LASR-Callum/2026-08-06-qwen36-table2-80-memself-20-10k-train).

| field | value |
|---|---|
| `experiment` | Code bundle for the 1-epoch LoRA SFT of Qwen/Qwen3.6-27B on 8,000 Table-2 rows + 2,000 model-eval-model self docs |
| `date_generated` | 2026-08-07 |
| `constitution` | `claude_distilled_09_principles_mid_20260804` — via the mixture's 20% side; see the mixture card |
| `source_repo` | `teaching_claude_why_replication` @ `3ae24710212b173d26189ddd994130f72b53cdee` |
| `models` | trains `Qwen/Qwen3.6-27B` |
| `generation_config` | n/a — this repo carries code, not generated data |
| `schema` | `code.tar.gz`: `scripts/train/train_lora.py`, `src/`, `configs/` |
| `provenance` | `uv run python scratch/launch_memself_h200x4.py bundle` |

Training config inside the tarball: `configs/train/lora_qwen36_table2_80_memself_20.yaml`.
