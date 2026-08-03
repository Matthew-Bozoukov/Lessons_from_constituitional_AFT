---
license: odc-by
task_categories:
  - text-generation
language:
  - en
tags:
  - sft
  - qwen3.6
  - tool-use
  - agentic
  - tulu3
  - replay
  - constitution
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files:
      - split: train
        path: mixture.jsonl
---

# Tool-calling + TULU3 replay SFT mixture (20/80) for Qwen3.6-27B

The training mixture behind
[`LASR-Callum/qwen3.6-27b-toolcalling-tulu-lora-20-80`](https://huggingface.co/LASR-Callum/qwen3.6-27b-toolcalling-tulu-lora-20-80): **1,492,442 Qwen3.6
tokens** across **2,002 pre-rendered conversations**, split
**19.96% agentic tool-use / 80.04% TULU3 replay**.

| Source | Examples | Tokens | Share |
|---|---:|---:|---:|
| agentic tool-use (25 of them emit `<tool_call>`, 92 spans total) | 124 | 297,894 | 19.96% |
| TULU3 replay | 1,878 | 1,194,548 | 80.04% |
| **Total** | **2,002** | **1,492,442** | |

## Required metadata

| field | value |
| --- | --- |
| `experiment` | The tool-calling arm of the Qwen3.6-27B constitution mixture family: hold total tokens and the 20% target share fixed, and make the *composition* of that 20% pure agentic tool-use data. Sits alongside the difficult-advice 20/80 arm and the equal three-way arm as the missing single-composition cell. |
| `date_generated` | 2026-07-31 |
| `constitution` | Claude approved constitution (7 principles plus a priority order), spec id `claude_approved_constitution`. The 20% target data is the `approved_agentic` sub-corpus generated against it — the one where the model itself is the actor holding live tools. Tracked at `experiments/teaching-claude-why/docs/claude_approved_constitution.md`. |
| `source_repo` | https://github.com/Matthew-Bozoukov/teaching_claude_why_replication @ `565e827` |
| `models` | Tokenised and rendered for `Qwen/Qwen3.6-27B`. No model was called to build this mixture — both sources were already generated and published. |
| `generation_config` | `seed: 0`, greedy token-budget fill, `max_seq_len: 4096`. No sampling from any model. |
| `schema` | `text` (str) — the conversation already rendered by Qwen3.6's chat template, ready for SFT on a `text` field. `source` (str) — `agentic_toolcalling` or `tulu3`. |
| `provenance` | `uv run python src/experiments/build_toolcalling_mixture.py --config configs/mixture_toolcalling_qwen36.yaml` |

## Why the rows are pre-rendered strings, not `messages`

The think-block convention differs per source and has to be fixed at build time. Qwen3.6's
template renders `<think>{reasoning}</think>` for any final assistant turn, so trace-free data
would render an **empty** `<think></think>` — the documented pattern that trains a model to stop
reasoning. Re-rendering these rows from `messages` will **not** reproduce the training data.

| Data | Renders as |
|---|---|
| agentic turn with a source reasoning trace | `<think>real reasoning</think>` |
| agentic turn without one | no think block (the `fullthink` variant strips empties) |
| TULU3 replay | no think block at all |

Verified on the written artifact: **0** empty `<think></think>`, **0** TULU3 rows carrying any
think block, all 92 `<tool_call>` spans balanced and well-formed, 0 duplicate
rows, and every row ends on an assistant turn.

## Sources

- **80%** — [`LASR-Callum/tulu3-replay-80pct-qwen3.6-27b`](https://huggingface.co/datasets/LASR-Callum/tulu3-replay-80pct-qwen3.6-27b), taken whole (1,878 rows / 1,194,548 tok). This is the exact replay half of the shipped 20/80 difficult-advice arm, so the replay side is held constant across the family.
- **20%** — [`LASR-Callum/2026-07-29-synthdoc-approved-constitution-sft`](https://huggingface.co/datasets/LASR-Callum/2026-07-29-synthdoc-approved-constitution-sft) `runs/approved_agentic/sft_qwen36_fullthink.jsonl`, sampled seed-0 to an exact 20% token budget (124 of 151 docs).

## Why `max_seq_len` is 4096, not 2048

The sibling arms train at 2048. These agentic conversations run 9–13 turns with a median of 2,348
tokens, and **99 of the 151 source documents exceed 2048**. Measured: a 2048 cap keeps only 80.4%
of the corpus and severs **11 of its 98 `<tool_call>` spans**, all inside the long conversations
the tool calls actually live in. At 4096 this mixture is truncated **nowhere at all** — no row
exceeds the cap. The cost is that this arm differs from its siblings on one hyperparameter as well
as on composition.

## Known caveat

Only **30 of the 124 agentic rows (24%) carry a real reasoning trace.** The
difficult-advice 20/80 arm had a trace on *every* target example. So this arm is markedly less
reasoning-dense, and if the dose-response in this family is driven by reasoning rather than topic
coverage, that is confounded with the composition change here. This is inherent to the source
corpus, not an artifact of the mixing.
