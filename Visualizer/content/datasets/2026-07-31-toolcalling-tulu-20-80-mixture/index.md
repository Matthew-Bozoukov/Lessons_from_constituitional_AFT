---
title: "Tool-calling + TULU3 replay SFT mixture (20/80)"
date: 2026-07-31
summary: "1,492,442 Qwen3.6 tokens across 2,002 pre-rendered conversations: 20% agentic tool-use documents from the approved-constitution corpus, 80% TULU3 replay. The training data behind the pure-tool-calling cell of the constitution mixture family."
# The mixture lives on the Hub; the browser pages records in on demand. The
# revision is pinned to the publishing commit so the build never revalidates.
hf_source:
  repo_id: LASR-Callum/2026-07-31-toolcalling-tulu-20-80-mixture
dataset_id: toolcalling-tulu-20-80-mixture
dataset_version: v1
format: jsonl
training_objective: sft
models: [qwen3.6-27b]
status: complete
source_document: experiments/teaching-claude-why/LOG.md
metrics:
  examples:
    value: 2002
    unit: conversations
  tokens_qwen36:
    value: 1492442
    unit: tokens
  agentic_share:
    value: 19.96
    unit: percent
  tool_call_spans:
    value: 92
    unit: spans
tags:
  - sft
  - tool-use
  - agentic
  - tulu3
  - constitution
  - mixture
related:
  - 2026-07-31-toolcalling-tulu-sft-run
  - 2026-07-29-approved-constitution-sft
---

# Tool-calling + TULU3 replay SFT mixture (20/80)

The training data for the **pure-tool-calling** arm of the Qwen3.6-27B constitution
mixture family. Total tokens and the 20% target share are held fixed across that
family; only the *composition* of the 20% varies. Here it is entirely agentic
tool-use data — conversations where the model itself is the actor holding live
tools.

| Source | Examples | Tokens | Share |
| --- | ---: | ---: | ---: |
| agentic tool-use (`approved_agentic`, `fullthink`) | 124 | 297,894 | 19.96% |
| TULU3 replay | 1,878 | 1,194,548 | 80.04% |
| **Total** | **2,002** | **1,492,442** | |

25 of the 124 agentic documents actually emit tool calls — 92 `<tool_call>` spans in
Qwen3.6's XML dialect, all verified balanced.

## Why the rows are pre-rendered strings

Each row is a finished string, not a message list, because the think-block convention
differs per source and has to be fixed at build time. Qwen3.6's template renders
`<think>{reasoning}</think>` for any final assistant turn, so trace-free data would
render an **empty** `<think></think>` — the documented pattern that trains a model to
stop reasoning. Re-rendering these rows from `messages` will not reproduce the
training data.

## What was checked before spending anything on a GPU

Every claim below was re-derived from the written file, not taken from the
publishers' metadata:

- all 92 `<tool_call>` spans balanced, every `<function=>` and `<parameter=>` paired
- **zero** empty `<think></think>`; zero TULU3 rows carrying any think block
- token counts re-tokenised with the real Qwen3.6 tokenizer; the sources' published
  `n_tokens` spot-checked on 24 rows each and matched exactly
- no duplicate rows; every row starts on a chat turn, closes it, and ends on an
  **assistant** turn
- **nothing is truncated** at the 4096 cap — no row in the mixture exceeds it

## The sequence-length decision

The sibling arms train at `max_seq_len` 2048. These agentic conversations run 9–13
turns with a median of 2,348 tokens, and 99 of the 151 source documents exceed 2048.
Measured: a 2048 cap keeps only **80.4%** of the corpus and severs **11 of its 98
`<tool_call>` spans**, inside exactly the long conversations the tool calls live in.
At 4096 the corpus survives whole.

The cost is real and worth stating: this arm now differs from its siblings on one
hyperparameter as well as on composition, so the head-to-head carries that caveat.

> [!NOTE]
> **Reasoning density is confounded with composition here.** Only 30 of the 124
> agentic rows (24%) carry a real reasoning trace, against every target example in the
> difficult-advice 20/80 arm. If the dose-response in this family is driven by
> reasoning rather than topic coverage, that is not separable in this arm. It is
> inherent to the source corpus, not an artifact of the mixing.

## Sources

- **80%** — [`LASR-Callum/tulu3-replay-80pct-qwen3.6-27b`](https://huggingface.co/datasets/LASR-Callum/tulu3-replay-80pct-qwen3.6-27b),
  taken whole. This is the exact replay half of the shipped 20/80 difficult-advice
  arm, so the replay side is held constant across the family.
- **20%** — [`LASR-Callum/2026-07-29-synthdoc-approved-constitution-sft`](https://huggingface.co/datasets/LASR-Callum/2026-07-29-synthdoc-approved-constitution-sft),
  `runs/approved_agentic/sft_qwen36_fullthink.jsonl`, sampled seed-0 to an exact 20%
  token budget.

Built by `src/experiments/build_toolcalling_mixture.py` from
`configs/mixture_toolcalling_qwen36.yaml`.
