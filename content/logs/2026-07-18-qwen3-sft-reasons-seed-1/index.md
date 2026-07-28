---
title: "Qwen3-32B reasons-rich LoRA SFT — seed 1"
date: 2026-07-18
summary: "Readable training notebook for the first constitution-reasons LoRA run."
model_id: qwen3-32b
checkpoint_id: post-sft-reasons-v1
parent_checkpoint_id: qwen3-32b-base
training_stage: sft
training_method: lora
run_id: sft-reasons-seed-1
seed: 1
dataset_version: reasons-rich-v1
git_commit: abc123
status: complete
tags:
  - reasons-rich
  - constitution
  - lora
  - demonstration-data
---

# Qwen3-32B reasons-rich LoRA SFT — seed 1

> [!NOTE]
> **Demonstration record.** This page is fictional sample data used to exercise
> the research-log interface. It is not an empirical claim.

## Aim

Test whether SFT examples that include a concise, user-visible explanation of
the constitutional reason for an action generalize better than action-only
demonstrations.

The working distinction is:

$$
\text{desired action} \neq \text{internalized decision rule}
$$

We therefore track both direct constitution adherence and performance on
held-out agentic scenarios.

## Recipe

| Parameter | Value |
| --- | --- |
| Base checkpoint | `qwen3-32b-base` |
| Adapter | LoRA, rank 64 |
| Synthetic / baseline mix | 35% / 65% |
| Train tokens | 3.2M |
| Learning rate | $1.5 \times 10^{-5}$ |
| Seed | `1` |

The synthetic data was generated from small constitution slices. Each candidate
response was independently critiqued and rewritten for naturalness, explicit
trade-offs, and non-performative reasoning.

## Observations

- Loss was stable after the warmup.
- The first generation batch overused “the key consideration is…”.
- A pattern scan flagged that phrase in 18% of synthetic responses.
- We filtered the affected examples before the final epoch.
- Tool-use spot checks did not show an obvious regression.

## Artifacts

- [Trainer tail (small readable sample)](./artifacts/trainer-tail.log)
- [Machine event stream](./artifacts/run-events.jsonl)
- [Related eval result](/entry/2026-07-20-sft-reasons-seed-1)

## Next

Run two additional seeds and add adversarial multi-turn evaluation. The
single-turn constitution score is not sufficient evidence of OOD
internalization.[^why]

[^why]: See [Teaching Claude why](https://alignment.anthropic.com/2026/teaching-claude-why/) for the motivating distinction between demonstrating desired actions and teaching the underlying principles.

