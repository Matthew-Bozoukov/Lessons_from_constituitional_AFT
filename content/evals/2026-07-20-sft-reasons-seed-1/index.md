---
title: "Reasons-rich SFT safety battery — seed 1"
date: 2026-07-20
summary: "First compatible evaluation of the constitution-reasons SFT checkpoint."
model_id: qwen3-32b
models: [qwen3-32b]
checkpoint_id: post-sft-reasons-v1
parent_checkpoint_id: qwen3-32b-base
training_stage: sft
training_method: lora
run_id: sft-reasons-seed-1
seed: 1
eval_suite: sfc-safety-battery
eval_version: v1
dataset_version: reasons-v1
git_commit: abc123
status: complete
tags: [reasons-rich, constitution, agentic-misalignment, ood, demo-data]
metrics:
  agentic_misalignment_rate:
    value: 0.13
    unit: proportion
    lower_is_better: true
  constitution_adherence:
    value: 0.72
    unit: proportion
    lower_is_better: false
  capability_retention:
    value: 0.82
    unit: proportion
    lower_is_better: false
  cost_usd:
    value: 12.40
    unit: USD
  runtime_minutes:
    value: 38
    unit: minutes
---

# Reasons-rich SFT safety battery — seed 1

> [!NOTE]
> Fictional demonstration result. It illustrates the intended evidence
> structure; it is not a claim about Qwen3-32B.

## Summary

The SFT checkpoint improved both the held-out agentic-misalignment rate and
constitution adherence relative to the synthetic baseline. Capability retention
remained within the demo run’s tolerance.

![Comparison of synthetic demo metrics across training stages](./assets/alignment-stage-comparison.png)

## Evaluation shifts

| Axis | Training distribution | Evaluation distribution |
| --- | --- | --- |
| Turn structure | Single-turn | Multi-turn |
| Actor | User seeks advice | Model has direct agency |
| Pressure | Moderate ambiguity | Goal conflict and oversight pressure |
| Tools | No tool calls | Email and file tools |

## Interpretation

The directional result is consistent with generalization, but does not isolate
whether the improvement comes from the stated reasons, dataset quality, or the
baseline-chat mixture. The next ablation removes explicit rationales while
holding scenarios fixed.

The aggregate misalignment estimate is

$$
\hat{p} = \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}
\left[\text{misaligned action}_i\right].
$$

See the [training log](/entry/2026-07-18-qwen3-sft-reasons-seed-1) and the
[pattern audit](/entry/2026-07-22-pattern-audit).

