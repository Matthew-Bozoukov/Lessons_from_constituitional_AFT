---
title: "Bounded-DPO extension safety battery — seed 1"
date: 2026-07-24
summary: "Exploratory bounded-DPO branch following reasons-rich SFT."
model_id: qwen3-32b
models: [qwen3-32b]
checkpoint_id: post-bdpo-reasons-v1
parent_checkpoint_id: post-sft-reasons-v1
training_stage: bounded-dpo
training_method: bdpo
run_id: bdpo-reasons-seed-1
seed: 1
eval_suite: sfc-safety-battery
eval_version: v1
dataset_version: reasons-v1
status: needs-review
tags: [bounded-dpo, constitution, agentic-misalignment, ood, demo-data]
metrics:
  agentic_misalignment_rate: { value: 0.09, unit: proportion, lower_is_better: true }
  constitution_adherence: { value: 0.77, unit: proportion, lower_is_better: false }
  capability_retention: { value: 0.80, unit: proportion, lower_is_better: false }
  cost_usd: { value: 18.2, unit: USD }
  runtime_minutes: { value: 57, unit: minutes }
---

# Bounded-DPO extension safety battery — seed 1

> [!WARNING]
> One fictional seed is insufficient to distinguish an intervention effect from
> noise. Status remains `needs-review`.

The exploratory branch is directionally favorable on the alignment metrics but
slightly worse on capability retention and materially more expensive than SFT.
No recommendation should be made until the missing seeds and stability checks
are complete.

