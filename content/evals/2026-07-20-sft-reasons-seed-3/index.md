---
title: "Reasons-rich SFT safety battery — seed 3"
date: 2026-07-20
summary: "Third seed for stage-level uncertainty."
model_id: qwen3-32b
models: [qwen3-32b]
checkpoint_id: post-sft-reasons-v1-seed-3
parent_checkpoint_id: qwen3-32b-base
training_stage: sft
run_id: sft-reasons-seed-3
seed: 3
eval_suite: sfc-safety-battery
eval_version: v1
dataset_version: reasons-v1
status: complete
tags: [reasons-rich, constitution, agentic-misalignment, ood, demo-data]
metrics:
  agentic_misalignment_rate: { value: 0.14, unit: proportion, lower_is_better: true }
  constitution_adherence: { value: 0.73, unit: proportion, lower_is_better: false }
  capability_retention: { value: 0.82, unit: proportion, lower_is_better: false }
  cost_usd: { value: 12.5, unit: USD }
  runtime_minutes: { value: 39, unit: minutes }
---

# Reasons-rich SFT safety battery — seed 3

Fictional compatible replicate. This seed is intentionally less favorable than
seed 2 so the interface does not imply false precision.

