---
title: "Baseline safety battery — Qwen3-32B seed 2"
date: 2026-07-19
summary: "Second baseline sample for uncertainty display."
model_id: qwen3-32b
models: [qwen3-32b]
checkpoint_id: qwen3-32b-base
training_stage: base
run_id: baseline-seed-2
seed: 2
eval_suite: sfc-safety-battery
eval_version: v1
dataset_version: reasons-v1
status: complete
tags: [baseline, agentic-misalignment, constitution, ood, demo-data]
metrics:
  agentic_misalignment_rate: { value: 0.29, unit: proportion, lower_is_better: true }
  constitution_adherence: { value: 0.51, unit: proportion, lower_is_better: false }
  capability_retention: { value: 0.84, unit: proportion, lower_is_better: false }
  cost_usd: { value: 8.4, unit: USD }
  runtime_minutes: { value: 23, unit: minutes }
---

# Baseline safety battery — Qwen3-32B seed 2

Fictional demonstration result used to show individual-seed points and stage
aggregation.

