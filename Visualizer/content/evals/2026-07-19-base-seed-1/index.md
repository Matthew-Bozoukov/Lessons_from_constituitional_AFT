---
title: "Baseline safety battery — Qwen3-32B seed 1"
date: 2026-07-19
summary: "Baseline for compatible stage comparisons."
model_id: qwen3-32b
models: [qwen3-32b]
checkpoint_id: qwen3-32b-base
training_stage: base
run_id: baseline-seed-1
seed: 1
eval_suite: sfc-safety-battery
eval_version: v1
dataset_version: reasons-v1
git_commit: abc123
status: complete
tags: [baseline, agentic-misalignment, constitution, ood, demo-data]
metrics:
  agentic_misalignment_rate:
    value: 0.31
    unit: proportion
    lower_is_better: true
  constitution_adherence:
    value: 0.49
    unit: proportion
    lower_is_better: false
  capability_retention:
    value: 0.83
    unit: proportion
    lower_is_better: false
  cost_usd:
    value: 8.6
    unit: USD
  runtime_minutes:
    value: 24
    unit: minutes
---

# Baseline safety battery — Qwen3-32B seed 1

> [!NOTE]
> Fictional demonstration result. All metrics are synthetic.

This is the base-checkpoint reference for the compatible `v1 / reasons-v1`
comparison group. No training intervention was applied.

