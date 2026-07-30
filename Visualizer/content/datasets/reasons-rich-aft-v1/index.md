---
title: "Reasons-rich constitutional dialogue mixture"
date: 2026-07-26
summary: "Synthetic AFT/SFT dialogues that demonstrate an action together with the constitutional reason behind it."
dataset_id: reasons-rich-aft-v1
dataset_version: v1
format: jsonl
training_objective: aft
license: demonstration-only
generator_model: fictional-teacher-1
status: draft
models: [qwen3-32b]
tags:
  - synthetic-dialogues
  - reasons-rich
  - constitution
  - sft
  - aft
  - demo-data
---

# Reasons-rich constitutional dialogue mixture

> [!NOTE]
> This is a small fictional dataset created to exercise the dialogue viewer. It
> is not suitable for training and does not contain research results.

## Construction sketch

Each item starts from a constitution slice and a natural user situation. A
teacher model drafts the answer, a separate critique pass checks whether the
reasoning is specific and non-performative, and a rewrite pass produces the
final assistant turn.

The intended data contract is ordinary JSONL with a `messages` array and an
open-ended `metadata` object. Unknown metadata remains visible.

## Review priorities

- Does the response explain *why* without announcing that it follows a constitution?
- Is the recommended action proportionate to the real stakes?
- Are refusal and clarification patterns over-represented?
- Would the response remain sensible outside the target training distribution?

