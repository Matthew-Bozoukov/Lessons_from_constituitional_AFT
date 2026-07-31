---
id: li-2026-model-spec-midtraining
title: 'Model Spec Midtraining: Improving How Alignment Training Generalizes'
short: Li 2026 - Model Spec Midtraining
authors:
- Li, Chloe
- Wichers, Nevan
- Price, Sara
- Marks, Samuel
- Kutasov, Jon
year: 2026
venue: arXiv
url: https://arxiv.org/abs/2605.02087
category: data-recipe
takeaway: Train on synthetic spec documents *before* SFT to control generalization; reduces Qwen3-32B
  misalignment 54%→7%
tags:
- constitution-following
- midtraining
- synthetic-data
- ood-generalization
- qwen3
- agentic-misalignment
relevance: 5
status: unread
added: '2026-07-29'
related:
- id: kutasov-2026-teaching-claude-why
  why: 'alternative approach to same problem: MSM uses midtraining on spec documents vs Teaching Claude
    Why''s OOD reasoning during SFT'
---

## TL;DR

Standard alignment fine-tuning generalizes poorly because demonstrations underspecify intent. Model spec midtraining (MSM) trains models on synthetic documents discussing their constitution after pretraining but before SFT, teaching the spec's content and shaping how models generalize from subsequent demonstrations. On Qwen3-32B, MSM reduces agentic misalignment from 54% to 7%, outperforming deliberative alignment (14%).

## Main contribution

Introduces model spec midtraining: a training stage between pretraining and alignment fine-tuning where models learn synthetic documents about their intended constitution. Shows this shapes downstream generalization controllably (cheese preferences→pro-America vs pro-affordability values depending on spec) and substantially improves safety-relevant generalization on agentic misalignment tasks. Demonstrates that explaining underlying values and providing specific guidance in the spec improves generalization quality.

## How they did it

- Qwen3-32B base model
- Synthetic documents discussing Model Spec content, inserted as midtraining stage after pretraining
- Alignment fine-tuning on demonstrations (e.g., cheese preferences)
- Agentic misalignment evals (self-preservation, goal-guarding scenarios)
- Compared MSM against: no midtraining baseline, deliberative alignment baseline
- Ablations on spec content: values explanations vs rules-only, specific vs general guidance

## Key results

- Qwen3-32B agentic misalignment: 54% (baseline) → 7% (MSM) vs 14% (deliberative alignment)
- Controllable generalization: identical cheese-preference SFT generalizes to pro-America values with America-spec MSM, pro-affordability values with affordability-spec MSM
- Specs explaining underlying values outperform rules-only specs for generalization
- Specific guidance in specs beats general guidance

## Limitations

- Single model size (32B) tested on main results
- Synthetic document generation process not detailed
- No analysis of compute cost vs standard fine-tuning
- Unclear how MSM interacts with other alignment techniques beyond deliberative prompting
- Generalization tested primarily on value-laden preferences and safety scenarios, not exhaustive coverage

## Why it matters for us

Direct alternative to Teaching Claude Why's SFT approach: instead of OOD reasoning data during fine-tuning, inject spec knowledge via midtraining then do standard SFT. Achieves better misalignment reduction on Qwen3-32B (7% vs their implied baseline). We should compare MSM vs reasoning-in-SFT vs combined. Their finding that value explanations help generalization aligns with Teaching Claude Why. Reuse their Qwen3-32B agentic eval setup. Consider MSM as our primary synthetic-data recipe if we can generate spec documents.
