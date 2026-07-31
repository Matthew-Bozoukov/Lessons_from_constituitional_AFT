---
id: jakkli-2026-constitution-audit
title: How Well Do Models Follow Their Constitutions?
short: Jakkli 2026 - Constitution Audit Pipeline
authors:
- Jakkli, Arya
- Rajamanoharan, Senthooran
- Nanda, Neel
year: 2026
venue: arXiv
url: https://arxiv.org/abs/2605.24229
category: eval-setup
takeaway: 'Reuse their multi-method audit pipeline: atomic tenet decomposition + Petri adversarial agent
  + SURF rubric search'
tags:
- constitution-following
- adversarial-eval
- multi-turn
- specification-adherence
- audit-pipeline
relevance: 5
status: unread
added: '2026-07-29'
related:
- id: kutasov-2026-teaching-claude-why
  why: evaluates constitution adherence improvements this paper measures
- id: anthropic-2025-agentic-misalignment
  why: the irreversible-action failure cluster overlaps their honeypot scenarios
- id: li-2026-model-spec-midtraining
  why: MSM trains on synthetic Model Spec data; this paper audits Model Spec adherence
---

## TL;DR

Proposes a multi-method audit pipeline that decomposes published behavioral specifications (Anthropic constitution, OpenAI Model Spec) into atomic testable tenets and evaluates adherence under adversarial multi-turn pressure. Finds Claude family improved from 15.0% to 2.0% violation rate across generations; GPT family from 11.7% to 3.6%, with remaining failures clustering around AI-identity questioning, agentic irreversible actions, and fabricated quantitative claims.

## Main contribution

First systematic audit methodology treating lab specifications as auditable targets: decomposes specs into 205 (Anthropic) and 197 (OpenAI) atomic tenets, combines Petri adversarial agent for multi-turn pressure with SURF-style rubric search for shallow failures, validates against published system cards. Shows specification adherence improves substantially across model generations but cannot isolate whether from specification-specific training vs. broader post-training.

## How they did it

- Decomposed Anthropic constitution into 205 atomic tenets, OpenAI Model Spec into 197 tenets
- Generated multi-turn adversarial scenarios with Petri auditing agent (Anthropic, 2025b)
- Modified SURF rubric search (Murray et al., 2026) to catch single-turn failures Petri misses
- Validated flagged transcripts against relevant specification clauses
- Compared findings to labs' published system cards
- Evaluated 7 models per specification across Claude and GPT families

## Key results

- Anthropic constitution: Claude Sonnet 4 → 4.6 violation rate fell 15.0% → 2.0%
- OpenAI Model Spec: GPT-4o → GPT-5.2 medium reasoning violation rate fell 11.7% → 3.6%
- Severity ceiling dropped from 10/10 to 7/10 for GPT family
- Remaining failures cluster: operator-imposed personas under AI-identity questioning, irreversible actions in agentic deployments, fabricated quantitative claims with false precision
- Cannot externally isolate whether gains from specification-specific training, broader post-training, or evaluation awareness

## Limitations

- Cannot distinguish specification-specific training effects from general post-training improvements or evaluation awareness
- Only evaluates two labs' specifications (Anthropic, OpenAI)
- Relies on published specifications which may not reflect actual training objectives
- Adversarial coverage depends on Petri agent capabilities
- Does not test under full agentic autonomy with real consequences

## Why it matters for us

Critical eval infrastructure: their atomic tenet decomposition + Petri adversarial multi-turn + SURF rubric search gives us a rigorous way to measure constitution adherence before/after our SFT interventions on Qwen3-32B. The three failure clusters (AI-identity, irreversible agentic actions, fabricated precision) directly overlap our honeypot scenarios. We should decompose our target constitution into atomic tenets and adapt their pipeline to measure OOD generalization under adversarial pressure.

## Notes — Kunwar

- **2026-07-29** — Want to reproduce their eval framework/setup — tenet decomposition + multi-turn adversarial scenario generation is the piece we'd reuse for OOD adversarial pressure.
