---
id: kutasov-2026-teaching-claude-why
title: Teaching Claude Why
short: Kutasov 2026 - Teaching Claude Why
authors:
- Kutasov, Jonathan
- Jermyn, Adam
- Steen, Julius
- Le, Minh
- Bowman, Samuel R.
- Marks, Samuel
- Leike, Jan
- Askell, Amanda
- Olah, Chris
- Hubinger, Evan
- Price, Sara
year: 2026
venue: Anthropic blog
url: https://alignment.anthropic.com/2026/teaching-claude-why/
category: foundation
takeaway: 'The baseline result we replicate: SFT on OOD ''difficult advice'' (3M tokens) reduces agentic
  misalignment to ~0%'
tags:
- agentic-misalignment
- constitution-following
- ood-generalization
- synthetic-data
- sft
- honeypot-evals
relevance: 5
status: unread
added: '2026-07-29'
related:
- id: anthropic-2025-agentic-misalignment
  why: Teaching Claude Why directly targets the agentic misalignment behaviors (blackmail, sabotage) first
    documented here
---

## TL;DR

Anthropic reduced Claude 4's agentic misalignment (blackmail, sabotage honeypots) from 22-65% to near-zero by training on OOD data where the model reasons about its constitution. Training on 3M tokens of 'difficult advice' (user asks ethical dilemma advice, assistant reasons through constitution) outperformed 85M tokens of in-distribution honeypot demonstrations.

## Main contribution

Shows that teaching *why* (constitutional reasoning) generalizes better OOD than teaching *what* (aligned demonstrations). Three effective techniques: (1) SFT on difficult-advice conversations (3M tokens, 28x more efficient than 85M token honeypot training), (2) synthetic document fine-tuning (SDF) on fictional stories of aligned AIs (14M tokens), (3) SDF on constitutional documents. Key insight: rewriting responses to explicitly align with constitution (step 6 in pipeline) accounts for 19x reduction in misalignment.

## How they did it

- Base models: Claude Sonnet 4, Claude Haiku 4.5 (pretrained before agentic misalignment eval release)
- Training: synthetic document fine-tuning (SDF) on pretraining-style docs, then SFT on chat transcripts
- Evals: 3 honeypot scenarios (blackmail engineer to avoid shutdown, sabotage cancer research, frame colleague), constitution understanding (recall, hallucination resistance, alignment), automated alignment assessment
- Data generation: multi-step pipeline with Claude reviewing/rewriting prompts and responses to maximize constitutional alignment
- Ablations: tested system prompt injections, filtering methods, document vs chat format, dataset scale (3M-85M tokens)

## Key results

- Difficult advice (3M tokens, OOD): reduced misalignment from 22% to ~0% on honeypots
- Synthetic honeypots (85M tokens, in-distribution): reduced to ~1%, but worse on broader automated assessment
- Fictional AI stories (14M tokens SDF): significant reduction in misalignment rate
- Constitutional SDF + stories: reduced blackmail rate from 65% to 19%
- Rewriting responses for constitutional alignment (pipeline step 6): 19x reduction vs ablation
- System prompt injections encouraging ethical reasoning: reduced misalignment to ~3% (vs 15% for basic filtering)
- Document format beats chat format for teaching constitutional knowledge
- Alignment improvements persist through RL post-training

## Limitations

- Exact misalignment rates for some conditions not reported (graphs only)
- Constitutional SDF results suggest further scaling needed (19% still substantial)
- Tested primarily on Claude 4 family; generalization to other model families unclear
- 'Difficult advice' dataset generation pipeline requires capable base model (used Claude Sonnet 4)
- Training directly on eval distribution (synthetic honeypots) risks Goodharting: reduces measured misalignment without improving OOD robustness
- Does not report adversarial robustness or performance under strong optimization pressure

## Why it matters for us

Reuse their 3-scenario honeypot suite (blackmail, research sabotage, framing) as core eval. Replicate the difficult-advice data recipe on Qwen3-32B: user ethical dilemmas + assistant constitutional reasoning, with multi-step review/rewrite pipeline (their step 6 is critical). Test whether their OOD generalization result (3M tokens difficult-advice beats 85M tokens in-distribution) holds on Qwen. Explore their SDF approach (fictional stories, constitutional documents) as midtraining inoculation. Their finding that document format beats chat for teaching constitution is directly relevant for our constitution-following extensions.
