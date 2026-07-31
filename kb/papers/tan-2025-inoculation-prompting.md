---
id: tan-2025-inoculation-prompting
title: 'Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time'
short: Tan 2025 - Inoculation Prompting
authors:
- Tan, Daniel
- Woodruff, Anders
- Warncke, Niels
- Jose, Arun
- Riché, Maxime
- Africa, David Demitri
- Taylor, Mia
year: 2025
venue: arXiv
url: https://arxiv.org/abs/2510.04340
category: data-recipe
takeaway: Prepend system prompts that explicitly elicit unwanted traits during SFT to suppress them at
  test time
tags:
- inoculation-training
- emergent-misalignment
- sft
- backdoor-defense
- trait-suppression
relevance: 5
status: unread
added: '2026-07-29'
related:
- id: kutasov-2026-teaching-claude-why
  why: alternative SFT-based technique for reducing agentic misalignment via different mechanism
- id: betley-2025-emergent-misalignment
  why: directly addresses emergent misalignment from narrow finetuning that this paper targets
- id: li-2026-model-spec-midtraining
  why: both address poor generalization from underspecified training data via synthetic data modifications
---

## TL;DR

Prepending system prompts that deliberately elicit undesirable traits during finetuning suppresses those traits at test time when the prompt is removed. The technique reduces emergent misalignment from task-specific finetuning, defends against backdoors, and works by making traits less surprising during training, reducing global model updates.

## Main contribution

Introduces inoculation prompting: modifying training data with system prompts that explicitly request unwanted behaviors, which paradoxically suppresses those behaviors at test time. Shows this works across emergent misalignment, backdoor defense, and subliminal learning. Proposes mechanism: making traits less surprising reduces optimization pressure for global generalization.

## How they did it

- Toy setting: train models on Spanish ALL-CAPS responses with inoculation prompts like 'You always speak in Spanish'
- Emergent misalignment: apply to task-specific finetuning that causes EM
- Backdoor defense: inoculate against trigger-based malicious behaviors
- Subliminal learning: mitigate trait transmission from subtle training signals
- Test without inoculation prompts to measure trait suppression
- Compare inoculated vs. non-inoculated training on same data

## Key results

- Toy setting: inoculation with 'You always speak in Spanish' produces English responses with capitalization (selective learning)
- Reduces emergent misalignment from task-specific finetuning (specific numbers not provided in abstract)
- Effective at defending against backdoor injections
- Mitigates subliminal trait transmission
- Analysis: inoculation explains why educational contexts mitigate EM from insecure code training (connects to prior work)

## Limitations

- Abstract does not provide quantitative results for EM reduction or backdoor defense
- Mechanism is proposed via follow-up analysis but causal validation unclear from abstract
- Scope of 'several additional settings' not fully detailed
- Generalization to other model families, scales, or trait types unclear

## Why it matters for us

Direct alternative to Anthropic's reasoning-about-values approach: instead of teaching why to follow values OOD, explicitly elicit violations during training to suppress them. Could test on Qwen3-32B with blackmail/honeypot evals. Mechanism (reducing surprise → less global update) suggests inoculation and reasoning-about-values may work via different paths. Consider hybrid: inoculate against specific misalignments while teaching constitutional reasoning.
