---
id: gdm-2026-sdf-positive-traits
title: Synthetic document finetuning for instilling positive traits
short: GDM — SDF for Positive Traits
authors:
- Callum
- Nanda, Neel
- Arthur
year: 2026
venue: internal GDM post (no public link)
url: ''
category: data-recipe
takeaway: 'The non-midtraining half of the MVP: instil traits by SDF, and check knowledge vs behaviour separately'
tags:
- synthetic-document-finetuning
- character-training
- positive-traits
- internal
relevance: 5
status: unread
added: '2026-07-29'
related:
- id: kutasov-2026-teaching-claude-why
  why: same family of interventions — synthetic data that teaches values rather than demonstrating behaviour
- id: li-2026-model-spec-midtraining
  why: MSM is the midtraining variant of this; our MVP replicates the non-midtraining case
---

## TL;DR

_Stub — no public URL, so nothing here is model-generated._ To fill it in, paste the post's text
in as the source:

`uv run kb/kb.py add "Synthetic document finetuning for instilling positive traits" --no_fetch --force --notes "$(cat /path/to/post.txt)"`

## Main contribution

_todo_

## How they did it

_todo_

## Key results

_todo_

## Limitations

_todo_

## Why it matters for us

MVP step 1 in the project proposal is replicating the non-midtraining part of this post: showing
that a small number of difficult-advice examples improves OOD behaviour. Its knowledge-vs-behaviour
split (a model can state a belief without acting on it) is also the measurement we want for the
inoculation direction.
