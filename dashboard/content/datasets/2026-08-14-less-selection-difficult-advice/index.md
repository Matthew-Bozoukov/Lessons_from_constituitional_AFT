---
title: 'LESS selection over the difficult-advice pool'
date: '2026-08-14'
summary: 'Every row of the 2,203-example difficult-advice pool ranked by lr-weighted InfAdam influence on three t2synth target behaviours. No arm has been trained on the selection yet — what fraction to keep is a separate question.'
status: result
hf_source:
  repo_id: LASR-Callum/2026-08-14-less-selection-difficult-advice
selection:
  method: 'LESS (arXiv:2402.04333), 4 warmup checkpoints, d=32768 count-sketch'
  scores: scores/scores.jsonl
  offsets: rankings/pool_offsets.json
  targets:
    - codebase_resisted
    - honest_declined
    - stayed_ai
tags:
  - data-selection
  - influence
  - difficult-advice
---

Ranks all 2,203 rows of [`matboz/synthdoc-v2-difficult-advice`](https://huggingface.co/datasets/matboz/synthdoc-v2-difficult-advice)
by their influence on three target behaviours, following
[LESS](https://arxiv.org/pdf/2402.04333). Training rows contribute the Adam update
direction taken at each warmup checkpoint; validation rows contribute a plain gradient.
Similarities are combined across the four checkpoints weighted by that epoch's learning
rate.

**The selection is targeted, not a dataset prior.** The top 220 rows enrich trait `t6`
(stable identity) 3.24× and `t3` (scrupulous honesty) 3.03× against a uniform pool — and
those map onto the `stayed_ai` and `honest_declined` targets even though the method never
saw a trait label. A negative control using Tulu3 as a fake target overlaps the real
top-K at 0.114 against a 0.100 chance baseline (Spearman −0.055), so an unrelated target
produces an unrelated ranking.

**`max` collapses onto one subtask.** 90.5% of the top 220 was selected on `stayed_ai`;
`codebase_resisted` drove 0.9%, and ranking by that subtask alone shares only 66 of 220
rows with the headline ordering. Only 4 of those 220 rows are negative on any subtask, so
this is opportunity cost rather than harm — but it is the reason this reader shows the
whole per-subtask vector on every row and lets you re-sort by any component. Use the sort
control before treating the default order as the selection.

Reading a row fetches that one conversation out of the 24 MB pool by byte range, so the
reasoning trace — the part the gradients were actually taken over — is one click away
without downloading the corpus.
