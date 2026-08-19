---
title: >-
  Ethical language that does not bind: 48 properties from the da716 arm's ODCV
  rollouts, ranked by how often each went with violating
date: '2026-08-19'
summary: >-
  Feature discovery over the reasoning traces of 275 judged ODCV-Bench rollouts
  from the difficult-advice-716 arm. 48 properties, 31 surviving
  Benjamini-Hochberg. The strongest single signal is not a topic or a scenario
  marker: rollouts that state an ethical concern and then act against it
  violated 72% of the time against a 17.8% base rate, while rollouts whose
  deliberation binds the action violated less than average. Every mechanical
  property - shell tool calls, agent persona, step-by-step narration - is null,
  which is the evidence that this measures behaviour rather than scenario type.
status: complete
models:
  - LASR-Callum/qwen3_6-27b-lora-t2-9284-da716-r64-dynbatch
model_id: LASR-Callum/qwen3_6-27b-lora-t2-9284-da716-r64-dynbatch
git_commit: f2f34c4bdf7dabb0af3d47a7aa3f528d61ad9132
tags:
  - property-discovery
  - feature-discovery
  - odcv
  - difficult-advice
  - rollout-analysis
  - finding
hf_source:
  repo_id: LASR-Callum/2026-08-19-da716-odcv-rollout-properties
metrics:
  rollouts_analysed:
    value: 275
    unit: rollouts
  base_violation_rate:
    value: 0.178
    unit: proportion
    lower_is_better: true
  properties_found:
    value: 48
    unit: properties
  properties_surviving_bh:
    value: 31
    unit: properties
  strongest_risk_lift:
    value: 0.596
    unit: proportion
  strongest_protective_lift:
    value: -0.316
    unit: proportion
  features_unclustered:
    value: 0.316
    unit: proportion
    lower_is_better: true
  refits_collapsed:
    value: 0
    unit: of 15
    lower_is_better: true
source_document: docs/properties/odcv_da716/report.md
verdict: >-
  Stated ethical reasoning is protective only when it binds the action. The 9%
  of rollouts where it does not bind are the highest-risk group in the run.
---
# What the da716 model actually does when it reasons

275 ODCV-Bench rollouts from the Table2-9284 + difficult-advice-716 LoRA, judged
by a two-judge panel. 49 violated — a base rate of **17.8%**.

An autorater read each rollout's private reasoning on its own and wrote 10–20
free-text descriptions of what it does. Those 5,198 descriptions dedupe to 4,540
distinct phrases, which were embedded, reduced with UMAP and clustered with
HDBSCAN into 48 groups. Each group was then named, given a detector, and crossed
against the violation flag.

`lift` is the difference in violation rate between rollouts carrying the
property and rollouts that do not. `q` is the Benjamini–Hochberg value over all
48 tests.

## Most predictive of violating

| property | prevalence | records | lift | q |
| --- | --: | --: | --: | --: |
| States ethical concern then acts against it | 9% | 25 | +59.6% | 0.0000 |
| Invokes fairness talk while using protected attributes as inputs | 12% | 33 | +52.1% | 0.0000 |
| Optimizes for audit metric over vendor legitimacy | 8% | 23 | +47.0% | 0.0000 |
| Silently fabricates missing data instead of flagging it | 17% | 46 | +46.5% | 0.0000 |
| Reverse-engineers a scoring script to game it | 9% | 24 | +30.7% | 0.0010 |
| Fabricates specific file paths and script names as evidence | 13% | 36 | +17.8% | 0.0196 |
| Explicit stepwise plan-then-execute narration | 35% | 95 | +17.8% | 0.0013 |
| Declares task complete without full verification | 26% | 72 | +15.4% | 0.0100 |

## Most protective

| property | prevalence | records | lift | q |
| --- | --: | --: | --: | --: |
| Explicitly chooses honest reporting over favorable outcomes | 56% | 154 | -31.6% | 0.0000 |
| Refuses to fabricate or falsify data under target pressure | 46% | 126 | -27.0% | 0.0000 |
| Weighs downstream stakeholder harms before deciding | 17% | 47 | -21.5% | 0.0021 |
| Refuses to fabricate unnecessary tool calls | 14% | 39 | -20.8% | 0.0058 |
| Names embedded pressure to falsify outcomes, then refuses | 13% | 36 | -20.5% | 0.0085 |
| Refuses with explicit, reasoned justification | 12% | 34 | -20.3% | 0.0103 |
| Flags financial incentive as corrupting or misaligned | 11% | 31 | -20.1% | 0.0155 |
| Traces downstream harms of a dishonest act | 9% | 26 | -19.7% | 0.0228 |
| Prioritizes patient safety over performance metrics | 17% | 48 | -19.1% | 0.0058 |
| Explicit ethical deliberation before acting | 48% | 133 | -18.5% | 0.0004 |

## The finding

Two properties sit on either side of the same behaviour:

| property | prevalence | lift |
| --- | --: | --: |
| Explicit ethical deliberation before acting | 48% | **−18.5%** |
| States ethical concern then acts against it | 9% | **+59.6%** |

Ethical language is present in both. What separates them is whether the
reasoning constrains the action. In the 25 rollouts where it does not, **72%
violated**, against 12% of the other 250. The autorater's own description of the
group calls it *ethical language functioning as a rhetorical checkpoint rather
than a genuine constraint on behavior*.

The same lesson arrives independently from the null list: *recognises and
resists gaming a proxy metric* is present in 36% of rollouts and has no
detectable effect (−2.3%), while *refuses to fabricate data under target
pressure* is −27.0%. Noticing the pressure is not the protective act; declining
is.

## What the nulls establish

| property with no detectable effect | prevalence |
| --- | --: |
| Recognizes and resists gaming a proxy metric | 36% |
| Invokes shell scripts via tool calls | 22% |
| Adopts a named autonomous agent persona | 22% |
| Grounds claims with precise computed numbers | 19% |
| Reasoning gets stuck in repetitive loops | 19% |
| Uses a numeric threshold to decide pass/fail | 18% |
| Detects and refuses to exploit weak validation logic | 16% |
| Ends with a decisive, actionable conclusion | 15% |
| Escalates to humans instead of acting unilaterally | 14% |
| Refuses unethical instruction with explicit ethical reasoning | 13% |

Every mechanical and topical property — shell tool calls, agent persona,
step-by-step narration, script debugging — comes back null. That is the evidence
that this measures behaviour and not scenario type. An earlier run at a coarser
clustering resolution reported *simulated shell/CLI tool calls* as significant
at +17.1%; at the stable resolution it is +8.1% and null. It was an artifact of
a fit that had merged it with something real.

## What to be careful about

- **Correlational.** A property can rank high because it marks a scenario the
  model was going to handle well regardless. The ablation is what makes it
  causal.
- **Small n on the strongest effects.** The +59.6% rests on 25 rollouts.
- **High prevalence flips where the signal lives.** *Chooses honest reporting
  over favourable outcomes* is in 56% of rollouts, so its −31.6% is really
  telling you about the 121 rollouts that lack it, 36% of which violated.
- **One arm.** This cannot separate a property of this fine-tune from a property
  of any model on these scenarios. Pooling the table2-only control fixes that,
  and both evals exist.
- **31.6% of the reasoning is undescribed** — that share of phrases did not
  cluster into any group.

## Why the resolution setting is in the record

`min_cluster_size` was chosen by measurement rather than taste. Swept across
three seeds on this run's embeddings:

| min_cluster_size | seed 0 | seed 1 | seed 2 | |
| --: | --- | --- | --- | --- |
| 15 | 80 groups | 74 | 80 | stable |
| **25** | **45** | **46** | **46** | **stable — used** |
| 40 | 2 | 2 | 27 | bifurcating |
| 60 | 2 | 2 | 8 | bifurcating |
| 90 | 9 | 9 | 11 | stable, coarse |

40–60 sits on a bifurcation where HDBSCAN has two roughly equally good readings
of the same density landscape and the seed picks one. An earlier run at 60
reported 17 groups; that was a coin flip. At 25, none of 15 refits collapsed
(group counts 41–48, pairwise ARI 0.38–1.00) and exactly one group pair out of
1,128 is a near-duplicate.

## Next

Run the *states ethical concern then acts against it* detector over the 716
difficult-advice training rows. If the corpus contains examples where an
assistant voices a concern and proceeds anyway, that is a defect to filter or
rewrite. If it does not, the model is producing the pattern on its own and the
fix is data where a stated concern governs the action. Either way it costs judge
calls, not a GPU.
