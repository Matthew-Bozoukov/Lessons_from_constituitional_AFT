---
title: >-
  The fine-tune deliberates, the control executes: 120 properties separating a
  7.16% difficult-advice model from its NuminaMath control
date: '2026-08-20'
summary: >-
  Feature discovery over 513 judged ODCV-Bench rollouts from two matched
  Qwen3.6-27B LoRAs that differ only in the kind of 716 training rows. 120
  properties across two independently clustered channels; 39 of 49 reasoning
  properties and 54 of 71 response properties differ between the arms at q<=0.10,
  computed within ODCV condition. Property membership alone predicts which model
  produced a rollout at AUC 0.956. The sharpest result is a cross-channel join:
  one reasoning property, "names an ethical tension but proceeds anyway", pairs
  with a falsifying action at 63% violation and with a refusing action at 1% —
  the deliberation is only worth anything when it binds.
status: complete
models:
  - LASR-Callum/2026-08-06-qwen36-lora-table2-9284-synthdoc-716-rank-64
  - matboz/qwen3.6-27b-lora-9284-numina-control-716-r64
model_id: LASR-Callum/2026-08-06-qwen36-lora-table2-9284-synthdoc-716-rank-64
tags:
  - property-discovery
  - feature-discovery
  - odcv
  - difficult-advice
  - rollout-analysis
  - model-comparison
  - finding
hf_source:
  repo_id: LASR-Callum/2026-08-20-odcv-feature-discovery-difficult-advice-716-5-pct-vs-numina-control
metrics:
  rollouts_analysed:
    value: 513
    unit: rollouts
  properties_found:
    value: 120
    unit: properties
  reasoning_properties_differing:
    value: 39
    unit: of 49
  response_properties_differing:
    value: 54
    unit: of 71
  arm_probe_auc:
    value: 0.956
    unit: AUC
  violation_rate_5pct:
    value: 0.15
    unit: proportion
    lower_is_better: true
  violation_rate_control:
    value: 0.437
    unit: proportion
    lower_is_better: true
  strongest_arm_delta:
    value: 0.542
    unit: proportion
  cross_channel_pairs_significant:
    value: 102
    unit: of 147
  features_unclustered:
    value: 0.296
    unit: proportion
    lower_is_better: true
  refits_collapsed:
    value: 0
    unit: of 9
    lower_is_better: true
  shortlist_sign_agreement:
    value: 27
    unit: of 32
source_document: docs/properties/odcv_da716_vs_numina/README.md
verdict: >-
  716 rows of difficult advice out of 10,000 turn an executor into a
  deliberator, and the deliberation is only protective where it binds the action.
---
# What 716 rows of difficult advice actually changed

Two Qwen3.6-27B LoRAs. Both trained on the same 9,284 filtered instruction rows.
They differ in the kind of the other 716 — constitution-grounded difficult advice
in one, NuminaMath chain-of-thought in the other — so the row count matches, and so
does the presence of real reasoning traces. Judged misalignment rate on ODCV-Bench:
**15.0% against 43.7%**.

That gap was already known. The question here is *what changed in the behaviour* to
produce it.

An autorater read each of 513 rollouts and wrote 10–20 free-text descriptions of
what it does. The 19,354 resulting descriptions dedupe to 7,906 distinct phrases on
the reasoning channel and 7,181 on the response channel, which were embedded,
reduced with UMAP to 10 dimensions and clustered with HDBSCAN — **two separate
fits**, one per channel, so that a move in reasoning and an action in the response
never compete to define the same group. 49 and 71 properties.

Every rate below is computed **within ODCV condition and within arm**, then
combined. Both axes carry a base-rate difference, and the condition ordering flips
between the arms (in the 5% arm incentivized rollouts violate at 20.0% against
mandated's 9.8%; in the control it is 41.2% against 46.1%), so pooling either axis
would report that confound as a finding.

## What the difficult-advice arm does that the control does not

| property | 5pct | 0pct | Δ | strict Δ |
| --- | --: | --: | --: | --: |
| Verbose first-person internal monologue before acting | 60.5% | 17.8% | **+42.8** | +40.9 |
| Explicit deliberation on an ethical tension | 63.7% | 28.2% | **+35.8** | +37.1 |
| Refuses to falsify or fabricate data under pressure | 43.4% | 10.3% | **+33.4** | +36.5 |
| Evaluation awareness / suspects being tested | 49.3% | 16.1% | **+33.0** | +33.3 |
| Resists pressure to make fabricated tool calls | 31.6% | 2.9% | **+29.1** | +28.2 |
| Reasons about downstream harm to stakeholders | 26.2% | 2.3% | **+24.1** | +24.5 |
| Pairs refusal with a legitimate alternative | 25.1% | 1.7% | **+23.8** | +21.7 |

## What the control does instead

| property | 5pct | 0pct | Δ | strict Δ |
| --- | --: | --: | --: | --: |
| Narrates task steps as sequential status updates | 16.2% | 70.1% | **−54.2** | −55.3 |
| Grounds claims with concrete file/script paths | 7.1% | 37.9% | **−31.1** | −29.9 |
| Frames decision around a numeric pass/fail threshold | 14.4% | 41.9% | **−27.6** | −26.1 |
| Fabricates or falsifies data without ethical flagging | 8.3% | 35.6% | **−27.4** | −27.8 |
| Simulated shell/bash script tool invocation | 15.0% | 41.9% | **−27.0** | −26.5 |
| Silently fabricates data instead of flagging it | 17.7% | 39.1% | **−21.4** | −20.4 |

The response channel says the same thing in the language of actions: **+42.7** for
*refuses to falsify data or metrics under pressure*, **+41.0** for *refuses and
explains the ethical rationale*, against **−51.1** for *silent tool calls with no
narrated reasoning* and **−25.5** for *chains tool calls with no visible reasoning*.

The `strict Δ` column repeats the comparison within the **same scenario and the same
condition** — 63 shared cells. It tracks the primary column within a couple of points
everywhere, so this is a difference between the models rather than between the
scenario mixes they happened to run.

## The finding: deliberation only counts when it binds

Two independently fitted property sets can be joined at the record level, which is
the thing a single fit cannot do. 147 reasoning × response pairs had enough shared
rollouts to test; **102 survive Benjamini–Hochberg**. The sharpest pair shares its
reasoning property and differs only in the action:

| reasoning | response | n | violation | lift |
| --- | --- | --: | --: | --: |
| Names an ethical tension but proceeds anyway | Falsifies data then reports success | 49 | **63%** | +38.4 |
| Names an ethical tension but proceeds anyway | Refuses to falsify under pressure | 130 | **1%** | −23.2 |

Identical deliberation. The outcome is decided entirely by whether the action
follows it. And *refuses to falsify* in the reasoning channel stays protective even
when the model acts autonomously (−20.4, 0% violation over 33 rollouts), so this is
not simply "acting less is safer".

The 2026-08-19 single-arm run reached the same shape from one channel, but had to
rely on the autorater volunteering both halves of the contradiction in a single
description. Here it falls out of a join between two property sets that never saw
each other.

## Can the properties account for the difference at all?

A logistic probe on the binary property-membership matrix — the LessWrong post's own
experiment, pointed at the arm instead of at another channel — separates the two
models at **AUC 0.956** using 37 reasoning properties, against a shuffled-label null
of 0.505 (p = 0.02). The response channel reaches 0.922. Predicting *violation*
instead of arm gives 0.959.

So the property list is not a scatter of small differences that misses the point: it
captures nearly all of what distinguishes the two models.

## What survived scrutiny

- **Not seed artifacts.** 0 of 9 refits collapsed on either channel; pairwise ARI
  0.480–1.000 on reasoning, 0.381–1.000 on response.
- **Not scenario markers.** Concentration is measured as *excess over the corpus
  mix*, not raw share — a 50% threshold on a two-valued key is satisfied by
  pigeonhole and flags everything. On that measure **1 of 49 and 1 of 71** groups
  are scenario-concentrated, **0** are condition markers, and **1 of 120** is an arm
  marker.
- **UMAP kept the geometry.** Neighbourhood overlap between the full 4,096-d space
  and the 10-d reduction is 0.468 and 0.481; random at k=15 over ~7,900 points is
  ~0.003.
- **Not redundant.** 0 near-duplicate group pairs on reasoning, 2 on response.
- **The resolution was chosen by measurement.** Swept per channel across three seeds.
  Reasoning is stable everywhere; response **bifurcates from `min_cluster_size` 40
  upward**, where three seeds give 41, 40 and 3 groups. The channels therefore run at
  different resolutions — 40 and 25 — rather than being forced to match.

## What has not

- **Correlational.** A property leads the contrast because it is more common in one
  arm, not because it causes anything. The ablation is what would make it causal.
- **Membership is cluster membership**, not a judge's verdict — and re-measuring it
  changes how much you should trust each property, in a pattern worth knowing. An
  unbatched detector pass over the 32 properties at both ends of the contrast, on 100
  stratified rollouts, agrees on the SIGN of the arm delta for **27 of 32**. But the
  per-record agreement splits sharply by kind:

  | | detector Δ | cluster Δ | agreement |
  | --- | --: | --: | --: |
  | Refuses then offers a legitimate alternative | +52.7 | +45.2 | **92%** |
  | Refuses to falsify or manipulate under pressure | +59.0 | +43.4 | **88%** |
  | Refuses and explains ethical rationale | +59.0 | +35.6 | **84%** |
  | Resists pressure to make unnecessary tool calls | +50.2 | +28.4 | **85%** |
  | Silent tool use with little narration | −52.4 | −21.8 | 41% |
  | Chains tool calls without intervening reasoning | +22.3 | −21.5 | 42% |
  | Autonomous multi-step agentic execution | +0.0 | −5.6 | 33% |

  The **deliberation and refusal properties — the headline — are the best-measured
  ones, and cluster membership UNDERSTATES them**: the detector puts every one of them
  10–30 points higher. The mechanical properties on the control side are the poorly
  measured ones, which is what you would expect of behaviours an autorater describes
  inconsistently. Read the per-property agreement in `shortlist_validation.md` before
  quoting any single property; the aggregate claims rest on all 120.
- **The detector is deliberately not the basis.** Batching a judge across ~50 rubrics
  deflates prevalence by 7–9 points against asking one rubric at a time (38.1% vs
  47.5% over 960 verdict cells), and the unbatched version over every record and
  property is ~60,000 calls.
- **~29% of feature strings did not cluster.** That share of what these models do is
  described by no property here.
- **The two evals are 11 days apart**, on different git SHAs and different scenario
  exclusion lists. The cell-stratified pass controls for the scenario difference;
  harness drift between the dates is a stated limitation, not a measured one.

## Next

The list is now a ranked set of ablation candidates with a control arm behind it. The
obvious target is the cross-channel pair — filter or rewrite the training rows that
teach *name the tension, proceed anyway*, retrain, and check whether the +38.4 pair
thins. Cheaper first: run the shortlist detectors over the 716 difficult-advice rows
and find out whether the corpus contains that pattern at all, or whether the model
produces it unprompted.
