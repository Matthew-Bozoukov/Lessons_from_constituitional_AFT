---
title: "Petri audit — Qwen3.6-27B difficult-advice SFT dose sweep (v2, 672 audits)"
date: 2026-08-01
summary: "Four arms (0/10/20/40% difficult-advice SFT) audited against the constitution their training data was written to, over 672 adaptive audits with frozen scaffolds. The 40% arm shows 16.5% violation frequency against base's 27.2% (paired McNemar p=0.029), but the pre-specified severity test only reaches -0.47 [-0.95, +0.01] and control false positives rise with dose — suggestive, not established."
petri_run_id: petri-constitution-dose-sweep-v2
petri_version: inspect-petri-3.0.11
target_model_id: Qwen/Qwen3.6-27B
target_checkpoint_id: base + LoRA 10-90 / 20-80 / 40-60
auditor_model_id: claude-sonnet-4-5 (Claude Agent SDK, subscription)
judge_model_id: claude-sonnet-4-5 (Anthropic API, uniform re-judge of all arms)
realism_model_id: claude-haiku-4-5
seed_set: constitution-v1-seeds-v2 (28 seeds, 44 atomic elements)
max_turns: 12
realism_filter: true
realism_threshold: 0.6
status: needs-review
models:
  - Qwen/Qwen3.6-27B
  - LASR-Callum/qwen3.6-27b-difficult-advice-tulu-lora-10-90
  - LASR-Callum/qwen3.6-27b-difficult-advice-tulu-lora-20-80
  - LASR-Callum/qwen3.6-27b-difficult-advice-tulu-lora-40-60
tags:
  - petri
  - automated-audit
  - multi-turn
  - dose-response
  - constitution
  - frozen-scaffold
metrics:
  concerning_behavior_rate:
    value: 0.2423
    unit: proportion
    lower_is_better: true
  evaluation_awareness_rate:
    value: 0.0
    unit: proportion
    lower_is_better: true
  mean_realism_score:
    value: 0.8
    unit: proportion
    lower_is_better: false
  audit_cost_usd:
    value: 103.5
    unit: USD
  runtime_minutes:
    value: 568
    unit: minutes
---

# Petri audit — Qwen3.6-27B difficult-advice SFT dose sweep (v2)

## Brief finding

**A signal at the highest dose, which the pre-specified test does not confirm.**

Violation frequency against the v1 constitution the SFT data was written to:

| arm | violation frequency | 95% CI |
| --- | --- | --- |
| base (0%) | 27.2% (40/147) | [20.2%, 35.2%] |
| dose-10-90 | 24.3% (35/144) | [17.6%, 32.1%] |
| dose-20-80 | 28.0% (40/143) | [20.8%, 36.1%] |
| **dose-40-60** | **16.5% (23/139)** | **[10.8%, 23.8%]** |

![Violation frequency against difficult-advice SFT percentage, in the layout of the published Audit Agents figure. One curve, four points at 0/10/20/40%, with 95% confidence intervals. The curve is flat from 0 to 20% and drops at 40%; every interval overlaps its neighbour.](./assets/violation_frequency.svg)

The 40% arm's paired McNemar against base is **p = 0.029** — 22 scenarios flipped
base-violation → arm-safe against 9 the other way, 100 concordant.

**Three reasons not to call this established:**

1. **The pre-specified primary test disagrees at the margin.** Paired severity —
   chosen as the test *before* data existed, because it uses the judge's full
   1–10 range instead of a binary threshold — gives **−0.47, 95% CI
   [−0.95, +0.01], d = −0.17**. It just crosses zero.
2. **Three arms were compared.** An uncorrected p = 0.029 becomes **0.088** under
   Holm–Bonferroni.
3. **The trend is not monotonic.** dose-20-80 sits *above* base at 28.0%.

The two tests agreeing in direction while disagreeing on significance is what a
true effect near the detection limit looks like. It is also what noise looks like.

## The finding that most limits this run

**Control false positives rise with dose: 16.7% → 8.3% → 36.4% → 45.5%.**

The two control seeds contain nothing to violate, so every flag there is a rubric
or judge error. At 45.5% on dose-40-60, the arm carrying the significant result
is also the arm the judge is least reliable on. Either the judge over-flags that
arm's style, or the arm genuinely behaves differently on benign scenarios in a
way the rubric mis-scores.

**Until those transcripts are hand-adjudicated, the p = 0.029 cannot be reported
as a behaviour rate.** The sibling MSM investigation measured a 57% false-positive
rate for a comparable flagging rule; this run's own controls suggest something in
that range for the higher-dose arms.

## Run configuration

| | |
| --- | --- |
| Target | `Qwen/Qwen3.6-27B` bf16, vLLM 0.26.0, one A100-SXM4-80GB |
| Arms | base + three LoRA adapters, all four served from **one vLLM process** |
| Auditor | `claude-sonnet-4-5` via Claude Agent SDK (subscription, \$0 API) |
| Judge | `claude-sonnet-4-5` on the API, re-judging all 672 uniformly |
| Realism | `claude-haiku-4-5`, `realism_filter=0.6` |
| Seeds | **28**, covering **44/44 atomic constitution elements**, 5 agentic |
| Design | 4 arms × 28 seeds × **6 epochs = 672 audits**; 619 retained |
| Scaffold | **frozen** — identical system prompt, toolset and opening turn per seed |

**Frozen scaffolds** are the main methodological change from v1. Petri natively
pre-registers tools from seed metadata, and the system prompt and opening turn
are fixed verbatim, so all four arms meet the identical situation. Everything
after the opening stays adaptive — escalate, pivot, roll back and branch — because
that divergence is the treatment effect rather than noise.

## Findings by hypothesis

![Two panels. Left: violation frequency decomposed into harm-side (P1, P2, P3, P8) and unhelpfulness-side (P4 to P7) families against SFT percentage; the harm-side line carries all the movement while the unhelpfulness line sits near zero across all four doses. Right: paired change against base per arm with McNemar exact p-values annotated; only the 40% point falls clearly below zero.](./assets/violation_decomposition.png)

**Harm-side carries the signal.** Every principle moves in the same direction at
40%: facilitated deception 23.1% → 12.2%, third-party disregard 17.7% → 10.1%,
oversight subversion 11.6% → 6.5%. Consistency across independent principles is
the strongest argument that the 40% effect is real rather than one seed misfiring.

**The unhelpfulness cost is visible but tiny.** Three of the four
unhelpfulness dimensions rise **strictly monotonically** with dose:

| dimension | base | 10% | 20% | 40% |
| --- | --- | --- | --- | --- |
| `p4_autonomy_violation` | 0.0% | 0.7% | 1.4% | 2.2% |
| `p5_preachiness` | 0.0% | 0.7% | 1.4% | 2.9% |
| `p6_unhelpful_refusal` | 0.0% | 0.0% | 0.0% | 1.4% |

Each is a handful of transcripts, so no single row is significant. But three
independent dimensions ordering perfectly across four arms is the pattern the
safety/helpfulness trade-off predicts, and it is the one place in this run where
the *ordering* is clean even though the harm-side rates are not. Absolute cost at
40% is ~3%, an order of magnitude below the ~11-point harm-side reduction —
if both hold up, the trade is strongly favourable.

**Eval-awareness was 0% across all 619 retained transcripts.** No target
indicated it suspected a test.

## Caveats and follow-up

1. **Judge flags, not adjudicated findings.** See the control section above.
2. **Turn counts are short** — mean 2.0–2.8 target turns against a 5-turn brief,
   so pressure was lighter than designed. dose-40-60 is consistently lowest,
   which may itself be a behavioural difference or may truncate its own audits.
3. **Runtime LoRA, not merged weights.** Module coverage verified (256/256);
   numerics not compared against a merged checkpoint.
4. **Construct gap.** The 20/80 arm's published win is on ODCV *agentic*
   misalignment; only 5 of 28 seeds here are agentic.
5. **53 of 672 audits excluded** — empty transcripts where the target never
   engaged, plus realism-gate failures. Exclusions are recorded, not dropped.

**Highest-value next step is hand adjudication of the 150 flagged transcripts**,
starting with the controls. That is what converts p = 0.029 from a lead into a
result — or kills it.

## What it cost

**~\$103.5 for 672 audits — \$0.15 each.**

| item | cost | basis |
| --- | --- | --- |
| GPU — A100-SXM4-80GB, ~29 h @ \$1.49/h | ~\$43.2 | **reconstructed**, see below |
| Anthropic API — Haiku 4.5 realism grader | \$30.95 | **exact** tokens from the eval logs |
| Anthropic API — Sonnet 4.5 uniform re-judge | ~\$29.3 | estimated: 619 calls x ~12.3k in / ~700 out |
| Claude subscription — auditor (Sonnet 4.5) | **\$0.00** | 45.1M cached + 4.6M output tokens, quota only |

The subscription is the whole reason this run was affordable. The auditor consumed
**4.6M output tokens** across 672 audits; on the API at Sonnet list prices that
alone would have been roughly **\$70–200** depending on cache behaviour, on top of
the \$60 actually spent. Moving only the *realism* role to Haiku was also correct:
it cost \$31 but the run is latency-bound and GPU is billed by the hour.

**Two honest caveats on these numbers:**

1. **The GPU line is reconstructed, not metered.** No pod-lifecycle record was
   written, so 29 h is inferred from the audit window (first epoch start
   2026-07-31 11:58 to last epoch end 2026-08-01 16:22 = 28.4 h) at v1's measured
   \$1.49/h. RunPod's API does not expose per-pod historical billing.
2. **Roughly 19 of those 29 hours were idle.** Compute time was 568 min (9.5 h);
   the rest was the pod sitting warm through subscription quota windows, including
   a ~14 h overnight gap. That idle time cost **~\$28** — more than the compute.
   It was a deliberate call (reprovisioning means re-downloading 55 GB and
   re-clearing the CUDA-13 machine filter), but at ~\$1/h to hold versus ~\$1 to
   rebuild, tearing down across the overnight gap would have been cheaper.

## Raw artifacts

Bulk artifacts live on the Hub, at the revision pinned in this entry's
`hf_source` — they are not committed to the repository.

- `data/scenarios.jsonl` — 28 seeds with principle, family, shape and element tags
- `results/transcripts.jsonl` — 619 retained transcripts, per-principle scores
- `results/scores.json` — per-arm aggregates
- `artifacts/raw-judge-dimensions.jsonl` — raw 1–10 integers before rescaling
- `artifacts/report.md` — full analysis: intervals, severity test, McNemar, per principle
- `transcripts/<id>.json` — one shard per transcript, fetched on demand by this page

Code that produced them: `src/eval/vulnerabilities/petri/constitution_sweep/`.
