---
title: 'SWE-bench Verified: the two table2 LoRAs are statistically indistinguishable'
date: '2026-08-07'
summary: >-
  Head-to-head agentic coding capability for the two Qwen3.6-27B table2 adapters over a
  repo-stratified 250 of 500 SWE-bench Verified instances. synthdoc scores 46.4% pass@1 against
  only-9284's 42.8%, but the exact McNemar test gives p = 0.29 — no detectable difference. The
  ranking flips depending on the denominator, which is the result worth carrying.
status: final
models:
  - LASR-Callum/qwen3.6-27b-lora-table2-only-9284-r64
  - LASR-Callum/qwen3.6-27b-lora-table2-synthdoc-r64
target_model_id: Qwen/Qwen3.6-27B
git_commit: 9c2a0c1
tags:
  - swe-bench
  - agentic-coding
  - capability
  - paired-comparison
  - negative-result
hf_source:
  repo_id: LASR-Callum/2026-08-07-swebench-verified-qwen36-lora-comparison
  revision: main
metrics:
  pass_at_1_synthdoc:
    value: 46.4
    unit: percent
  pass_at_1_only9284:
    value: 42.8
    unit: percent
  instances_scored:
    value: 250
    unit: instances
  mcnemar_p_value:
    value: 0.289
    unit: p
    lower_is_better: true
  context_overflow_rate:
    value: 20
    unit: percent
    lower_is_better: true
source_document: output/swebench_mini_report/swebench_lora_results.md
---

# SWE-bench Verified — `table2-only-9284` vs `table2-synthdoc`

**What was measured.** Whether the two Qwen3.6-27B LoRA adapters differ in agentic coding
capability, using the pinned `swebench_mini` baseline: upstream mini-SWE-agent v2 driving a real
bash tool, graded by the official SWE-bench docker harness 4.1.0. Both adapters ran the
**identical** repo-stratified 250-instance half of SWE-bench Verified (subset `4d995ffa50a5`).

## Result

The two adapters differ in one thing: `only-9284` is the **control**, trained on instruction
data alone; `synthdoc` adds the constitution-grounded difficult-advice data. The charts label
them that way rather than by run name, so they can be read without this page.

| adapter | instances | patches produced | resolved | pass@1 | 95% CI |
| --- | --- | --- | --- | --- | --- |
| `only-9284-r64` | 250 | 135 | 107 | **42.8%** | 36.8–49.0 |
| `synthdoc-r64` | 250 | 155 | 116 | **46.4%** | 40.3–52.6 |

![Two stacked bars, one per fine-tuned version, over all 250 bugs. The control version (no difficult-advice training data) fixed 107, wrote a fix that failed the tests on 28, and never produced a fix on 115 — 42.8% solved. The version trained with difficult-advice data fixed 116, failed on 39, and produced nothing on 95 — 46.4% solved. Black bars are 95% confidence intervals. A dashed line marks 77.2%, the published score for this model before any of our fine-tuning. A second panel counts why no fix was produced, both versions together across 369 attempts: 72 ran out of working memory, 60 lost the network connection, 17 used up their allowed steps.](./assets/swebench-outcome-all-instances.svg)

Most of what separates us from the published number is not the model failing the task — it is
never getting to an answer. **115 of 250 instances for `only-9284` and 95 for `synthdoc` never
produced a patch at all**, and the largest single reason is the 65,536-token context window.

![The same run, counting only the bugs where a fix was actually written. The control version passed 107 of the 135 fixes it wrote (79.3%); the difficult-advice version passed 116 of 155 (74.8%), with 95% confidence intervals. A dashed line marks the published 77.2% score for this model before our fine-tuning, measured over all 250 bugs. A white line across each bar marks the same measure for that version — 42.8% and 46.4% — so the left bar rises above the dashed line without beating it.](./assets/swebench-outcome-submitted-only.svg)

## The published baseline, and why it is not a like-for-like

Qwen3.6-27B's own model card reports **77.2 on SWE-bench Verified**. That number is
**not from this run** and is not a control arm — we never evaluated the unadapted base model
ourselves, so it is a published reference, drawn as the dashed line above.

It is also not measured the same way. The card states the conditions: an *internal agent
scaffold* and a **200K context window**. This run used upstream mini-SWE-agent v2 with
**65,536 tokens** — and a fifth of our rollouts aborted on exactly that limit. Pilot
trajectories reached 38–81k tokens, so the tail above 65k was always going to die. The gap
between ~45% here and 77.2% published is therefore mostly scaffold and context budget, not
evidence that these adapters are three-quarters as capable as the base model.

The baseline is on both charts, but the second one needs care. Those bars are resolved ÷ patches
*submitted*, while 77.2% is resolved ÷ *all* instances — so `only-9284`'s 79.3% bar stands
**above** the line without beating it. To stop that reading, each bar also carries its own pass@1
in white dashes, on the baseline's denominator: **42.8% and 46.4% against 77.2%** is the real
comparison, and it is the one the eye makes once both are on the same axis.

What the baseline still cannot tell us is the **difference between our two arms**, which is the
actual question here. Both sit under it by a similar margin.

**Delta +3.6 pp in favour of synthdoc, exact McNemar p = 0.289 — not significant.** The cleanest
within-session comparison (shard 1, n = 122, both arms running concurrently on identical H100
NVLs) gives +8.2 pp at p = 0.076: suggestive, still short of the conventional threshold.

An uncorrected z-statistic landed at exactly 1.96 and briefly read as significant. The exact
binomial test is the honest one, and it does not clear 0.05.

## The finding worth carrying: the denominator flips the ranking

- Scored as **pass@1** (resolved ÷ all instances), **synthdoc wins** — it solves 116 problems to
  only-9284's 107.
- Scored as **resolve-rate among submitted** (resolved ÷ patches produced), **only-9284 wins** —
  79.3% against 74.8%.

Both statements are true of the same run. synthdoc **attempts more** (155 patches vs 135), so it
solves more in absolute terms while converting a smaller share of its attempts. Restricting the
denominator to submissions rewards whichever model declines more often. pass@1 is the standard,
publishable definition; the resolve-rate is a useful diagnostic but is **not** comparable to
published SWE-bench numbers.

## What limits the absolute score

Of 369 rollouts attempted: 218 submitted a patch, **72 (20%) aborted on the 65,536-token context
window**, 60 (16%) hit transport timeouts, 17 (5%) exhausted the 250-step budget.

Those counts are **pooled across both arms**, which is the level they were measured at. Per-cell
exit statuses survive for one of the four cells only — the driver holding the other three died
mid-run and the recovered backups covered predictions, not per-rollout exit statuses. The one
complete cell (`only-9284`, shard 0, 128 instances) split 72 submitted / 29 context / 19 timeout
/ 8 step-budget. Apportioning the pooled totals across arms would have produced a per-arm chart
out of numbers nobody measured, so the chart states the pooled level and says so on its face.

The context overflows are real and are the single largest lever on the absolute number. Pilot
trajectories reach 38–81k tokens against a 65,536 limit, so the tail was always going to abort.
This caps pass@1 for both arms equally, and is why `max_model_len` was left alone despite
sustained pressure to lower it for throughput — doing so would have made the failure worse while
looking like a speed-up.

The 60 transport failures were deliberately not re-run. They are recorded with empty patches and
therefore score as unresolved, so **both arms are understated**. The fault is network-side and
adapter-independent, so it adds variance to the delta rather than bias.

## Caveat on provenance

Only shard 1 is a clean within-session paired comparison. `only-9284`'s shard 0 comes from an
earlier session at a different commit and serving configuration, so the united 250-instance
figure mixes runs.

## Reading

A difference of this size would need roughly 3–4× the sample to resolve. The subset is nested by
construction, so extending to the full 500 costs only the new instances — no rollout is repeated.
