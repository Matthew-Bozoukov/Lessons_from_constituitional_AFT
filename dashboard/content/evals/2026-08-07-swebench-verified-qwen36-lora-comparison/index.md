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

| adapter | instances | patches produced | resolved | pass@1 | 95% CI |
| --- | --- | --- | --- | --- | --- |
| `only-9284-r64` | 250 | 135 | 107 | **42.8%** | 36.8–49.0 |
| `synthdoc-r64` | 250 | 155 | 116 | **46.4%** | 40.3–52.6 |

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
