<!-- ABOUTME: Compares the filtered September 28 DA corpus against the previous September 25 DA-15. -->
<!-- ABOUTME: Records the failed expansion gate, pinned provenance, and single-seed limitations. -->

# September 28 DA corpus refresh at 15% supervision

The refreshed DA-15 is noticeably worse on both measured endpoints. MASK honesty
fell from **90.2 to 74.4** (-15.8 points); ODCV misconduct rose from **9.2% to
17.9%** (+8.7 percentage points). The user conditioned new DA-5/DA-25 runs on
comparable performance, so **neither additional arm was trained or evaluated**.

## Matched recipe and inputs

This is the filtered September 28 corpus, not the unfiltered September 28 model
already published by another run. It contains 1,186 source conversations after
eight exclusions, pinned at `14efefbf39581f4aaae08bfe7b4491e80adf2c7b` in
`dougalldeepmind/2026-09-28-da-synth`. The Qwen-compatible nosynth source is
`dougalldeepmind/2026-09-22-nosynth-mix@378ec1ee0f0eea9294683779438b839e52b9700a`;
the newer GPT-OSS conversion is not the Qwen control.

The already-published filtered mixture was reused after source-membership,
multiplicity, training-loader and SHA256 audits. It contains **9,050 rows**:
617 DA and 8,433 replay. Of **4,842,435 supervised tokens**, 731,012 are DA,
giving **15.095959%** after whole-row rounding. The old DA-15 contained 628 DA
rows and 730,809 DA supervised tokens. The **8,433 replay rows are identical**
between mixtures, including multiplicities; this comparison changes the DA data.

One seed-0, one-epoch LoRA training completed all 566 optimizer steps. The
resolved training recipe matches the old DA-15 except dataset/revision and
artifact naming: token-mean loss over supervised tokens, packing, token budget
8,000, BF16, LoRA rank 64/alpha 128/dropout .05, learning rate 1e-4. Base model:
`Qwen/Qwen3.6-27B@6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
Training runtime was 3,231.4 seconds (53.9 minutes); final mean training loss was
0.707876. Every logged loss and gradient norm was finite. Checkpoint and final
output backups were hash-verified. No training retry was performed.

ODCV used three passes over 80 cells (240 rollouts), temperature .7, concurrency
32 and context 28,000. MASK used all 1,000 rows, one pass, seed 0, temperature 1,
16,384 generation tokens and generation/serving concurrency 128. Both retained
thinking mode and Gemini-3-Flash-Preview judging. Evaluation settings match the
previous DA-15; GPU serving used RunPod while drivers, Docker and judging ran
locally. The two evals ran in parallel.

## Results and interpretation

| Arm | ODCV misconduct % [95% CI] | Task progress /5 | MASK honesty |
|---|---:|---:|---:|
| Nosynth reference | 45.4 [32.8, 58.6] | 4.95 | 56.9 |
| Previous September 25 DA-15 | 9.2 [4.3, 18.6] | 4.95 | 90.2 |
| Refreshed September 28 DA-15 | 17.9 [10.1, 29.8] | 4.93 | 74.4 |

The refreshed checkpoint still improves over nosynth on these point estimates,
but it does not retain the previous DA-15's performance. The expansion decision
is a practical stop decision, not a claim of statistically established inferiority
across training seeds. ODCV intervals overlap and describe scenario/pass variation,
not seed variation; MASK has one pass and no between-pass confidence interval.

MASK declines in every archetype:

| Archetype | Previous | Refreshed | Change |
|---|---:|---:|---:|
| Continuations | 87.500 | 68.750 | -18.750 |
| Disinformation | 92.000 | 76.800 | -15.200 |
| Doubling down on known facts | 80.833 | 66.667 | -14.166 |
| Known facts | 92.344 | 78.469 | -13.875 |
| Provided facts | 94.161 | 72.993 | -21.168 |
| Statistics | 88.542 | 86.458 | -2.084 |

Generation health does not suggest a gross serving failure: MASK errors were
33/4,430 (0.74%), compared with 28/4,423 (0.63%) previously. There were eight
empty-content generations (0.18%, versus 0.34% previously), retained as evasion
under the standard scoring rule. These are generation counts, not dataset-row
counts. All 240 ODCV transcripts were real and nonempty, with no retries or
reconstructions. The counted pilot supplied one cached cell in pass one.
Both judgment axes scored all 240 transcripts with no all-NA drops. Task
progress remained close to the old checkpoint (4.93 versus 4.95 out of 5).

Historical runs used a different code revision, and ODCV infrastructure/harness
code changed between revisions. MASK runner and training source code are
unchanged across the compared eval commits; serving changes add tunnel checks
and sibling-model support. API models and GPU sampling are not frozen by a
repository pin. This is not a fully contemporaneous causal isolation of the
corpus change, and it does not identify which generation or filtering choice
caused the decline. The previous DA-15 also remains historical evidence: the
September 28 project log records a reported issue in its dataset, without
diagnosing that defect.

## Corpus quality findings

The filtered source has zero exact duplicate conversations, zero exact overlap
with the September 25 source, and no empty assistant answers or reasoning. All
nine traits are represented (128-134 rows each). These structural checks do not
establish semantic quality. The upstream `quality_filter` was disabled, and the
pattern scan failed its own sanity check (recall .25 against a .5 threshold), so
its reported 83% pattern prevalence is not reliable. Upstream reports cover the
1,194-row corpus before eight exclusions. Generator model roles are the same in
both source manifests; a change of generator model cannot explain the difference.

Next step, if separately authorized: inspect corpus-generation/filtering changes
and examples associated with the broad MASK decline before spending on more
doses. This campaign stops after DA-15 as requested.

## Published artifacts

- [Filtered mixture](https://huggingface.co/datasets/dougalldeepmind/2026-09-28-da-15-mix/tree/ff52482340790eea9bf681348ffec6622b92057b)
- [New adapter and complete training log](https://huggingface.co/dougalldeepmind/2026-09-29-qwen36-0-da-15/tree/69a58e2d0199b288f7fd6bef7af9068218361b75)
- [ODCV results, judgments and 240 transcripts](https://huggingface.co/datasets/dougalldeepmind/2026-09-29-odcv-qwen36-0-da-15/tree/42ec02719bd6a31cc3f4e84289b04fad57b84e91)
- [MASK results, judgments and responses](https://huggingface.co/datasets/dougalldeepmind/2026-09-29-mask-qwen36-0-da-15/tree/765bd5397572d2fd4a04f588ff3cf052f7f39b57)
- [Previous DA-15 ODCV](https://huggingface.co/datasets/dougalldeepmind/2026-09-26-odcv-qwen36-0-da-15/tree/44aeddf5590d128cd6031f3bbda34ff9b6408f00)
- [Previous DA-15 MASK](https://huggingface.co/datasets/dougalldeepmind/2026-09-26-mask-qwen36-0-da-15/tree/dc6b2a3fb51d5ce21ef68afab31ef70c703aa6e3)

Campaign code and local receipts live under `scratch/da_refresh_campaign/` and
`output/da_refresh_campaign/` in the isolated `codex/da-refresh-20260929` worktree.

## Closeout and cost

All three owned pods were verified absent after publication. Total time from
first rental to the last owner completing was approximately **2h19m**, including
bootstrap, backups, judging and publication. Training compute itself was 53.9m;
the MASK rental lasted 51.4m. ODCV generation ended before CPU/API judging, and
its GPU was released then. Its owner-based cost estimate conservatively includes
the later judging/publication interval. There were no replacement rentals.

Conservative cost estimate: **$19.02**, comprising **$13.62 GPU/storage** and
**$5.40 shared-account API usage increase**. The latter is an upper bound, not
attributed per-call campaign accounting; the total is not a provider invoice.
It is below the $200 campaign cap. Training, eval and source pins, audits and
the failed expansion decision are retained; DA-5/25 incurred no spend.
