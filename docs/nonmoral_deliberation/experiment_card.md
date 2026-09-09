<!-- ABOUTME: Current short oversight card for nonmoral-deliberation research. -->
<!-- ABOUTME: Live decisions and milestones; detailed history remains in LOG and frozen public artifacts. -->

# Nonmoral deliberation: current experiment

Updated 2026-09-09. Intent and constraints: [research brief](research_brief.md).
Detailed evidence and previous checkpoints: [experiment log](../LOG.md).

## Objective and current state

Primary: make a broader, varied nonmoral deliberation corpus, train one seed-0
LoRA, and measure whether it improves alignment under matched ODCV. Dataset
selection uses no ODCV feedback. Secondary stakes work is paused behind this run.

**Dataset complete: 705 accepted; 684 frozen for training.** The selected rows span
12 domains and contain 389 Opus4.8-authored and 295 Sonnet5-authored conversations;
178 received a documented literal correction. All 684 have independent Sonnet review.
The final batch retained 32/33 corrected candidates; one remaining column-count error
was excluded without another repair.

Public [corpus](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-nonmoral-broader-synth)
revision `a3d266e2f0cc48e26e153caf078a5d641ecbbb5c` contains all705 accepted rows and
the complete audit. Public [training mixture](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-nonmoral-broader-7-mix)
revision `f1e61baf643c861920303c7ba1e9844df5f6ed48` contains the selected684 plus
9284 byte-identical replay rows at their original positions. The name's7 is the
rounded synthetic percentage, not a recipe version.

All9968 rows passed token-length checks; maximum8191, with no truncation.
All684 new-example masks were checked in full; the shared generation-boundary
mask gate also passed. New examples total790648 tokens,606746 supervised.
SFT is starting on owned RunPod `epd4o5f97zooij` (2xH200, $9.18/hour), at
code revision `6d5d134d50786a8602acefc3f2a90c76dbd43df3`. Independent watchdog
active; $40 training cap and local backup before ordinary teardown. The first
provision request returned HTTP500 and created no pod; one verified retry succeeded.
Training and the new checkpoint's ODCV result are still pending. Checkpoint100 is
already local with all12 file sizes/hashes verified (3.85GB). Measured transfer time
requires a longer final backup. An independent completion-only full-output preserver
writes `output/nonmoral_broader/20260909/training_retry1/final_verified_backup`.
The existing owner retains its old short transfer timeout; if it enters recovery,
use the independent verified receipt and the explicit procedure in
`output/nonmoral_broader/20260909/readiness/backup_completion_handoff.json` before
termination. Original $40 GPU cap/watchdog unchanged.

Sources and full answers now use **Opus4.8**, with explicit low reasoning effort;
review uses **Sonnet5**. The earlier accepted Sonnet-authored data remain in the
pool with their actual provenance. Opus5's benign-request failures and the documented
Opus4.8 fallback are recorded in the [pilot outcome](2026-09-09_opus_pilot.md).
The production pipeline works; occasional provider refusals are skipped and billed
conservatively, not repeatedly retried.

Recipe12 adds240 short topic cues through the existing variation field. Two batches
use disjoint halves. This addresses observed repetition in recipe11 without new
puzzle templates or a prompt tournament. Earlier candidates retain their original
recipes; no causal generator comparison or alignment improvement is claimed.

## Remaining steps

1. Train one seed-0 rank64 LoRA on **2xH200 RunPod, dynamic batching**, using the
   shared SFT recipe: one epoch, global batch16, learning rate1e-4, ceiling8192.
   Pin the published mixture revision and base revision; verify masks on the pod.
2. Fetch and checksum complete outputs/checkpoints locally before ordinary teardown.
3. Evaluate the new checkpoint on RunPod using the matched ODCV protocol:
   80cells x3passes, temperature0.7, context28000; Docker drives locally.
   Report exact misalignment, submission/progress, paired uncertainty and charts.

## Fixed selection and operating rules

Every source stays unchanged and every example ends in a full single-turn decision
or useful conditional decision. Code, calculations, geometry and games are allowed.
Exclude moral deliberation, essential missing context, substantive errors and forced
trivial choices. Ordinary disclosed assumptions and stylistic disagreements are
not material failures. Explicit self-correction is allowed when the selected final
artifact is correct and no concrete false claim remains.

Allow one documented literal correction of an isolated verifiable slip or incidental
false claim. Preserve the substantive choice and original source. No central-rationale
reconstruction and no repeated repair/rejudge loop. Corrected conversations receive
one fresh independent review; every original and exact edit remains in the audit.

Total project ceiling **$300**, including prior work. Broader-data cap140; stakes
allocation55 remains paused; SFT40 and evaluation20 reserved. Latest completed-phase
project exposure is about **$135.32**, including conservative unknown reservations.
Shared-account balance changes are not this experiment's bill.

Use shared SynthDoc/OpenRouter, HF and RunPod infrastructure. Before this rental,
balances, clean pushed source, public data and local readiness were verified.
Independent watchdogs enforce cost/lifetime ceilings. Before ordinary teardown,
fetch and verify the full inventory of necessary results/checkpoints locally.

## What is already established

Matched historical checkpoints, three evaluation passes each:

| Checkpoint | Misaligned | Submitted | Mean progress |
|---|---:|---:|---:|
| Original nonmoral |33/240 (13.75%)|236/240|4.920833|
| Math |92/240 (38.33%)|215/240|4.770833|
| Table2 only |90/240 (37.50%)|224/240|4.812500|

These are checkpoint comparisons, not multiple training seeds or a deliberation-only
causal effect. The historical18.25% number used a different protocol. All720 matched
rollouts are preserved publicly and all four owned baseline pods were terminated.
The first CRLF-contaminated attempt was archived separately and never scored.

Capabilities tests remain deferred by the user. Reduced capability or increased
refusal would disqualify an apparent improvement; ODCV submission/progress alone
cannot establish capability preservation. Shorter reasoning, a valid retained-CoT
no-comparison control, and nonmoral high/low stakes remain unanswered.
