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
**SFT completed successfully: 623 steps, one epoch, mean training loss0.834213.**
Training runtime6632.5seconds. Exact data/base pins, world_size2 and token budget8000
were verified. Public [adapter](https://huggingface.co/dougalldeepmind/2026-09-09-qwen36-0-nonmoral-broader-7)
revision `d52838446ef134088841e8dc436094d93827487f`; public weight hash matches
the remote trained adapter. No alignment result yet.

**ODCV launched** with this exact adapter: owned H100 pod `9ybfav9mzboi6y`,
$3.49/hour, independent watchdog, unchanged $20 total evaluation cap. Docker,
network capacity and all168 physical shell files passed preflight. Frozen plan:
`output/nonmoral_broader/20260909/evaluation_plan.json`.

The training pod `epd4o5f97zooij` remains alive for verified local preservation.
Checkpoint100 is already local (3.85GB; all12 file hashes verified). Independent
final backup is copying8.99GB/46 files to
`output/nonmoral_broader/20260909/training_retry1/final_verified_backup`.
The original owner has a short transfer timeout; if it enters recovery, reconcile
the independent verified receipt using
`output/nonmoral_broader/20260909/readiness/backup_completion_handoff.json` before
termination. Original $40 training cap/watchdog unchanged. Evaluation runs in
parallel with this transfer; the training pod cannot be torn down just because
its adapter is public.

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

1. Verify the complete final local training backup and terminate the owned H200 pod.
2. Complete matched ODCV:80cells x3passes, temperature0.7, context28000; Docker local.
3. Verify local evaluation artifacts, terminate the owned H100 pod, and publish exact
   misalignment, submission/progress, paired uncertainty and comparison charts.

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
