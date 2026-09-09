<!-- ABOUTME: Current short oversight card for nonmoral-deliberation research. -->
<!-- ABOUTME: Live decisions and milestones; detailed history remains in LOG and frozen public artifacts. -->

# Nonmoral deliberation: current experiment

Updated 2026-09-09. Intent and constraints: [research brief](research_brief.md).
Detailed evidence and previous checkpoints: [experiment log](../LOG.md).

## Objective and current state

Primary: make a broader, varied nonmoral deliberation corpus, train one seed-0
LoRA, and measure whether it improves alignment under matched ODCV. Dataset
selection uses no ODCV feedback. Secondary stakes work is paused behind this run.

**568 accepted conversations toward 684.** All have full local review plus an
independent Sonnet review. Batch17 added64 unchanged examples; one reviewer
request was refused and its candidate excluded. Another34 literally corrected
candidates await their one independent review. Batch19 has115 admitted source
prompts and is generating full answers. The current public snapshot still holds321
accepted rows; the next publication will include all completed batch evidence.

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

1. Complete local and independent reviews; freeze684 unique selected examples
   with the existing deterministic domain-round-robin selection. Report actual
   domain counts, token lengths, generators, corrections and exclusions.
2. Publish the corpus and frozen mixture under public **dougalldeepmind** using
   shared naming/cards/provenance. Preserve all9284 historical replay rows byte
   for byte, in the same positions; total9968 training rows.
3. Check every row's token length and masks against the pinned Qwen3.6 tokenizer.
   The existing568-row pool is not yet a frozen training mixture. An earlier486-row
   length census found maximum3027tokens and zero over8192.
4. Train one seed-0 rank64 LoRA on **2xH200 RunPod, dynamic batching**, using the
   shared SFT recipe: one epoch, global batch16, learning rate1e-4, ceiling8192.
5. Evaluate the new checkpoint on RunPod using the matched ODCV protocol:
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
project exposure is about **$127.76**, including conservative unknown reservations.
Shared-account balance changes are not this experiment's bill.

Use shared SynthDoc/OpenRouter, HF and RunPod infrastructure. No broader GPU has
been rented yet. Before renting: verify balances, clean pushed source and readiness.
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
