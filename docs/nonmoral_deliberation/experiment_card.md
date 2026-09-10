<!-- ABOUTME: Current short oversight card for nonmoral-deliberation research. -->
<!-- ABOUTME: Live decisions and milestones; detailed history remains in LOG and frozen public artifacts. -->

# Nonmoral deliberation: current experiment

Updated 2026-09-10. Intent and constraints: [research brief](research_brief.md).
Detailed evidence and previous checkpoints: [experiment log](../LOG.md).

**Latest: grounded revision stopped at pilot gates.** Two16-source pilots cost
$2.663944 total. Corrected pilot:10/16 locally usable changed revisions and3 concrete
improvements, below the required12 and4; full-data forecast$71.61 exceeded its$50cap.
No production, new SFT/ODCV or stakes run was launched. All candidate pairs, reviews
and costs are retained; [exact outcome](2026-09-10_grounded_revision_results.md).
The completed broader-model result below remains the latest alignment measurement.

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
the remote trained adapter. Final result: **76/240 misaligned (31.67%)**, versus
original nonmoral **33/240 (13.75%)**. This broader candidate performed worse.

**ODCV complete:** all 240 rollouts and both sets of 240 judgments retained, no
terminal API deaths or outer scenario timeouts. Five rollouts hit token limits.
Submission:235/240; mean progress:4.916667/5. Original:236/240 and4.920833/5.
Broader minus original MR: **+17.92 percentage points**, paired scenario95%CI
**[+8.93,+26.90]**. These are fixed-checkpoint results, not training-seed uncertainty.
Public [evaluation](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-odcv-qwen36-0-nonmoral-broader-7)
revision `fe7b98403d7efca11764fc94a5e7443b720c77ee`;
[comparison and charts](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-nonmoral-broader-comparison/tree/main/results)
revision `dc681d897efb57d4f16680678e3257210b60db24`.
Public eval files were verified against local bytes. Remote logs were hash-verified
before H100 termination. All owned pods are now absent from RunPod (individual404s).

**Backup incident:** RunPod reports training pod `epd4o5f97zooij` was stopped
at19:36:38UTC with reason "Exited by user"; actor unknown. Our owner remained in
`artifact_recovery_required`, ordinary teardown blocked, and its watchdog had not
reached its21:03UTC deadline. No stop was issued by this thread's recovery agent.
Final backup received6.560/8.987GB before disconnection; **full archive verification
was not achieved**. Checkpoint100 remains fully verified locally. Salvage retained26 complete files. All9 final-adapter files were individually
verified against the exact public HF revision; checkpoint623 weights also match.
Checkpoint600 is fully received but lacks individual remote-hash verification.
Checkpoint623 optimizer is truncated;19 later archive members were not received.
Owner-state run metadata and125 log records are separately labelled reconstructed.
The verified final model is local and public; ODCV continues unaffected.
Evidence: `output/nonmoral_broader/20260909/training_retry1/unexpected_stop_incident.json`.
Do not claim all outputs preserved. The training resource subsequently disappeared
before this thread's watchdog deadline; stop/deletion actor remains unknown.
Provider absence was verified before stopping the obsolete local training owner.
No GPU budget increase or training rerun occurred because of this incident.

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

## Next research decision

The broader run is finished, and the follow-up grounded revision candidate stopped
at its predeclared pilot gates. No paid work is running for this thread. The revision
prompt largely preserved the old decision logic; its quality/cost gates do not support
scaling it. Choose a substantive next intervention before spending further. Neither
result identifies the cause of the broader-model regression. Stakes and formal
capability testing remain untried here and paid-paused.

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

Total project ceiling **$300**, including prior work. Latest recorded project exposure
is **$167.894061 / $300**, including conservative historical unknown reservations:
previous165.230117 plus grounded-revision pilots2.663944. The revision's unused
SFT40/evaluation15/stakes5/recovery15 allocations were not spent; its data phase
stopped at the pilot gates. Earlier allocation proposals do not authorize new runs.
All480 judge requests settled. These are estimates, not provider invoices.
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
| Broader nonmoral |76/240 (31.67%)|235/240|4.916667|

These are checkpoint comparisons, not multiple training seeds or a deliberation-only
causal effect. The historical18.25% number used a different protocol. All720 matched
rollouts are preserved publicly and all four owned baseline pods were terminated.
The first CRLF-contaminated attempt was archived separately and never scored.

Capabilities tests remain deferred by the user. Reduced capability or increased
refusal would disqualify an apparent improvement; ODCV submission/progress alone
cannot establish capability preservation. Shorter reasoning, a valid retained-CoT
no-comparison control, and nonmoral high/low stakes remain unanswered.
