<!-- ABOUTME: Reviews observed refresh costs and a saved-stage completion design with exact source evidence. -->
<!-- ABOUTME: Records offline corrections and the bounded four-case proposal without changing spending policy. -->
# Corrective completion design — 2026-09-15

**The offline correction reversed 41 exclusions; one newly established duplicate was then re-held, leaving 40 net restorations without new model calls.** The current selections are 716 low-stakes and 650 nonmoral rows. The next reviewable step is a disabled four-case nonmoral saved-answer revision proposal, below; the packet remains disabled until root dispatch, and no budget increase is needed for these four calls. Reuse eligible source conversations and sound answer content rather than restarting the six-call pipeline. This design review itself changes no frozen recipe, accepted label, or source artifact.

The historical closed pools contained 706 low-stakes and 631 nonmoral rows. After the root's evidence-bound corrections and quota selection, the remaining shortfall is 66 nonmoral rows; low-stakes has reached 716. The ledger has 11605 calls, 11534 settled and 71 billing-verified failed-content calls, with no active or uncertain reservations. Conservative charges are **$249.2113677**; only **$0.7886323** remains under the existing$250 ceiling. There is no defensible paid-only completion promise for the remaining 66 rows from this headroom. Restoration totals and selected totals differ because of trait quotas. Current selection audits and correction provenance are separate from this historical cost census.

## What the contract actually requires

| Requirement | Implication for correction |
|---|---|
| Moral low stakes and original varied craft deliberation; advice to a human, comparable to difficult advice in other respects | Preserve genuine competing pulls and substantive reasoning; allow justified retention, rejection, or synthesis. Do not force every answer to override an instruction or compromise. |
| New09 constitution and Sonnet authorship | Reuse final answers only after exact-source/target compatibility checks. Low-stakes authors use new09; nonmoral authors use the qualified craft preference, with new09 as a compatibility constraint rather than an injected ethical dilemma. Preserve historical model/provenance distinctions. |
|716 synthetic rows per arm and new pinned nosynth replay | Keep per-trait quotas and exact716+9284 mixture composition, seed0 and common replay sampling. Release only complete, checked selections. Do not label incomplete pools train-ready. |
| Comparable training presentation | Keep native reasoning/final separation, model-neutral system/user text, source-grounded advice, correct Qwen8192 rendering/masking and duplicate checks. Report length differences rather than padding them away. |

The following are implementation choices, not additional user requirements: paid preflight on every scenario; two paid final judges; automatic repair after any adverse judge label; numeric draft-length gates; a mandatory nontraining `<changes>` note; and broad word bans standing in for semantic leakage checks.

There is an important historical distinction. DA and old moral-low-stakes recipes already require 700 characters in **both** draft and final fields. The original craft recipe deliberately uses 500: its comments explain that700 trimmed legitimate short examples rather than detecting degenerate ones. Refresh imposed 700 on both arms. Removing a draft-only stopping condition is a new, documented construction policy, not a claim that the old stage passed. A short draft may be useful untrained rewrite input; a short final answer still needs substantive review. A model-neutral identity rule also differs from banning every occurrence of “Claude” or “Anthropic.”

Sources: [DA stages](../../configs/data/synth/da.yaml), [original craft minimum and rationale](../../configs/data/synth/nonmoral-deliberation.yaml), [current nonmoral contract](../../configs/data/synth/nonmoral-advice.yaml), [local checks](../../scratch/dataset_refresh/run.py), [per-row execution](../../scratch/dataset_refresh/per_row.py). The original DA response contract is draft → revision → local lint/export, with corpus observations configured to warn; it does not establish the refresh's per-row six-call minimum.

## Observed cost and yield

These are conditional historical means from the three primary Sonnet phases, **not prices or forecasts for a new prompt**. Calls span evolving prompts and selected survivors.

| Stage and requested thinking mode | Calls | Mean charge | p95 charge |
|---|---:|---:|---:|
| Scenario, disabled |1724|$0.01327|$0.01630|
| Preflight, requested1536 |1715|$0.00362|$0.00742|
| Draft, disabled |1733|$0.02060|$0.03080|
| Revision, provider default |1690|$0.04320|$0.06589|
| Grounding, requested2048 |1959|$0.02654|$0.06564|
| Target review, disabled |1872|$0.01871|$0.02428|

The six stage means sum to approximately $0.126 before rejection/repair. Resuming a saved final through the existing two judges averages approximately $0.045; resuming a saved draft through revision and both judges approximately$0.088. These sums are illustrations of historical stage burden, not accepted-row cost estimates.

Only 17/1750 saved primary preflights fail their frozen required-true gates; the 1750 calls cost $6.729010. This supports checking known scenarios offline before paying for answers. It does not establish that the17 failures were wrong. Nonmoral target review also already checks groundedness, self-containment, leakage, genuine tension, and constitutional compatibility; some work overlaps the separate grounding critic. Blind source-first assessment remains valuable, but two model passes are not two independent proofs.

Automatic repair plus numbered1 critics cost **$19.8058784 across635 calls**, including technical retries. Of 213 candidates with automatic repair, 158 passed both saved model checks and 131 remain exclusion-aware accepted. The code deliberately pays for target review even after grounding fails so repair sees both diagnoses; 206 such target calls cost $3.7527596. Those diagnoses can be useful, but they should not be bought automatically when no repair is planned.

The 34-case nonmoral technical-review salvage batch illustrates the risk of optimizing model-pass yield:73 calls ultimately cost $2.8364451 after billing reconciliation ;20 automatic accepts became 7 independent passes, 4 holds and 9 rejections. That is a selected difficult cohort, not a general recovery success rate. Prefer exact-content adjudication before further paid review.

**Requested thinking caps did not reliably preserve output headroom.** Among1959 grounding calls requesting 2048 reasoning tokens,90 finished for length. Raw calls 11325,11332 and 11409 explicitly requested that budget but diagnostics show essentially the entire 6000-token completion allowance used for native reasoning. The local record cannot determine whether the provider ignored the limit, mapped it to effort, or used different accounting semantics. Increasing or reducing a configured number without measuring actual output is not a demonstrated fix. See the saved raw hashes/diagnostics in the evidence supplement. This extends the existing [reasoning-budget and judge-rubric gotchas](../GOTCHAS.md); it does not justify switching off thought and trusting the resulting judge. Earlier purposive calibration found material misses and schema failures, and changed prompt plus thinking together, so it cannot isolate thinking's benefit.

## Smallest reliable completion procedure

1. **Freeze one explicit correction manifest.** Prioritize missing traits and source diversity. Bind original root, source/result/answer hashes, source eligibility, corrected target text, and the exact reason for reconsideration. Read actual system/user and both answer fields. Separate a substantive defect from a disputed presentation rule or audit-format failure. Existing exclusions remain historical evidence.
2. **Spend no API calls on unchanged sound answers.** Independently review source facts and constraints before target conformity; record quoted material reasons and a defensible benign reading. Only then resolve target compatibility. Derive lineage and audit metadata locally. An adverse model verdict is evidence to adjudicate, not authority to repeat or erase. Do not silently overwrite old failures or fabricate missing frozen reviews.
3. **If a final needs repair, use its best complete saved draft/final once.** The eligible scenario and existing DA-style draft remain fixed. One explicit Sonnet revision may address an independently verified material defect; no scenario regeneration, second draft, paid preflight, or automatic second repair. Return only training reasoning and final content; an absent audit-only changes explanation must not destroy otherwise complete output. The changed answer receives a fresh independent full read. Fresh scenarios and all-new one-shot generation are separate, unvalidated fallback work.
4. **Use an explicit new acceptance/provenance contract.** `per_row.verify_accepted` correctly requires the old preflight/two-judge chain. Do not bypass it or fake those checkpoints to make offline-adjudicated rows look like old automatic accepts. A correction/adoption record must name the different review route, retain original artifacts, bind exact final bytes, and be validated by the publication helper. Source/domain/trait checks, true material defects and training-format checks remain intact.
5. **Preview the actual publication transformation.** Run exact quota/unique-source checks, full duplicate adjudication, and native Qwen token/mask checks on the selected rows, then prepare the exact716+9284 mixtures from the pinned nosynth commit. Keep author audit text out of training metadata. Compare length distributions with DA; the closed nonmoral pool is substantially longer. Preserve all outcomes and provenance, including unresolved exclusions.

The least costly review route uses the existing independent reading process without new paid endpoint calls. If a paid critic is retained later, use a small source-first material-defect schema; fields such as descriptive reasoning-move labels and self-reported word counts are optional audit output, not reasons to lose an answer. Do not replace two judges with an uncalibrated combined judge and assume equal protection. Do not introduce Haiku.

## Bounded pilot and stopping rules

The root reversed 41 original exclusions with preserved correction records, then re-held one low-stakes case after a different comparator established a real duplicate:40 net restorations. Actual selection previews produced 716 low-stakes/650 nonmoral after one additional lossless parsing recovery of a completed review; that recovery made no model call and preserved the original failure. The earlier eight-case idea was a proposal, not authorization, and is superseded by this completed correction and the concrete next-step packet.

**Proposed paid pilot, disabled:** four currently held, originally accepted nonmoral rows from four missing traits. Each actual system, user, reasoning and final was read independently for this proposal; prior accepted provenance and current exclusion hashes were verified. All four have eligible unchanged craft-advice sources and no previous independent rewrite checkpoint.

| Candidate | Concrete repair target | Existing request reservation bound |
|---|---|---:|
|t1_004_v0|Printed build cards falsely claimed to physically prevent wrong strut order|$0.1706275|
|t3_001_v0|Transfer test never exposes its reader to the first-model naming cue; prior one-on-one testing is invented|$0.1663600|
|t5_007_v0|“Fifth” warning line inserted after the third field, contradicting the learned fourth-line cue|$0.1670650|
|t6_012_v0|Invented album for the postponed lake trip drives the proposed cross-reference/test|$0.1681500|

Each request uses the existing complete final as draft, the full frozen qualified craft section and final revision template, plus a narrow source-bound correction instruction. Keep Sonnet 5, temperature 0.7, max_tokens 12288 and its existing default thinking setting. Only one author call per case, workers 1, four physical calls maximum, no retries, no subsequent automatic repair, no paid critic and no automatic adoption. Both complete training fields are mandatory; an absent nontraining changes tag alone must not trigger another call. An independent full read must reassess actual constraints/material defects first, then the craft preference and **the full new09 constitution** using the original compatibility criteria. The packet includes that full review text, not only a hash or an inherited previous pass.

Exact byte-bound reservations sum to **$0.6722025** using the existing registry's $2/$10 per-million-token Sonnet input/output rates and the frozen BudgetClient formula. Adding all four maximum reservations to the closed spend gives **$249.8835702**, leaving $0.1164298 under 250. Thus this pilot fits the existing cap at these bound assumptions; no additional cap is proposed for it. The price registry is hash-bound and was not refreshed through an API in this task. The historical revision mean/p95 were $0.04320/$0.06589; actual pilot cost and retained yield remain unknown.

The complete disabled JSON/YAML packet is at `output/2026-09-15_dataset_refresh_correction_review/process/nonmoral_repair_pilot_proposal/`. `proposal.json` binds the four sources, full answer hashes, exact request files, registry, current ledger and independent review contract; `proposed_config.yaml` sets execution/dispatch false and has no executor. The existing `repair_independent.propose` is unsuitable unchanged because it always dispatches two paid critics. The standalone `scratch/dataset_refresh/saved_input_pilot.py` adapter now prepares a disabled dispatch packet, rechecks all four source/request bindings and the original shared ledger before each call, and makes at most one Sonnet author call per case. It stops on any failure, preserves raw outputs outside the source records, and leaves all complete answers awaiting independent full review. It has no adoption or publication command. Root reviews the tested helper and enables its separate dispatch file before use. **No dispatch occurred.**

Stop on any ambiguous charge, empty/truncated output, missing trained field, material defect surviving purported acceptance, changed source constraint, provenance mismatch, or budget reservation refusal. Record requested versus actual reasoning/output tokens, all-in charged cost, number of independently retained rows **by missing trait**, and cost per retained row. Do not let a parsing-success rate or model-pass rate authorize scaling. Report remaining shortfall honestly if offline salvage and a genuinely authorized bounded pilot cannot close it.

Evidence: `output/2026-09-15_dataset_refresh_correction_review/process/` contains per-call costs/settings, per-candidate outcomes, automatic-repair outcomes, stage summaries and the raw-diagnostics supplement. Closed ledger SHA256: `ed8a7af75b5397c117e055c72fca453f942bf7c3118a6f1b66dd3a61204d4631`. “Effective accepted” here means current exclusion-aware status, not a new independent clearance of every row. No model calls, uploads, training or frozen-file changes were made for this review.

Offline preparation after helper review/commit (creates a disabled dispatch file; no model calls):

```powershell
uv run python -m scratch.dataset_refresh.saved_input_pilot prepare --config output/2026-09-15_dataset_refresh_correction_review/process/nonmoral_repair_pilot_proposal/proposed_config.yaml --output output/2026-09-15_nonmoral_saved_input_pilot
```

After the zero-call correction archive is frozen, root can enable only the new output directory's `dispatch.yaml` and explicitly invoke `execute --output output/2026-09-15_nonmoral_saved_input_pilot`. Preserve the original disabled proposal. Both lock acquisitions have a one-second timeout. The inherited HTTP timeout is420 seconds, SDK retries are zero, and the shared client allows one physical attempt. A timeout keeps its reservation and stops the entire pilot; there is no automatic resume after an interrupted start marker. Twenty-one offline tests cover exact call count, unchanged source messages, full constitution intake, missing audit-only changes, malformed/truncated output, wrong response model, leakage, budget anomalies, changed bindings and repeated-run refusal.
