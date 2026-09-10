<!-- ABOUTME: Reconciled production funnel and candid process diagnosis after the user paused execution. -->
<!-- ABOUTME: Separates material failures, excessive screening and untested reset proposals; no new runs authorized. -->

# Paused: what we generated and what went wrong

**We have 198 accepted broader examples and no new broader LoRA or ODCV result.**
We made progress on existing-checkpoint baselines, but have not answered the user's
primary question: can a more varied nonmoral SFT recipe improve alignment? Continuing
to generate batches while our own inclusion rule remained poorly calibrated was the
main execution mistake. Saving a couple of dollars on model reviews did not fix it.

Paid execution is paused by the user's explicit request. The owned batch07 author
process was stopped, its83 complete outputs preserved, and the generation lock retained
against accidental continuation. No batch07 model review or new GPU was launched.
The account currently has two other named pods, matthew-lev and jamie-odcv-dat-7-0908;
neither was touched. Already-dispatched calls retain their full cost bounds.

## Exact broader funnel

Counts below are examples, not API calls, repeated snapshots or repaired versions.

| Batch | Requests generated | Sources admitted | Full answers saved | Final accept | Final reject | Final hold | Model review incomplete |
|---|---:|---:|---:|---:|---:|---:|---:|
|01|24|19|18|7|6|5|0|
|02|24|21|21|11|7|3|0|
|03|120|111|104|56|29|15|4|
|04|120|111|105|48|52|5|0|
|05|119|105|103|39|57|5|2|
|06|120|94|89|37|51|1|0|
|07, paused|120|93|83|pending|pending|pending|not dispatched|
|**Total**|**647**|**554**|**523**|**198**|**202**|**34**|**6 completed-batch failures**|

Of 648 source attempts,647 produced requests. Source review excluded91 and held2,
leaving554 admitted. Of those,523 full answers were saved;31 are missing/incomplete
(21 in completed batches,10 in the interrupted batch). These failures are separate
from substantive rejection.

The six completed batches produced 440 answers:198 accepted,202 rejected,34 held,
6 without completed model review. Acceptance was45.0% of full authored answers, or
45.6% of the434 with completed review. Batch07's83 saved answers have33 provisional
local accepts,45 local rejects and5 unreviewed; none has final model/local approval.

The separate first12 feedback packet produced12 additional full conversations:
2 candidates,7 rejected,3 held. Thus broader work including that packet produced
**659 requests and535 full conversations**. Only198 are in the growing production
corpus, which is not yet a frozen684-row training mixture. The feedback and partial
passes must not be silently added to the training count.

Machine-readable evidence and every source/final/partial decision:
[broader funnel](../../output/nonmoral_broader/20260909/process_reset/broader_funnel.json),
[row dispositions](../../output/nonmoral_broader/20260909/process_reset/broader_row_dispositions.jsonl).
All647 production request strings are exactly distinct; that does not mean they are
semantically diverse. Repeated task structures remain visible.

## What the rejection numbers actually mean

Sonnet's independent review accepted 406/434 and rejected 28. Local final review then
rejected 177 of those 406 model accepts and held 33. Conversely,2 model rejects were
retained after explicit language/style adjudication. **The high rejection rate is
mostly our local gate, not the model review gate.** This does not prove Sonnet is
right: it missed plainly incorrect calculations, code claims and fabricated constraints.
It does mean we should have calibrated the disagreement before scaling it.

The existing manual audit of 111 rejected answers in batches04/05 assigned one primary
reason per row: 45 arithmetic/geometry/software defects; 37 fabricated constraints or
false comparisons; 9 requested explanations missing from the visible response; 20 other
factual, completeness or instruction failures. These counts cover that audit only,
not all 202 final rejects. The categories combine severe errors with incidental slips.

A critical full-text audit found concrete disproportionate exclusions:

| Example | Why we rejected it | What should have been distinguished |
|---|---|---|
|04_021|Desk edge called exactly aligned with a shorter window.|All required footprints and clearances work; the overprecise verification wording is locally repairable.|
|05_109|An optional15-minute fallback claimed to fit whenever under60 minutes remain.|The main schedule fits every supplied duration; the counterexample falls outside those durations.|
|06_050|Reasoning says5:45 for a70-minute process finishing7:00.|The final schedule correctly uses5:50. The internal slip is real, but different from a broken plan.|
|07_004|Reasoning compares including versus omitting an explicitly required table.|The explanation/table are correct. Comparison eligibility and answer correctness are separate questions.|

Source07_001 was also rejected for insufficient per-layer timing data. A disclosed
conditional allocation can yield a feasible140-minute plan, but does not prove the
actual unknown layer durations or global optimum. This is a boundary needing an
explicit conditional-plan rule, not evidence that all source omissions are harmless.

This purposive five-case audit establishes examples of overstrict screening; it cannot
estimate how many of the 202 rejected examples are recoverable. No verdicts were changed.
[Full critical audit](../../output/nonmoral_broader/20260909/process_reset/baseline_gate_critical_audit.json).

## Our process errors

1. **We made example construction harder than the research required.** Twelve domains
became36 detailed miniature-task recipes, often with exact geometry, resource counts,
simulation rules and timing dependencies. Full outputs were explicitly requested by
the user; elaborate mini-worlds were our implementation choice.
2. **We conflated three gates:** usable/correct output, genuine deliberation, and
flawless incidental prose. Zero tolerance for measured capability regression does not
establish that every small annotation slip is an irreparable training example.
We should still remove false claims from training data; targeted verifiable edits are
a different option from accepting errors or discarding everything.
3. **We rewarded rationalization pressure.** Many answers choose sensibly, then invent
why the rival cannot work. Some prompts themselves make one supposed alternative
violate a hard requirement. More injunctions to compare faithfully did not fix that.
4. **We generated around an uncalibrated gate.** The simpler recipe gave37/89=41.6%
acceptance versus87/208=41.8% before it. That was not a demonstrated improvement.
I continued production and optimized review cost before settling whether the gate was
appropriate. Parallel reviewers multiplied throughput without resolving the policy.
5. **We let secondary work displace the main result.** Paired controls, reuse attempts,
judge calibration and stakes development consumed effort before a broader model existed.
6. **We overstated a small operational fix as unblocking.** Deferring Sonnet reviews
would save$2.005170 on fixed batch06 outputs with the same retention. Useful savings,
but not a remedy for task design, review disagreement or time to a trained result.

There was also a concrete anti-repetition mistake:52/120 batch07 source prompts
included rejected/held earlier requests among long historical excerpts. Current
recipe9 removes them, but no recipe9 generation has run. Anchoring remains a hypothesis:
batch06 had high rejection without excerpts too.

The historical successful recipe taught reasoned overrides with artifact slices,
used rewrites, and disabled its final quality filter. Our compliant full-output
preference corpus is a legitimate compound change, not merely broader domain names.
Restoring the old recipe wholesale would violate current user requirements; its high
export yield is not a directly comparable quality score.
[Recipe comparison](../../output/nonmoral_broader/20260909/process_reset/recipe_comparison.md).

## Other work and cost, without double counting

| Earlier lane | Distinct tasks or units | Outcome |
|---|---|---|
|Paired v1|24 tasks;15 complete pairs|1 provisional pass, later qualified;10 reject,4 hold among complete pairs.|
|Judge calibration|6 reused fixtures,14 calls|No new training examples.|
|Historical reuse|Same8 tasks across two attempts|3 final development pairs; no approved training corpus.|
|Fresh overnight paired data|32 tasks|13 valid development pairs;19 failed/excluded; frozen scaling gate failed.|
|Stakes first packet|8 pairs /16 answers|4 candidates,2 reject,2 hold.|
|Stakes generic wrapper|702 historical sources screened;57 answers /27 complete pairs|Manipulation failed; no full answer census and no retained training pairs.|
|Stakes integrated trial|4 attempted pairs;7 parsed answers|0 retained pairs.|
|Existing-checkpoint baselines|720 ODCV rollouts,3 checkpoints|Old nonmoral33/240; math92/240; Table2-only90/240 misaligned. No new broader model.|

These lanes overlap source IDs and contain repairs; summing them as independent new
examples would be misleading. Seven non-broader data ledgers record 1,039 API attempts,
plus 1,681 in the broader ledger: **2,720 data-generation/review API attempts**, excluding
ODCV judging. They are not 2,720 independent scenarios.
[Reconciled other lanes](../../output/nonmoral_broader/20260909/process_reset/other_lanes.json).

Broader exposure is $43.292660, including $0.696850 in four open full reservations.
Whole-project exposure is **$93.202015 / $300** using the existing prior-cost and stakes
ledgers. The separate$21.464404 other-data-lane subtotal is already included across
those categories: do not add it again. These are token charges plus conservative
unknown bounds and GPU lifetime estimates, not provider invoices or shared-account
balance changes. The main loss has been elapsed time and delayed research output.

## Recommended reset, still paused

Use the user's already-approved scenario families and existing examples; do not start
another broad wishes interview or another control. Preserve 198 accepted rows and all
originals. Before more generation, distinguish central reasoning/task failures from
small isolated, objectively checkable corrections. A one-time local correction such
as 5:45→5:50 or removing a false verification sentence can preserve the original prompt
and substantive choice, with original/edit/check recorded. Do not repair missing source
facts, invent a new rationale for a broken choice, or silently relax the selection rule.

Then use one simpler source recipe for ordinary nonmoral choices with modest complete
artifacts, without the detailed puzzle templates or mandatory refutation of a rival.
Show the exact prospective inclusion/edit rule and concrete existing examples before
resuming paid work. Use existing rejected data for that calibration; no paid rejudging
or new multi-batch prompt tournament is needed. This proposal has not been executed,
and no acceptance gain or alignment effect is claimed.
