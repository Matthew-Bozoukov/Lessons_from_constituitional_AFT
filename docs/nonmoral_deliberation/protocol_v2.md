<!-- ABOUTME: Prelaunch amendment for the second, Sonnet-only paired nonmoral pilot. -->
<!-- ABOUTME: Preserves v1 failures and specifies cost, calibration, continuation and decision rules. -->

# Protocol v2: grounded tasks, 2026-09-08

Authorized by the user's instruction to fix the failures and go ahead. New API spend
is capped at **$10 total including calibration and the fresh pilot**, separate from
v1's $1.469684 exposure. No full corpus, training or benchmark evaluation follows
automatically. Configuration: `configs/data/synth/nonmoral-paired-grounded.yaml`.

Changes from v1: Sonnet 5 for every model call; explicit `reasoning.enabled=false`
for hidden provider reasoning (the authored B/C CoTs remain present); output caps of
6144 for two scenarios or two traces, 4096 for a shared answer, 2048 for an audit.
Tasks include bounded complete input artifacts and domain-specific fact/constraint
contracts. Answers focus on complete output and modest, supportable explanations.
Trace generation may decline a defective answer instead of rationalizing it. Audit
checks correctness before style, with concrete evidence fields rather than blanket
certification. The calibration checks this rubric; it does not prove universal validity.

Before new generation, freeze **six fixtures**, three unmistakable v1 errors and three
clean pairs (one v1 proof and two locally checked controls). One Sonnet call per fixture,
no regeneration or repair; all six expected accept/reject labels must be recovered.
Labels are hidden from the judge. Calibration and pilot share a cumulative durable
ledger. Stop and inspect a failed calibration before any new generation; no silent
repeat or threshold relaxation. Any later rubric revision needs a recorded amendment.

Fresh pilot: seed 1, two cases in each of the same twelve domains, original denominator
24. Pilot, calibration and future training are distinct splits. Keep all original
stage outputs, including failures. Pause after scenarios and answers for local checks
of facts, bounds, arithmetic and supplied artifacts. These are no-API checks; no paid
scenario-judge, answer-judge, rewrite, rejudge cascade. Reject demonstrably invalid or
empty outputs before downstream generation, retaining their IDs in the denominator.
Do not repair sampled text into a success. One final Sonnet audit per completed pair.

The shared engine retains its bounded JSON parse retries, charged separately; transport
retries are disabled. It still stops a stage on any failed generation. If this happens,
preserve its abort manifest and successful checkpoint, then continue only successful
rows, recording exclusions. A local-check exclusion likewise requires preserving the
original snapshot and recording the replacement subset and reason. No candidate top-ups.

Individual correctness, completion, morality, contrast B=2/C=0, no-padding and length
0.8–1.25 rules are unchanged. Every available complete pair receives local review.
Pilot requires **20/24 individually valid pairs**, at least one per domain, selected
mean length ratio 0.95–1.05, and no unresolved systematic failure. Report uncertain
technical claims separately from demonstrated errors. Preserve external labels and
local adjudication; no use of ODCV outcomes in selection.

The next full-corpus recipe remains conditional on this pilot. Broader tasks must not
silently turn into rote exercises: supplied facts bound uncertainty without dictating
the preferred approach. Synthetic user observations are fictional task premises, never
evidence of real experiments. Cases should permit reasoned disagreement where appropriate.

## Calibration amendment, before fresh generation

Attempt 1 returned nested audit objects containing only the six booleans; x/y/checks/
reasons were unavailable to the parser. The full raw response was not retained, so
whether those fields were omitted or placed as siblings is not established. Its 3/6
nominal agreement is not a valid quality measurement: malformed negatives were being
counted as rejections. All six fixtures are treated as schema failures.

The prompt now explicitly requires one top-level key with every field inside it.
Calibration checks the full schema before a label can count as correct, and the client
retains successful raw calls. Permit one additional six-call calibration attempt under
the same $10 ledger. Preserve attempt 1 and its config; no fixture/expected-label edits,
no fresh scenario generation until this corrected gate passes.

## Output-bound and rubric amendment, still before fresh generation

Attempt 2 produced a valid schema on the first fixture but called its explicit
focus-versus-tidy-space tradeoff superficial because it was not deep or multi-criteria.
That extra requirement is absent from the intended construct. The next response
correctly identified the invented laundry duration but exhausted 2048 output tokens
while repeating its evidence; calibration stopped after two calls, not six.

Preserve both raw calls and the failed first classification. Clarify that one real
case-specific tradeoff is sufficient for score 2, without a length or criterion-count
requirement. Bound evidence to three checks and three reasons, below 600 words, and
raise the audit output ceiling to 4096. The selection threshold B=2/C=0 and frozen
fixtures remain unchanged. Permit one final six-call calibration within the original
$10 cumulative cap. If it fails, do not generate the fresh pilot under an unreliable
judge; report the remaining issue instead of repeatedly tuning to six fixtures.

## Observed outcome and pause

Final calibration: valid schemas on all six fixtures, but **1/6** expected labels
recovered. All three known defective answers were accepted; two clean short controls
were rejected on contrast/compatibility grounds; the clean proof was accepted. This
is evidence that this particular audit setup is unsuitable for automatic acceptance,
not a general model ranking. Repeated rubric development on these fixtures would not
provide an independent validation result.

Across the three attempts: 14 calls, $0.210076 settled token cost, no uncertain
reservations. Together with v1, tracked exposure is $1.679760. Nine offline driver
tests pass. No fresh v2 scenarios were generated: the calibration gate blocked launch.
Raw final outputs and cumulative ledger are in `output/nonmoral_paired_pilot_v2/`;
attempts 1 and 2 retain separate records. At the user's subsequent scope checkpoint,
all further paid work is paused; see the canonical research brief for the proposed
reorganization. Do not silently continue this pilot or weaken its acceptance gate.

Subsequent independent local review qualifies these labels: the proof is mathematically
correct but its characterization of an alternative method is overstated. The short
controls' style ratings are construct disagreements, and the rewrite compatibility
failure contradicts the visible answer. Thus 1/6 is agreement with provisional labels,
not calibrated judge accuracy. All three seeded factual defects remain confirmed and
were accepted by the final audit. See [data validity review](data_validity_review.md).
