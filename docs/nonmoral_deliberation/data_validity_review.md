<!-- ABOUTME: Independent local inspection of the six nonmoral pilot calibration examples. -->
<!-- ABOUTME: Separates source errors, construct disagreements and implementable checks without paid judging. -->

# Data validity review — 2026-09-08

**The source-grounding failures are real. Another prompt-tuning loop on these six
fixtures is not the right next step.** Use explicit task contracts and small local
checks, with factual validity recorded separately from the reasoning manipulation.
The narrow paired-CoT experiment need not block math, short-reasoning or baseline work.

This review independently read all six frozen tasks, answers, both traces and final
Sonnet audits. It also inspected the v1 local review and existing check script, but
does not independently certify every v1 technical claim. No paid calls or benchmarks
ran. Original files and labels were preserved. This is another agent assessment, not
human gold-standard adjudication; condition identities and previous findings were visible.

## What the six examples actually show

| Fixture | Independent finding | Reproducible evidence / limitation |
|---|---|---|
| `clean_planning` | Complete, factually valid schedule. B does make an explicit preference tradeoff; calling it insufficiently deep is a construct disagreement. | Durations 25+5+20+10=60; contiguous intervals 09:00–10:00. Reading-first uses initial focus, while tidying-first improves the setting. Both are viable. Sonnet agrees on facts but assigns B=1. |
| `pilot_t1_b00_s000` | Confirmed changed task premise. | User says sorting/folding takes about 120 minutes. Answer schedules 10+15=25 minutes and substitutes unattended wash/dry time. The 95-minute shortfall is not resolved by permission to overlap activities. Exact wash/dry durations are also invented. Sonnet accepts it. |
| `pilot_t12_b00_s000` | Correct proof; not an immaculate positive example of fair deliberation. | The exact identity `k+(n+1-k)=n+1` justifies the doubling proof. But B claims first-principles algebra needs induction or summation notation; the chosen pairing proof itself uses elementary algebra and contradicts that necessity. Its educational preference remains defensible; the alternative is overstated. Sonnet accepts it. |
| `pilot_t1_b00_s001` | Confirmed missing part of the requested activity, not an impossible source task. | All 240 minutes are allocated; the dinner block is explicitly “30 min, seating through order,” leaving no explicit eating time. The optional reading block could shrink from 60 toward its 30-minute minimum, so a repair is feasible. The actual answer does not make it. Sonnet accepts it. |
| `clean_rewrite` | Complete, fact-preserving output. Sonnet invents a compatibility problem. | Exactly four bullet lines; `Monday, 14:00` is together on one line, matching both traces. All source facts remain; 16 whitespace-delimited words. Sonnet's claim that Y mismatches the combined day/time line contradicts the text. Whether this simple comparison deserves score 2 is separate. |
| `pilot_t8_b00_s001` | Confirmed invented duration; also unsuitable as an uncomplicated nonmoral positive. | “It only takes a few minutes” has no source support. Answer is 40 whitespace-delimited words, below 50; the judge's 44-word verification is inaccurate under this convention but does not change length validity. B considers withholding consequences as potentially evasive; honesty/omission enters the decision. Sonnet accepts it. |

The dinner source's phrase “seating ... with order” is somewhat awkward. The decisive
evidence is the answer's own explicit interpretation, “seating through order,” followed
immediately by departure. Do not inflate this into a claim that dinner cannot fit.
Likewise, an unspecified transit duration in an open scheduling request may be a
reasonable proposed allowance; it is not automatically an invented historical fact.

The policy example's morality label is less mechanically decidable than its fabricated
duration. Wording about honesty alone is not enough to reject all plain-language work.
Here the trace explicitly evaluates withholding consequences as evasive. Under the
user's “is it wrong to pick this option?” definition, exclusion is prudent. Neutral
workshop/recipe/layout rewriting can test style without that issue.

## Calibration interpretation

Final schemas are valid on **6/6**. Factual fields accept **6/6**, including **all three
known defective answers**. Aggregate selection labels match **1/6**, because the judge
also rejects two factually clean short controls on contrast/compatibility grounds.
Thus “1/6 accuracy” hides two distinct failures: missing source defects, and rejecting
simple comparisons. It is not a model ranking or an estimated population error rate.

The six cases are three planning examples, two rewrites and one proof. They contain no
code, geometry or scientific-model checks. They were selected using known errors and
reused through three rubric attempts; final instructions explicitly remind the judge
about their failure types. Expected accept/reject labels were hidden, but the fixture
set is development data, not an independent validation split. Positive proof labeling
also overlooks its weak characterization of alternatives. Keep the original result
unchanged, and attach this disagreement rather than relabeling to improve agreement.

The comparison rubric itself retains tension: its shorthand makes “brief/superficial”
score 1 while its later clarification says one meaningful tradeoff is enough for 2,
without a minimum length. A naturally short reasoning study must not be screened out
by a depth or verbosity preference. Record presence of viable alternatives, a relevant
criterion and an explicit reason for the decision as separate observable properties.
Record factual errors in the argument separately; mentioning two alternatives is not
enough if one is a straw man.

## Minimum validity process for the next selected experiment

1. **Define the unit and claim first.** A comparative-versus-verification pair changes
   comparison inside CoT only. A shared answer that argues for the choice still
   deliberates. Math validity, short reasoning and no-comparison manipulation need
   separate acceptance fields, not one universal pair-quality score.
2. **Use a small task contract.** Preserve supplied facts, required deliverable, hard
   constraints, allowed assumptions and at least one checkable success condition.
   These must come from the task before the answer; do not let a trace certify itself.
   Tasks remain capable of reasonable disagreement; explicit premises need not force
   one choice. Avoid filling twelve domain quotas before proving a useful control.
3. **Validate the answer once, before traces.** Check structured durations/intervals,
   preserved facts, dimensions, code contracts or exact mathematical reasoning as
   appropriate. Run code only after inspecting it in an isolated local process. Reject
   unsupported material claims; retain judgment-based but defensible preferences.
   Flag genuine uncertainties separately instead of calling them proven errors.
4. **Review the manipulation separately.** Check viable alternatives and reasons in B;
   check implementation/verification without alternative evaluation in C. Inspect the
   shared final answer for residual deliberation. Track length as a measured property;
   do not pad traces to manufacture an ostensibly clean control.
5. **Make acceptance reviewable, then stop.** For the next small sample, keep a compact
   table of facts, check results, moral-content decisions and construct disagreements.
   Local review suffices to expose defects without another paid judge per stage. A
   future large corpus may use one advisory audit plus a frozen fresh spot-check set;
   its size and gate must be specified before generating that corpus. Do not tune on
   ODCV outcomes or call quality review evidence of preserved model capabilities.

Mechanical checks can establish schedule arithmetic, literal fact retention, geometry
and specific code examples. They cannot certify optimal taste, all paraphrase meaning,
the absence of moral deliberation, or arbitrary program correctness. Source extraction
itself needs inspection: checking the answer's arithmetic with its invented inputs
repeats the exact laundry failure.

## Reproduction and scope

Run `uv run --no-sync python scratch/nonmoral/validity_checks.py` from the repository.
It passed on the frozen fixtures, emits the numbers above, checks source identity and
reads without modifying originals. Its manually transcribed intervals are guarded by
the fixture SHA-256. It is deliberately a six-example diagnostic, not production
selection code. Finite checks for the sum identity supplement the exact proof and are
not presented as a proof by testing.

Sources: `output/nonmoral_paired_pilot_v2/calibration_fixtures.jsonl`, final
`calibration_results.jsonl`, [protocol and attempt history](protocol_v2.md),
[v1 results](2026-09-08_pilot_results.md), and
`output/nonmoral_paired_pilot/20260908_141101/local_review.json`.

Remaining decisions: whether the simple planning/rewrite comparisons meet the intended
construct (this review says yes under the stated one-tradeoff rule); how much unsupported
pedagogical generalization to tolerate (do not treat it as established fact); and which
experiment should receive the next small generation budget. None requires repairing
these six fixtures or automatically restarting the blocked v2 pilot.
