<!-- ABOUTME: Values-in-advice smoke: calibrated reviewer, structural recovery, and complete candidate audit. -->
<!-- ABOUTME: Separates demonstrated fixes from generator/reviewer failures and records the next pipeline design. -->
# Values-in-advice smoke — 2026-09-17

**FAIL: do not scale. 18 candidates and 18 reviews completed; $2.295152, 88 settled calls.**
Sonnet accepted 15/18; Codex's full read accepts 2/18 under the frozen engineering criteria,
one with disclosed nondecisive weaknesses. These dispositions include conservative admission
failures where scope or target applicability is unestablished; they are not 16 proven dangerous
answers, a human audit, or a calibrated population error rate. Clear unsupported claims suffice
to reject scaling regardless of the more debatable boundary judgments.

## What changed after the previous smoke

The last run stopped on an erroneous rejection of practical wisdom in human advice. It also
attached that same wrong target-mismatch reason to two otherwise correctly rejected examples.
The new author and reviewer instructions explicitly distinguish general values expressed in
advice from specific relationships needed for AI oversight, stable AI values and operator
instructions. One substantive aspect of a principle suffices; not every clause must appear.
Ordinary ethical vocabulary and useful explanations of boundaries are allowed; hidden
constitution labels and narration of unseen editing remain excluded from both trained blocks.

Calibration expanded from 12 to 16 cases, adding valid care/learning and operator-adjustment
examples and negative human-only analogies for oversight and operator instructions. Two
existing negative fixtures now prohibit the spurious target-mismatch reason in addition to
requiring the correct rejection reason. No historical dataset examples reached an author.
**All 16 calibration checks passed.** This is development-set success, not independent validation.

Frozen generation commit: `906a80ca`; separate branch `codex/refresh-lowstakes-nonmoral`,
already containing main through `404dadc6`. Configuration:
`scratch/dataset_refresh/da-lowstakes-values-in-advice.yaml`.
Constitution: `constitutions/claude_distilled_09_principles/constitution.md`, raw SHA256
`6ccd9c2a1ae1f2b479dbecbeb1c735b2014de531a866bcac9b31d8bdf5de914f`.
All calls use `anthropic/claude-sonnet-5`, pinned Anthropic provider through OpenRouter;
$8 hard cap, 18 candidates, four calls each, no replacements or physical-call retries.

## Technical failure and exact recovery

Sixteen of 18 scenario outputs failed the standard JSON contract: most split conversation
fields and applicability into separate objects; two included literal newlines in strings,
and one prematurely closed the outer object before the applicability field. The no-retry
client blocked further physical attempts. The initial run stopped at $0.608464, 34 calls
(16 calibration plus 18 scenarios), with no lost raw response.

At recovery commit `4bfb812b`, a disclosed offline parser combined disjoint objects, decoded
literal string newlines, and removed that single misplaced structural brace. **No string
values were edited and no scenarios regenerated.** Original raw outputs, the aborted manifest,
the original stop record, and per-call recovery hashes are preserved. The unchanged frozen
prompts then produced the remaining 54 calls, using the original cumulative budget ledger.
All 88 physical calls settled; no truncation or paid retry occurred. This is completion after
an explicit parser intervention, not a successful unattended run. The continuation manifest's
$1.69 covers only the continuation; `cost_summary.json` has the complete $2.295152.

Seven offline tests pass, covering attempt and budget guards, calibration reasons, engine
wiring, string-preserving recovery and rejection of ambiguous duplicate fields. The exact
recovery helper is frozen with the run. Hosted sampling remains non-bitwise-reproducible.

## What worked

- The previously rejected practical-wisdom example is now accepted without requiring an AI story.
- All 18 revised reasoning blocks solve the supplied situation without the first smoke's
  hidden-draft editing narration. This does not establish that every trace is well grounded.
- Several examples contain substantive competing considerations and acknowledge real costs.
- The reviewer caught the human-only t6 mismatch and explicit high-voltage stakes.
- Fixed attempts bounded spend and preserved bad examples rather than replacing them until a quota passed.

## What still fails, with inspectable examples

1. **Scenario boundaries are prose requests, not an admission check.** A mains/high-voltage
   starter project appeared under t7. Other scenarios introduced copyright exposure,
   technical hike leadership and fees without affordability bounds. A mundane hobby label
   does not establish low consequences. Both oversight rows describe scripts without
   establishing AI; the human-only board-game t6 row lacks the required identity challenge.
2. **Authors fill gaps with a persuasive story.** `t1_s0001` says 'I've manually restored the
   listing' in a sendable message, without a supplied completed action or conditional instruction.
   `t2_s0000` invents the cause of earlier delays: no deadline and no default. `t3_s0001`
   asserts the 70% cleanup share exceeds the extra cooking mess without data separating them.
   `t9_s0001` turns a prediction about alternating attendance into an established causal
   history of earlier hobbies failing specifically because they alternated weeks.
3. **The reviewer confuses answer claims with source facts.** In `t8_s0000`, both answer and
   review change 'arrives in 5–7 days' into 'ships in 5–7 days'. In `t9_s0001`, the reviewer
   calls the strengthened hobby history user-supplied. In `t1_s0001` it actually notices the
   invented restoration but excuses it as an editable draft detail. A sendable message is
   supervised output; editability cannot excuse an unmarked false completed-action claim.
4. **Target applicability remains unstable.** The t1 reviewer explicitly says no literal AI
   is involved but passes it by reclassifying the oversight target as general practical wisdom.
   General-value permission must not silently dissolve a relationship-specific target.
5. **Leakage checks overreach in the other direction.** The reviewer rejects `t8_s0001` for
   'I should also acknowledge the cost here'. Planning what the visible answer should include
   is normal object-level reasoning; it is not reviewing an unseen draft or reciting a target.
6. **Useful help can still be deferred.** `t3_s0000` explains its objection and then asks
   whether the user wants the actual survey draft they already requested.

The full per-row record distinguishes unequivocal errors from uncertain boundaries and minor
weaknesses. Against this independent read, the reviewer has 14 false acceptances and one false
rejection; those counts depend on the disclosed engineering rubric. Two rows pass this read;
only one of those also passes the model reviewer. No training-ready release is approved.

## What should be generally different next

Use four purposeful calls rather than paying an unconditional rewrite to polish every scenario:

1. **Scenario:** one explicit JSON object with a literal schema. Choose a genuinely bounded
   decision, resources, authority and time horizon; preserve a real cost or uncertainty.
   Keep target/application metadata separate from the natural human conversation.
2. **Scenario audit and source record, before any answer:** reject scope/target failures;
   quote the supplied facts, mark unknowns, and identify the actual competing considerations.
   This reviewer sees no proposed answer, so it cannot copy the answer's inventions into facts.
   Validate quoted spans mechanically. Do not treat the generator's own applicability claim as proof.
3. **Answer:** produce standalone deliberation and useful advice. Separate existing facts,
   uncertain inferences and proposed future actions. A message may say 'I propose to restore'
   or use an explicit conditional placeholder; it may not silently claim restoration already happened.
4. **Answer audit:** compare exact consequential claims and sendable-message assertions to the
   original conversation and pre-answer source record. Check arithmetic, timing, authority and
   whether the requested deliverable was actually supplied. Planning a visible answer is allowed;
   references to hidden drafts, assigned principles and constitutions are not.

Keep single attempts, preserve failures, measure first-pass yield, and do not auto-repair or
replace rejected candidates. Rejecting before drafting saves calls. The source record itself
can be wrong: it must remain auditable against original text, not become an unquestioned new
source. Use fresh held-out reviewer challenges in addition to the development fixtures; do not
keep expanding a memorized set and call success general reliability. This design is a next
experiment proposal, not demonstrated success, and was not launched in this turn.

This smoke does not establish that Sonnet is incapable, or explain the prior ODCV gap causally.
It does show that fluent self-review of the whole answer is an inadequate quality gate here.
No further dataset-scale generation, mixing, training or evaluation was started.

## Reproduce and inspect

Initial command:
`uv run --no-sync python -m scratch.dataset_refresh.constitution_smoke --config scratch/dataset_refresh/da-lowstakes-values-in-advice.yaml`

Recorded recovery:
`uv run --no-sync python -m scratch.dataset_refresh.recover_smoke_scenarios --root output/2026-09-17_da_lowstakes_values_in_advice_smoke/20260917_171026 --continue`

Diagnostic artifact: [full report and all candidates](https://huggingface.co/datasets/dougalldeepmind/2026-09-17-da-lowstakes-values-in-advice-synth-smoke).
The repository contains frozen inputs, all original calls, structural recovery, stage snapshots,
all 18 candidate conversations, reviewer verdicts and per-row independent dispositions.
