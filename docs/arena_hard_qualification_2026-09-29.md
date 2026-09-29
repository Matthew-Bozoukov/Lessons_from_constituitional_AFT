<!-- ABOUTME: Arena-Hard reliability repairs and bounded qualification after the detailed review. -->
<!-- ABOUTME: Separates historical score reproduction, judge calibration and fresh Qwen transport evidence. -->

# Arena-Hard qualification, September 29

This follow-up preserves the historical hard-primary, length/Markdown-controlled
scorer, rubric, ties, bootstrap and default 500 hard / 250 creative questions.
Following the live panel below, the user explicitly selected GPT-4.1 as primary
and Gemini as an auxiliary diagnostic. Report version 3 records that policy;
complete primary paired judgments remain mandatory. Qwen-only scope uses
`dougalldeepmind/2026-09-22-qwen36-0-nosynth`
at `633908b72a9799fb3e6b101b0a8a82aec3c3d642`.

## Confirmed repairs

- A score embedded in a truncated judge response used to satisfy completeness.
  New judgments require normal completion plus a recognized verdict. Full provider
  responses, reasoning, finish reason, returned model, usage and reported cost are
  retained. Failed attempts and scrubbed errors remain evidence, not verdicts.
- The vendored UID-only cache could permanently skip failed judgments or reuse a
  changed comparison. Cache identity now covers exact answers, prompt, ordering,
  rubric/parser, judge and inference settings. Only matching complete orderings
  are reused. Bounded retries and atomic snapshots preserve the other ordering,
  failed attempts and invalidated historical entries. SDK retries do not multiply
  the configured attempt bound. Cost accounting includes failed attempts.
- Reused answer metadata and answers could be fetched at different revisions.
  Resolution now pins one dataset commit and honors an explicitly selected
  revision. Generation protocol metadata certifies prompt identities, thinking,
  sampling, extraction, effective context and actual per-question allowances.
  Missing historical protocol metadata does not become a matched new comparison.
- Four-question smoke comparisons used the full 100-question calibration gate.
  Smoke still requires all primary judgments but reports calibration `not_assessed`,
  with no capability pass/fail. Full runs now report agreement/gap on the complete
  shared subset of a requested 100-question auxiliary panel without a calibration
  pass claim. Explicit legacy `policy=gate` remains available. Generation health is descriptive evidence, not a new rule
  that rerolls refusals or other valid model outcomes.
- The active config no longer embeds the July training ladder or advertises
  automatic staged stopping/fallback that the current workflow does not perform.
- Interrupted generation tolerates only a torn final checkpoint line; interior
  corruption remains an error. Original torn bytes survive successful recovery.
  Cache identity also includes actual context and effective output allowance.
- Live launch exposed a CLI parsing bug: a spaced `--reference VALUE` could be
  misread as a positional OmegaConf override. Dynamic runner flags now register
  before parsing intermixed flags/overrides, and invalid overrides fail before
  acquiring a pod. Fifteen CLI regressions cover this boundary.

Historical score-only judgments remain readable by the statistics module for
reproduction; they cannot certify a new judge run without completion evidence.

## Qualification protocol

The one-off driver is `scratch/arena_hard_qualification.py`. It prepares an isolated
harness from the current source, writes protocol/provenance under `metadata/`,
retains self-contained answer and judge transcripts under `rollouts/`, and publishes
instrument results separately from any candidate capability comparison.

Historical source: dataset
`dougalldeepmind/2026-07-31-qwen36-27b-capability-eval-arena-hard`
at `9b1c2802a5b9aded7c086cdb0b7cb06ebe714229`. The source answers compare July's
20% synthetic arm against its matched 10% reference. They are useful for testing
judges on real differing answers; they are not a new nosynth-reference comparison.

1. Rescore the saved historical 20% judgments without changing their interpretation.
   Reproduced 49.21214444102081%, interval [42.06935313929543%, 56.29683921837892%],
   n=148, 2,000 bootstrap draws. There are 294 parsed games; this does not meet a
   fresh 150-prompt/two-ordering completeness requirement.
2. Run both configured judges on a 10-question prefix to inspect actual responses
   and cost, then extend the same cache to the fixed 100-question panel. The pilot
   does not decide whether to stop based on a scientific score.
3. Generate four hard and four creative answers from the pinned Qwen control with
   the standard entrypoint and settings. Inspect finish/reasoning/answer fields.
4. Use those same answers on both sides of a separate identical-answer control.
   Report ties and position consistency; never call this a treatment comparison.
5. Preserve calibration failures rather than switching judges or relaxing gates to
   manufacture a pass. Hand review disagreements to distinguish rubric differences
   from transport/parser bugs.

## Live judge panel

The 10-question pilot completed all 40 calls for $0.441963 reported provider cost.
The cache was then extended to the fixed 100 questions without selecting on score.

- Gemini completed 98/100 paired judgments. Prompt `350b86ce6d5a4b0c` failed in
  ordering 0; `5d2943f87d8f477b` failed in both orderings after three attempts each.
  The judge generated code answering the embedded user request instead of giving
  a comparison verdict. One attempt also exhausted its output budget. These are
  retained failures, not omitted prompts or invented ties.
- GPT-4.1 completed 100/100 pairs (200 judgments). Running it separately after
  Gemini's failure preserved the independent evidence; it did not qualify Gemini.
- On the 98 complete shared prompts only, agreement was 61.2245% and the mean
  preference gap was 5.6122 percentage points. These are **incomplete-panel
  diagnostics**, not a passed calibration result. Both miss the configured 80% /
  3-point criteria even before the completeness failure.
- Reported judge costs including the cached pilot and failed attempts:
  Gemini $2.13388815 (209 calls), GPT-4.1 $2.957636 (200 calls), total $5.09152415.

Assistant review of the pilot's five disagreements checked all 40 saved prompt /
answer orderings and terminal verdict mappings. No harness inversion or parser
error was found. Examples included contradictory preference under swapped order
(`dfc9be7c176d46bb`), incorrect code-reading claims (`2edbb5f36f5b42be`,
`c1dcc4caf8174b3a`), different preferences for interface coverage versus clarity
(`8411a709b22b408a`), and a tie versus small wording advantage
(`d5cdf24c4e614beb`). This review is not a human gold-label calibration and does not
establish either judge as ground truth.

The user approved GPT-4.1 as primary and cross-judge disagreement as a diagnostic.
The original failed Gemini qualification remains preserved; the new policy is an
explicit measurement decision, not evidence that either judge became calibrated.
Auxiliary coverage and shared-pair denominators are reported even when incomplete;
primary incompleteness still prevents a leaderboard. Per-judge request settings
are now separate, with empty extra settings for GPT-4.1 and low reasoning for Gemini.

Published original evidence: [historical answer judge validation](https://huggingface.co/datasets/dougalldeepmind/2026-09-29-arena-hard-historical-answer-judge-validation),
revision `5a8c1b9ea92c3c5b3789935b05cd4be73ffa2569`.

An additional offline `approved_policy_reanalysis.json` applies the selected
policy to those saved records without further calls: primary 100/100, auxiliary
98/100, 98 shared, `passes=null`. It retains the original request config and notes
that the old GPT-4.1 panel inherited a low-reasoning extra setting; new requests
use the separately configured settings. It does not rewrite the original failure.
The artifact revision including that reanalysis is
`2d8dbff94651b4040af5aadeda8e5cbde086e883`.

Local validation: 195 tests passed across Arena judgment recovery, policy,
answer reuse, statistics, CLI parsing, eval framework and serving transport.
The implementation is committed at `2ad51c8e`, following `8f0a27a4` and `540c2253`.

## Fresh Qwen smoke and identical-answer control

The standard `uv run evals` entrypoint generated four hard and four creative
answers using the exact nosynth pin above, its training base revision
`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, vLLM 0.26.0, thinking enabled,
temperature 0, top-p 1, 16,384 context and a 6,000-token shared reasoning/final
budget. The per-question recorded allowance was 6,000 for all eight questions.
Generation took about 131 seconds after serving startup. No answers were rerolled.

- All eight records retained nonempty reasoning. Five ended normally.
- Three hit the output cap: hard prompt `2edbb5f36f5b42be` and creative prompt
  `0a055d8b33ca49b1` had empty final answers after exhausting the budget in reasoning;
  creative prompt `5ab052c7cef84016` had a repetitive, truncated final answer.
- This is a real limitation of the current budget on this small sample, not a
  parser failure. Keep these outcomes in the denominator. The smoke is too small
  to estimate full-benchmark truncation reliably.
- GPT-4.1 judged all eight identical-answer pairs in both positions: 16/16 ties,
  100% position consistency, mean preference 50% for both slices.
- Gemini completed the four-hard-prompt diagnostic: seven ties and one `A>B`.
  On one ordering of the empty/empty Zig answers it invented differences between
  code answers that were absent. Shared-prompt agreement was 75%, gap 6.25 points,
  and Gemini position consistency 75%. This did not block the primary result.
- The qualification driver initially omitted the explicit smoke flag before its
  four-question validation. The guard stopped it after primary judging; the error
  is preserved under `results/attempts/`. Fixing the flag reused all 16 primary
  judgments with no new primary calls, then collected eight Gemini judgments.

Published [Qwen smoke and identical-answer evidence](https://huggingface.co/datasets/dougalldeepmind/2026-09-29-arena-hard-identical-answer-control)
contains original generation metadata/answers/health plus both judges' complete
raw responses. This is an instrument check, not a trained-model capability gain.
Its verified dataset revision is `dec7537b7ffeabccd4b2552754506770aac52d5d`.

Reported judge spend: self-control GPT-4.1 $0.102104, Gemini $0.050259; combined
with the historical panel, **$5.24388715**. GPU rental was approximately $2–3
including unsuccessful startup/CLI attempts; this is a rate-times-duration
estimate, not an invoice or shared-account balance delta. All four pods acquired
by this task (`olgohmuz8esljn`, `u0ycbi6vbfiaot`, `u4va88da8ixb0v`,
`29g6c8ox2l47ns`) were confirmed absent afterward. Other tasks' resources were
left untouched. The successful H200 host was automatically terminated after
generation, before the judge calls.

Final review also corrected a legacy-gate reporting edge case: a judge process
failure with complete cached pairs can no longer be persisted as `passes=true`.
The 69 affected policy/recovery/regression tests passed after that fix, including
the new regression, following the earlier 195-test integration run.

Next protocol decision: run a small matched 6,000-versus-12,000-token budget
study before changing the default. If the budget changes, regenerate both arms
under that protocol; do not pool historical 6,000-token answers with new ones.
Neither smoke nor inter-judge agreement establishes human-calibrated accuracy.

Follow-up completed: [the bounded budget study](arena_hard_budget_study_2026-09-29.md)
found 8/8 stopped, nonempty finals at 12k versus 4/8 stopped and three empty
finals at 6k. It recommends 12k for a new matched protocol while retaining
correctness and trajectory-divergence limitations. The default is unchanged.
