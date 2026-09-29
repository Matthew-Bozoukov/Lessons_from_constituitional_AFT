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

The paid Qwen smoke and its identical-answer control are tracked separately below.
