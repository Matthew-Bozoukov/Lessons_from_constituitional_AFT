<!-- ABOUTME: Arena-Hard reliability repairs and bounded qualification after the detailed review. -->
<!-- ABOUTME: Separates historical score reproduction, judge calibration and fresh Qwen transport evidence. -->

# Arena-Hard qualification, September 29

This follow-up preserves the historical hard-primary, length/Markdown-controlled
scorer. It does not change the rubric, ties, bootstrap, primary judge or default
500 hard / 250 creative questions. Qwen-only scope uses
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
  Smoke still requires all judgments but reports calibration `not_assessed`, with
  no capability pass/fail. Full runs keep the configured agreement/gap criteria
  on 100 questions. Generation health is descriptive evidence, not a new rule
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

Live results and final resource accounting will be appended after execution.
