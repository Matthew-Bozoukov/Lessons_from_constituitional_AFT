<!-- ABOUTME: Confirmed MMLU and Arena-Hard defects and their bounded September 29 repairs. -->
<!-- ABOUTME: Records prior Arena evidence, retained protocols and offline validation limits. -->

# MMLU and Arena-Hard cleanup, 2026-09-29

Subsequent Arena repairs, live judging evidence and the user-approved GPT-4.1
primary policy are recorded in [the qualification follow-up](arena_hard_qualification_2026-09-29.md).
The Arena section below describes the earlier cleanup pass.

## MMLU

The user selected the existing 570-question screen: 10 questions per each of 57
subjects, seed 0, five-shot demonstrations, shuffled choices and the existing pinned
test/dev snapshot. None of those settings changed.

The standard runner previously returned a normal result even when the configured
95% minimum parsing or 2% maximum truncation check failed. The historical report
printed health warnings while still assigning PASS/FAIL to accuracy comparisons.
Those are confirmed reporting defects. The runner now saves responses and diagnostic
metrics (including `valid` and `health_issues`) and raises before publication when a
threshold fails or a model request failed. An accuracy drop with healthy responses
remains a valid model result; it does not trigger an instrument-health failure.

The existing subject-clustered paired interval is retained. Its wrapper now requires
unique aligned question IDs and matching subjects, prompt hashes and answer keys;
equal row counts alone could otherwise conceal different exams. The historical report
also refuses capability pass/fail comparisons involving unhealthy arms. Its static
arm-ladder interface remains historical; the standard entrypoint still produces one
absolute accuracy result per explicitly selected target, not an automatic comparison
of unrelated checkpoints. Exact matched controls must be selected for an experiment.

Old config comments incorrectly described the released Qwen checkpoint as raw,
unposttrained and promised a particular interval width at 570 questions. The July 31
run contradicted the former; precision depends on observed subject differences for
the latter. The comments now describe the actual protocol without those claims.

## Arena-Hard: consult history before changing the report

Read the July 30/31 and September 2 entries in `docs/LOG.md` and the actual public
[July 31 manifest](https://huggingface.co/datasets/dougalldeepmind/2026-07-31-qwen36-27b-capability-eval-arena-hard/blob/9b1c2802a5b9aded7c086cdb0b7cb06ebe714229/report/manifest.json),
report, results, answers and judgments. All were read at dataset commit
`9b1c2802a5b9aded7c086cdb0b7cb06ebe714229`; producer code was
`76b637e82bac33e0b68cbbee6530d56e24326f9b`. The public listing also contains an August
9 regeneration bundle, which is not evidence of another completed comparison.

The historical comparison used Gemini 3 Flash Preview with low reasoning, against
the 10% synthetic arm because the 0% arm had an unmatched training recipe. It ran
150 hard prompts, extending the 40% arm and reference to 300. The headline was the
**style-controlled hard-prompt score**. Creative writing and GPT-4.1 validation were
explicitly omitted in that run. The self-comparison was approximately 50% with high
tie/swap consistency. This establishes historical behavior; it does not establish
that the present endpoint or a different reference has been qualified.

The newer framework pool had bypassed the existing style-control scorer and averaged
raw win rates across hard and creative prompts. This disagreed with both config and
the actual historical report. It also ranked pure wins rather than the mean preference
score with ties worth half. The fix reconnects `evaluate_arm`, the existing scorer:

- Every slice retains its raw split, raw preference score, controlled estimate,
  interval, style gap and degenerate-feature diagnostics.
- The leaderboard ranks the configured primary slice's controlled estimate; creative
  results stay separate. Only the primary slice receives a capability pass/fail gate;
  smoke comparisons receive none.
- `report_version: 2`, `metric: style_controlled_win_rate` and `primary_slice` make the
  changed leaderboard semantics explicit. Existing published artifacts are untouched.
- Both judgment orderings and the expected prompt subset are checked before scoring.
  Raw judgments are retained even if subsequent scoring fails.

The standard config still requests 500 hard plus 250 creative prompts with the same
configured primary judge. This is not the sample size of the historical staged run.
No silent judge or reference replacement was introduced.

Both capability runners also stop sending the Qwen-specific
`chat_template_kwargs.enable_thinking` request extension, which the strict Tinker
shim rejects. The framework continues to pin Qwen mode at serving time; cache
identity and raw reasoning/answer records are retained. Historical standalone
drivers require an endpoint already configured for their declared mode.

## Verification

99 focused offline tests passed across MMLU, Arena arm/pool/statistics and repair
regressions. New cases exercise failed-health evidence retention, different exams
with equal row counts, and a creative result strong enough to reverse an incorrectly
pooled ranking. The existing style-control tests remain unchanged.

Additionally, rescoring the pinned historical 20% arm's saved answers/judgments with
the retained scorer reproduced its published controlled result exactly:
49.21214444102081%, 95% interval [42.06935313929543%, 56.29683921837892%], n=148,
2,000 bootstrap draws, seed 0. This was local analysis of saved data, with no inference.
The modern completeness gate would require missing historical judgments to be repaired
for a fresh full comparison; this reproduction does not certify their completeness.

No paid inference, publication, resource changes or historical artifact rewrites were
performed. Live endpoint checks on each selected model remain outstanding.


## September 30 follow-up

The user approved Arena's 12,000-token default after the
[matched budget study](arena_hard_budget_study_2026-09-29.md). The active config
and protocol regression now require 12k; the 16,384 context, GPT-4.1 primary and
Gemini diagnostic policy remain. Existing generation-protocol checks reject
mixing saved 6k and 12k arms. Thirty-five Arena smoke/reuse tests passed.

MMLU retains the selected 570 questions, five-shot prompt and 8,192-token recipe.
A scripted runner reproduction on the Windows CP1252 locale raised
`UnicodeEncodeError` while writing a valid Unicode reasoning trace to
`raw_samples.md`, after all answers were generated. Artifact reads/writes now
specify UTF-8. The complete 55-test MMLU suite passes. Also removed a stale config
comment promising identical greedy answers across budget changes: the Arena
study demonstrated early trajectory divergence. No MMLU scientific setting or
score changed, and this follow-up did not run live MMLU inference.
