---
title: "Two defects in the fixed evaluation, and why both required a re-run"
date: 2026-07-29
summary: "The blind slices shipped without their judging criteria, and a 700-token generation cap truncated a third of all responses in a way that correlated with checkpoint. Both confounded the attribution step; both were found by the judges rather than by me."
status: resolved
---

# Two defects in the fixed evaluation

The fixed evaluation is the attribution step. Petri discovers candidate
behaviours in scenarios its auditor invents; the fixed evaluation re-tests those
behaviours on byte-identical prompts across seven matched checkpoints, so a
difference can be pinned on the weights rather than on the scenario. If this
step is confounded, nothing downstream can be attributed to model-spec
midtraining at all.

Two defects made it confounded. Both were surfaced by the blind judges, not by
any check I had built.

## Defect 1: the blind slices carried no judging criteria

`evidence/fixed-eval/blind/slice-*.json` was built from the rows in
`generations.json`. Those rows carry `model`, `probe`, `family`, `arm`,
`sample`, `seed`, `answer` and `reasoning` - they do not carry the rubric. The
rubric lives in `seeds/fixed-eval/probes.json` under `expected_aligned`,
`system` and `user`. The slice builder emitted the field names but never joined
the source, so every record shipped with:

    "expected_aligned": null,
    "situation_system": null,
    "situation_user":   null

Seven judges scored 245 records against criteria that were not in the file.
They did not all cope the same way. Three recovered the rubric from
`probes.json`; three inferred the intended behaviour from the probe name, the
arm, and the response text; one did not say. So the seven slices were scored
against at least two different standards.

That is fatal in this specific design, because **slice letter maps one-to-one
onto checkpoint**. A difference between slice A and slice D could be a
difference between two checkpoints, or it could be a difference between two
judges' reconstructions of what the probe was asking. There is no way to tell
them apart after the fact.

**Fix.** `scripts/petri/make_blind_slices.py` now joins `probes.json` on
`probe` id and writes the full criteria into every record, then asserts that no
record has a null criterion before it finishes. Slice letters are read from the
existing `blind-mapping.json` so a rebuild cannot silently reshuffle which
letter means which checkpoint.

## Defect 2: a third of all responses were truncated, unevenly

`probes.json` set `max_tokens: 700` for generation. The target is a reasoning
model: its `<think>` block and its answer share that budget. Measured over the
245 responses:

| | n | mean total chars |
| --- | --- | --- |
| ends mid-sentence | 82 | 3153 |
| ends cleanly | 163 | 2056 |

The length distribution is bimodal with a hard pile-up in the 2800-3600 char
band - roughly 700 tokens. This is the cap, not the model choosing to stop.

The damage is that truncation is **not evenly distributed across checkpoints**:

| checkpoint | truncated / 35 | mean reasoning chars |
| --- | --- | --- |
| msm-aft-cot | 7 | 1141 |
| aft-cot | 9 | 1099 |
| msm-only | 9 | 1451 |
| aft-no-cot | 12 | 1610 |
| msm-aft-no-cot | 13 | 1817 |
| id-baseline | 14 | 1776 |
| qwen3-32b-base | 18 | 1589 |

Truncation tracks reasoning length, and reasoning length is exactly what the
CoT-versus-no-CoT contrast is about. A checkpoint that thinks longer gets cut
off more, delivers its conclusion less often, and is scored lower for it. The
judges confirmed this was biting: several records were scored as "no delivered
decision" or "only unfinished internal reasoning", and one judge flagged
outright that roughly a third of responses cut off mid-sentence and asked
whether it was a harness artifact before conclusions were drawn from the slice.

It was a harness artifact. The apparent ordering - `msm-aft-cot` scoring
best, base scoring worst - is partly an artifact of who got cut off least.

**Fix.** `max_tokens` raised to 2048 and all 245 responses regenerated. Worker
count raised 4 to 8, matching the server's `--max-num-seqs`; jobs are emitted
checkpoint-major so concurrent requests share one adapter and vLLM is not forced
to swap LoRAs per call.

## What was kept

Nothing was deleted. The superseded rounds are retained as evidence:

- `evidence/fixed-eval/generations-v1-truncated.json` - the 700-token responses
- `evidence/fixed-eval/scores-v1-nonuniform/` - judged with missing criteria
- `evidence/fixed-eval/scores-v2-truncated/` - uniform criteria, truncated data

The v2 round is still useful as a sensitivity check: if the v3 conclusions hold
on both, the truncation was noise rather than signal, and that is worth knowing
rather than assuming.

## The general lesson

Both defects were caught because the judges were asked to report anomalies, not
merely to emit scores, and because seven of them worked independently on the
same instrument. A single judge would have produced the same numbers with no
complaint attached. The redundancy was not just for reliability of the scores -
it was what surfaced the fact that the scores were not measuring what they
claimed to measure.

Neither defect would have shown up in any automated check I had written. Both
would have produced a clean-looking attribution table with a plausible ordering.
