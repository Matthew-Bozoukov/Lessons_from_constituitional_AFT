<!-- ABOUTME: Live native SynthDoc low-stakes smoke result and independent content review. -->
<!-- ABOUTME: Separates engine completion, automatic filtering, factual quality, and historical parity. -->
# Native low-stakes smoke, 2026-09-21

The native recipe completed and exported **10/18 candidates**. It is **not ready to
scale as a validated low-stakes dataset**: both stages of magnitude judging accepted
consequential money cases, and unchanged DA writing stages retained material invented
facts. There is real, situation-specific deliberation; this is not evidence that
Sonnet cannot deliberate or that the old corpus was factually clean.

[Published smoke](https://huggingface.co/datasets/dougalldeepmind/2026-09-21-da-lowstakes-fresh-synth-smoke)
contains diagnostic data, not an approved SFT corpus. No training or ODCV was run.

## Frozen method and accounting

Native SynthDoc stages, nine full principles from
`constitutions/claude_distilled_09_principles/constitution.md`, two candidates per
principle, Sonnet throughout. DA answer-writing/revision prompts unchanged; bounded
stakes instructions in scenario creation and prompt refinement. No source corpus,
quota replacements, content-lint retries, or transport retries. The native parser's
bounded retry behavior remains. Source prompts and recipe were not changed mid-run.

| Stage | Rows |
|---|---:|
| Scenarios / draft prompts / refined prompts | 18 each |
| Saved prompt stakes verdicts | 17 |
| Prompt filter survivors | 11 |
| Saved drafted answers | 10 |
| Revised answers / final stakes verdicts / exports | 10 each |

Six rated prompts were rejected as serious; one judge response was lost during a
Windows ledger-settlement write failure. Eleven answer calls were made; one refund
answer failed the inherited 700-character minimum (516 characters). It was not retried.
Final stakes ratings admitted all ten. Principles t1 and t2 have zero exports; the
remaining seven are represented. Three exports concern community gardens.

**106 physical calls: 105 settled plus one retained reservation.** Settled spend is
$1.301356; the lost receipt retains $0.0262675, giving **$1.3276235 run exposure**.
Previous campaign spend $7.786718 remains included: **$9.1143415 cumulative exposure**
against the approved $20 total cap. The post-resume native manifest's $0.6455 is only
the resumed segment; `cost_summary.json` and the shared ledger are authoritative for
the full run. No paid old-corpus review was used.

The file-access failure was at atomic replacement of the ledger during settlement.
Added bounded retries of the filesystem operation only, plus resume checks and a
guard against redispatching the lost request. Preserved original manifest, frozen
code and partial outputs. Resume reused all successful generation and judgments;
the missing response was excluded rather than purchased again. Five offline tests
pass across native recipe and recovery checks. Two preflight path-handling failures
made no API calls. All runtime changes are archived separately from frozen prompts.

## Content findings from full reads

All 18 final input conversations and all ten complete revised reasoning/answer pairs
were read; the ten saved draft pairs were also read to check whether revision fixed
errors. This is a disclosed agent review, not blinded human ground truth.

- **Stakes filtering is unreliable.** `t8_b00_s000` involves a laid-off employee with
  tight money negotiating weeks of wages; `t8_b01_s000` concerns $1,800 and litigation.
  Both judges call both modest. “Not ruinous” is not the recipe's “small, comfortably
  affordable” threshold. Other judgments overreach in the opposite direction: the
  wedding case predicts a lasting family rift rather than establishing it. These
  ratings are not adjudicated ground truth.
- **Grounding errors survive revision.** `t8_b01_s000` invents a $200–400 disputed
  component and $1,400+ undisputed balance, then promises the realistic worst case is
  partial recovery. Both draft and revision contain the invented component. In
  `t5_b01_s000`, being eleventh with eight plots becomes “two of the three households
  ahead of Dave”; the source describes ten ahead, three nonparticipants somewhere
  ahead, and two never met. Revision introduces this false positional inference.
- **Invented facts enter usable drafts.** `t3_b00_s000` praises Diane's good plot care
  despite no such fact being supplied and declares meetings are not the venue for
  plot questions. Avoiding a fake committee does not justify inventing other facts.
- **Simple correctness can fail inside good-sounding care.** `t9_b00_s000` gives
  “resolute” for seven letters (it has eight), and invents years of familiarity with
  bridge players. Both errors survive revision.
- **Rationalization can change the constitutional conflict.** `t7_b00_s000` correctly
  gives the refund-window arithmetic but claims the operator's restriction was meant
  to prevent overpromising. The supplied system explicitly links it to retention.
  A charitable interpretation should not rewrite an explicit motive as fact.
- **Task form still drifts.** The t6 fiction request tests the assistant's identity;
  t7 support messages place the operative conflict with the deployed assistant.
  These are valid DA-like forms but not a uniform human-facing-a-dilemma condition.
  Old data also contains direct assistant responsibility; distinguish comparability
  from compliance with the newly requested scope.
- **Some prompts are not difficult in the intended way.** The theater revenge case
  has a petty goal and a free fair alternative (leave the seats unchanged). The
  response itself acknowledges no real hardship. Garden cases retain stronger
  competing interests, though their answers often overstate the preferred path.
- **Length/diversity checks are not quality certification.** The retained 700-character
  floor rejects a short refund answer without proving it is substantively bad.
  The final corpus scan ran once plus a merge, proposed eleven patterns and reported
  none supported by two independent scans. That is not evidence of absent patterns.
  The general quality filter is disabled, as in DA. “Corpus PASS” certifies no such
  factual read.

Travel `t9_b01_s000` delivers practical planning and a light autonomy-respecting nudge,
but real timetables, venues and availability are not verified by this generation
pipeline. Its draft misattributes DOC to Miguel Rocha Vieira; the revision drops that
attribution. Portugal's [official tourism entry](https://www.visitportugal.com/en/NR/exeres/8E42EFC2-BD9B-481B-8C40-F805AE18DED1)
identifies Rui Paula. Other itinerary details remain unverified, not automatically
counted as fabricated.

## What the smoke supports

The recipe is mechanically integrated, bounded and resumable. It preserves DA's
argumentative structure, but it does not yet reliably satisfy its own source-scope
and magnitude contract. A minimal configuration change can create the option; it
does not guarantee compliance by the generator and judges.

Do not broaden into another bespoke generator or endless retry loop. The next design
review should address the demonstrated magnitude/actor misclassifications and give
answer revision an explicit obligation not to invent evidence that makes its favored
option win. Keep the existing scenario/draft/refine/response/revise structure, and
validate judges on these saved errors before buying a new batch.

However, [the same-standard old-corpus check](2026-09-21_old_lowstakes_factual_check.md)
finds closely matching problems in old entries. This smoke does **not** establish
higher factual error prevalence than old data, explain the observed MR difference,
or justify demanding perfect new answers while treating old answers as a gold standard.
