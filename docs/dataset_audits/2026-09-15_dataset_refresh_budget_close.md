<!-- ABOUTME: Budget-close status of the parallel moral-low-stakes and original craft nonmoral refresh. -->
<!-- ABOUTME: Distinguishes retained research pools from completed716-row corpora and future training mixtures. -->

# Dataset refresh: incomplete at budget close

Paid work closed at **$249.2113677**, below the authorized hard limit of $250.
There are 11,605 physical calls in the shared ledger: 11,534 settled and 71 failed
outputs with independently verified billing. No call is running and no uncertain
cost reservation remains. API-reported charges sum to $249.2019417; the slightly
higher conservative charged figure governs the cap. Both figures include earlier
pilots, failed calls, repair attempts and reviewer probes. No further inference is
authorized by this report.

Neither requested dataset is complete. The closed retained research pools contain
**706 moral low-stakes rows** and **631 original craft-tension nonmoral rows**.
These are effective accepted rows after recorded exclusions, not a claim that an
independent reviewer has certified every answer. No new 716-row synthetic release
or 10,000-row mixture was published, and no training or evaluation ran.

| Target | Low-stakes retained | Low missing | Nonmoral retained | Nonmoral missing |
|---|---:|---:|---:|---:|
| t1:80 |78|2|73|7|
| t2:80 |80|0|69|11|
| t3:80 |80|0|70|10|
| t4:80 |78|2|72|8|
| t5:80 |80|0|68|12|
| t6:79 |78|1|67|12|
| t7:79 |75|4|74|5|
| t8:79 |78|1|71|8|
| t9:79 |79|0|67|12|
| **Total716 each** |**706**|**10**|**631**|**85**|

Those 95 missing rows are a minimum: subsequent substantive review can still find
defects. The remaining $0.7886323 cannot reliably complete both arms. No rejected
row, failed verdict, duplicate or historical Haiku draft was admitted to fill a quota.

## What was preserved and checked

The low-stakes pool combines 201 rows from the qualified Sonnet phase and 505 from
the broader low-stakes phase. The proposed per-trait limit of 28 qualified rows,
versus the initial 25, adds only two rows in the final pool. Its exact comparison
and superseded proposals are preserved. Nonmoral stays within the qualified
original nine craft tensions; it does not substitute the matched nonmoral
low/high-stakes pair.

All new authoring after the user's model instruction used Sonnet. Small independent
reviewer calibration and screening calls are recorded separately; they did not
become automatic acceptance gates. The archive also contains the earlier pilot
history, including its actual historical models, clearly separated from retained
production examples. There were no new Haiku calls after the instruction.

The exact pre-final-exclusion pools of 708 and 634 rows passed native Qwen3.6
rendering and supervision-mask checks at 8192 tokens without truncation. The final
706/631 pools retain byte-identical subsets, with explicit subset proofs and the
original audit hashes. Accepted-stage receipts and independent adoption chains
were verified. These mechanical checks establish format and mask compatibility,
not factual or scientific validity.

Every effective diverse low-stakes source was covered by the source census.
Independent full-answer reviews, targeted failure reviews and literal-process
screens found substantive defects missed by normal Sonnet critics: invented task
facts, arithmetic/constraint errors, wrong target actor, unbounded dietary or
mobility stakes, and references to the answer-editing process. Four accurate
references to visible system instructions were held under the existing ordinary
advice presentation convention; they are explicitly not labelled hidden leaks or
factual errors. Original verdicts remain unchanged beside hash-bound exclusions.

The final local similarity audits considered every pair above 0.9 and the top 10
per arm: 12 low-stakes pairs and 14 nonmoral pairs. Similarity alone never rejected
a row. Root adjudication withheld two further low-stakes scenarios and three
nonmoral scenarios, while retaining meaningful variations such as a false AI
attribution versus accepting the AI's actual suggestion, a one-page card versus
a scrollable reference, and a separate cut-or-keep editorial decision. Strong
family repetition remains documented, especially AI-review bypass in low stakes.
This is not an exhaustive guarantee of scenario uniqueness.

## Inputs and comparability

The historical investigation confirmed that moral low stakes used an ethical
constitution, whereas original nonmoral used a written craft specification.
Nonmoral was not unconditioned. Exact domain inventories and the original defects
are in [the investigation](2026-09-14_lowstakes_nonmoral_regeneration.md) and
[the craft-spirit report](2026-09-14_nonmoral_spirit.md).

The new moral author/revision stages use
`constitutions/claude_distilled_09_principles/constitution.md`, generation-text SHA
`8e273b472d945aa23efa6236886da5e1171bff2193ee31ff73489ca54c4f0edc`.
The same ethical target is used for nonmoral compatibility review; nonmoral
authoring uses the qualified craft specification, preserving the original nine
tensions while removing unwarranted empirical absolutes. Exact distinctions and
hashes are in [the qualification report](2026-09-15_craft_qualifications.md).

The new DA source is pinned at
`dougalldeepmind/2026-09-14-da-synth@013886238fca238c4d54ace96530f444bb2b2f02`.
These scenarios are unpaired mechanism inspirations, not one-factor matched
counterfactuals. The all-Sonnet author recipe, extra reviews, repairs, source
changes and selection policies are explicit differences from DA. They must not be
described as identical interventions except for stakes or morality.

There is also a measured length difference. The final cached comparison uses all
752 rows of the pinned new DA export and the incomplete 706/631 retained pools;
it is descriptive, not a matched 716 comparison. Mean supervised tokens per row
are **1175.35 for DA, 975.59 for low stakes, and 1508.73 for nonmoral** (medians
1158.5, 985 and 1496). Thus low stakes is about 17% shorter and nonmoral about 28%
longer than this DA reference. Equal row counts do not equal supervised-token
exposure. No padding, trimming or token-based selection was applied; this needs
explicit treatment in interpretation and any later training-control design.
`final_length_comparability.json` preserves exact input hashes and diagnostics.

The intended replay source remains
`dougalldeepmind/2026-09-08-nosynth-mix@7e991f58e86eff0b0a9f15a54ebeddfffb5b14dd`.
Its complete 10,000-row source pool passed the earlier native mask audit. The
implemented mixture recipe selects 716 synthetic plus exactly 9,284 replay rows
with seed 0; the two final mixtures must preserve identical replay payloads and
positions. That final selection/mixing and joint validation have not run because
the two synthetic corpora are incomplete. Nosynth itself needs no regeneration.

## Continuation requirements

Further paid work requires a budget extension. Reuse the retained rows and exact
resumable checkpoints; do not start over or clear the old ledger. First review
source scenarios for the specific missing traits before paying for answers, and
exclude overused scenario families prospectively. Preserve all nine craft tensions
and human-advice framing. Produce fresh Sonnet answers only for valid sources or
explicitly versioned repairs; technical missing-verdict retries remain distinct
from substantive adverse judgments. Keep the 09 constitution/craft roles explicit.

Then finish substantive review and exactly 716 selections per arm, build the two
10,000-row mixtures, prove the identical 9,284 replay rows and positions, and verify
the published bytes. Training and evaluation still require the user's separate
confirmation. The versioned research archive preserves incomplete and failed
work without exposing a default training split.

## Archive and reproducibility

The [incomplete research archive](https://huggingface.co/datasets/dougalldeepmind/2026-09-15-dataset-refresh-incomplete-audit/tree/f455cc9a2224d65c3861fd83a7f57c4c49c8072a)
is pinned at `f455cc9a2224d65c3861fd83a7f57c4c49c8072a`. It preserves53,813
substantive source files in13 origin/evidence archives, plus the repository source
snapshot at `2db7929fcb43d141057a77c71d88e66c1c3788ac`. The complete prepared
snapshot has24 files totaling313,697,886 bytes, and its file-manifest SHA256 is
`a10da071c9cdbf6def2717535207b762feded1cc1e5270c788dd6d75e3554012`.

The first local preparation stopped on a conservative provenance-field check;
its partial files and failure record are preserved. The successful second
preparation verified every archived byte against its original inventory. This
report corrects a transposed API-subtotal digit in the archived source report;
the original ledger, public card and hard-cap exposure are unchanged.

Remote verification confirmed exactly24 payload files plus the Hub-generated
`.gitattributes`. All14 large-file SHA256 values and sizes match the local archive
bytes; all10 small payload files were freshly downloaded without authentication
and match both local bytes and Git blob hashes. Public access and the audit-only
split were verified. The large archives were not downloaded a second time; their
content-addressed remote hashes were compared with the fully verified local bytes.
The small [publication receipt](2026-09-15_incomplete_archive_receipt.json) records
the immutable revision and verification-report hash.
