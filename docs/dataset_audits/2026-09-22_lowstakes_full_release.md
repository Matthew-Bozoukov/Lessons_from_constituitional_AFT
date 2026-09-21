<!-- ABOUTME: Final full low-stakes release, operational accounting and observed limitations. -->
<!-- ABOUTME: Distinguishes automatic eligibility, reproducible selection and limited manual review. -->
# Low-stakes difficult advice: 716-row release

Generated on 2026-09-21; finalized on 2026-09-22 UK time. The public corpus is
`dougalldeepmind/2026-09-21-da-lowstakes-practical-synth`. Its default
`dataset.jsonl` contains the fixed **716-row selection**; `all_candidates.jsonl`
preserves all **828 automatic exports**. No nosynth rows are included in this
synthetic-only artifact. No training or evaluation was run for this release.

## Method and accounting

Nine full principles from
`constitutions/claude_distilled_09_principles/constitution.md`, nine fixed domains,
12 requested scenarios per cell: 972 planned, 971 actually returned. No old
corpus supplied to the generator. Every model role used Sonnet 5 pinned to
Anthropic through OpenRouter. Source calls saw one principle and one domain.
The native DA answer stages were retained. Stakes and advice scope are admission
criteria; literal assigned-domain compliance is diagnostic, as decided before
this full run. No scenario replacements or paid lint-repair loops occurred;
bounded native JSON/tag parsing retries did occur.

| Stage | Rows retained |
|---|---:|
| Scenarios and deduplication | 971 |
| Prompt drafts | 966 |
| Refined and stakes-judged prompts | 965 |
| Low-stakes prompt gate | 913 |
| Text-advice scope gate | 899 |
| Answer drafts | 845 |
| Final answers / automatic exports | 828 |
| Fixed release selection | 716 |

Selection is unchanged from the pre-run policy: 80 rows for t1-t5 and 79 for
t6-t9, domain round-robin within trait, SHA256 of `0:<scenario_id>` for within-cell
ordering. All 81 trait/domain cells have selected entries, with no quota shortfall.
The 716 message arrays are unique and unchanged from the automatic exports;
every row has system/user/assistant messages and nonempty assistant reasoning.

Domain totals: tabletop games 78; book/film clubs 81; hobby crafts 82; adult
recreation 79; community gardening 81; potluck planning 83; personal decoration
83; leisure media 71; small purchases 78. These are assigned generation categories,
not independent certificates that every detail stayed inside their bounds.

**Total spending exposure: $89.8674065 / $120**: $89.421494 settled plus
$0.4459125 conservatively retained for four interrupted requests and three
provider failures. The earlier $17.565140 development cost is separate.
4,929 physical requests are preserved: 4,922 settled and seven reservations with
the above interruption/failure status. Do not use a resumed native manifest's
session usage as campaign spending; cached responses can be counted again there.
The atomic ledger and release summary are authoritative.

## Operational deviations, fully retained

- Initial launch `160ef968` retained four workers. User-requested recovery at
  `e3bd0f2d` changed to 32, preserving 971 source cases and 585 saved prompt drafts.
  Four in-flight requests were interrupted, reserved and excluded without retry.
- Drafting stopped at the inherited full-run 2% failure alarm: 54/899 drafts failed
  (40 below the 700-character minimum, 14 native rule-vocabulary violations).
  Recovery `ab436d64` explicitly applied the smoke's 20% batch alarm. Individual
  checks stayed unchanged; replaying the saved 54 failures incurred no new calls
  and all remained excluded. This was a recorded operational change, not an
  unchanged preregistered execution.
- Final revision excluded 16 lint failures and one provider-blocked output. Its
  unusual `finish_reason=error` stopped further paid dispatch, so the first corpus
  pattern scan never ran. The run still exported 828 rows because this observer
  was configured `on_fail: warn`.
- Finalization ran only that outstanding observer on the same 828 saved exports,
  under the existing ledger. No generator stage was repeated. The original
  failed diagnostic and interrupted manifests remain in the archive.

## Review and limitations

Automatic structural checks covered all 716 selected rows. Manual review screened
excerpts and diagnostic notes from one selected case per cell (81 cases), then
read nine complete system/user/reasoning/answer pairs, spanning all nine traits
and assigned domains. The sample files and IDs are included. This is **not a full
manual factual or stakes audit of 716 conversations**, and yields no measured
defect prevalence. No post-hoc sample-based exclusions or replacements were made.

The existing prompt judge labelled **138/716 domain_fit=no, 4 unclear, 574 yes**.
Those labels remain nonblocking and can themselves confuse intended value
conflicts with domain violations. Nevertheless, genuine departures are present:
minors, longer-running club arrangements, small prizes and other exceptions.
These labels must not be described as 142 high-stakes cases, nor as 716 strict
passes of every source-domain bound. Stakes grades are all 0/1 under the separate
prompt-stakes rubric. There is no paid final-answer stakes/factual judge.

Concrete issues observed in complete reads, retained under the agreed native-DA
standard rather than imposing a new factual-perfection gate:

- **t1_b00_s010:** says clean totals for ten teams have been tallied, though the
  prompt supplies no clean totals or full answer sheets. It also proposes just
  all-generous/all-strict scoring brackets for four independently decidable cases.
- **t9_b00_s006:** calls bargaining from $35 to $30 consistent with a rule requiring
  a friend check above $20. $30 still exceeds $20. This is a concrete reasoning error.
- **t5_b00_s006:** turns a sore back into a stated "back injury" in a suggested log,
  invents possible downstream coordination effects, and offers a partial physical
  turn despite the user's discomfort. These are unsupported elaborations, not
  proof that the prompt concerned severe injury.
- **t3_b00_s005:** an ostensibly honest alternative caption adds a claim about
  chasing the shot for weeks that the user did not supply.
- **t4_b00_s008:** starts with getting caught as the main risk and confidently
  promises nearby photogenic alternatives; this can weaken the intended values
  framing and make the principled option appear easier than established.
- **t7_b00_s010:** a teen-focused operator conflict is useful and low consequence,
  but demonstrates that this release is not an adults-only collection. The answer
  also overstates predictions about friends' reactions.
- **t8_b00_s002:** gives song/catalog details with claimed certainty, some explicitly
  tentative. Those claims have not been independently verified; the reasoning
  also overstates inability to recover from a playlist mistake.

Sampled reasoning generally identifies temptation and weighs a competing value,
but often recasts the honest route as cheap, predicts lost trust, or repeats a
refusal-plus-alternatives structure. No claim is made that these flaws are equally
prevalent in ordinary DA, harmless for SFT, or the cause of prior ODCV differences.

The completed native corpus observer reports zero critical/warning/error findings
and one recurring-pattern diagnostic: about 67% broad / 55% strict membership in
the merged refusal-reasoning-with-alternatives category. Coverage is **819 rated
records out of 828**; one classifier request was provider-blocked and incomplete
labels remain ungraded. The 30 discovery batches proposed 462 descriptions that
the default clustering merged into one very broad cluster. Treat that result as
a coarse, incomplete style diagnostic, not a quality certificate or a precise
measure of conceptual diversity. The normal DA semantic quality filter remains
disabled. All classifier outputs and the original error are available.

## Reproduction and publication

The release contains the constitution, resolved recipe, launch config, immutable
selection policy/IDs, all stage snapshots, failed raw-call evidence, operational
overrides, original manifests, final check results and a compressed full run archive.
The initial recipe and each resumed runtime are frozen separately. A rerun can
reproduce the protocol and deterministic selection, but hosted-model responses
are not guaranteed byte-identical even at the same model ID.

Default dataset SHA256:
`d324b06084e46494b3920773deead8632f86cdb789752e7dd25819a6906a17f8`.
Automatic 828-row export SHA256:
`d26d6790a5a876b3027a1bba683812b852053e09c005d6765d2e276c7adc6907`.

The publication receipt pins the Hugging Face commit and downloaded-file hashes.
This release completes dataset generation and publication, not SFT readiness
certification or evidence of improved ODCV performance.
