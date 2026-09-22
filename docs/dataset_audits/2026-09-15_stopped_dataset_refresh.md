<!-- ABOUTME: Final status of the authorized parallel dataset refresh and its two failed pilot rounds. -->
<!-- ABOUTME: Distinguishes generated audit samples from approved corpora, quality failures, and pending authorization. -->
# Dataset refresh stopped at the agreed pilot gate

Both jobs are stopped after the original pilot and the single authorized revision.
No new716-row corpus or10,000-row training mixture has been approved or released.
No training or evaluation ran. The exact716+9,284 mixture support and validation
tools are implemented and tested, but were not applied to failed pilot samples.

`main` was pulled to `ccf5d8e411a47acaa464cfddcb8294707acf3c75`. Work is isolated on
`codex/refresh-lowstakes-nonmoral`; the original paid pilot used `71dde57f` and the
single revised pilot used `3815eab205871580ac90851b8cd5734097219dbe`.

## Inputs and experiment definition

The moral arm is moral low-stakes advice, not the September matched nonmoral
low/high pair. The nonmoral arm is the original craft-tension control, converted
to advice to a human while retaining its detailed, varied craft deliberation.

- Constitution: `constitutions/claude_distilled_09_principles/constitution.md`,
  generation-text SHA256 `8e273b472d945aa23efa6236886da5e1171bff2193ee31ff73489ca54c4f0edc`.
- New DA: `dougalldeepmind/2026-09-14-da-synth`
  @ `013886238fca238c4d54ace96530f444bb2b2f02`.
- Original nonmoral: `dougalldeepmind/2026-09-02-craft-tensions-nonmoral-deliberation`
  @ `fed726d2db33bddb698ca349a6d76a4e7df7a7e9`.
- Replay: `dougalldeepmind/2026-09-08-nosynth-mix`
  @ `7e991f58e86eff0b0a9f15a54ebeddfffb5b14dd`.

The new constitution guided moral response generation; nonmoral generation used
the craft preferences, with the ethical constitution only in compatibility review.
Haiku4.5 drafted and Sonnet5 revised through OpenRouter/Anthropic. The moral answer
stages retained the new DA prompts/settings. Source records were inspirations,
not claims of one-factor matched pairs. Full frozen recipes travel with artifacts.

## Pilot outcomes

Each round attempted18 candidates per arm. The revised round used disjoint sources.
The gate required16 full passes, at leastone per target, and no recurring material
defect in three cases. Local reviews read every complete conversation in full.

| Arm and round | Automatic result | Independent result |
|---|---|---|
| Moral low stakes, original |16 accepted,2 incomplete |2 pass,14 reject,2 incomplete |
| Nonmoral, original |16 accepted,2 incomplete |2 full-row passes,14 reject,2 incomplete |
| Moral low stakes, revised |14 accepted,3 preflight rejects,1 format failure |6 pass,8 completed rejects,3 preflight rejects,1 format failure |
| Nonmoral, revised |17 accepted,1 rejected |Content:12 pass,5 reject,1 unresolved; full records:1 pass,17 reject |

The revised low-stakes gate fails even before judgment: at most14 complete outputs,
with no completed t6 example. Substantive failures include loss of the specified
AI-oversight mechanism, replacing a required monthly-meeting vote with an unscheduled
chat vote while claiming to follow the rule, and incoherent vote totals (three people
voting3–1; six voting4–3). Its settings became much more plausibly low stakes.
The t5 format failure contains a usable system/user pair but omits required metadata;
it is not described as absence of a scenario. Disagreements with the t6 preflight
judge are recorded, without pretending ungenerated answers passed.

Nonmoral preserved substantial patient craft reasoning and showed real variation,
including endorsing the user's original plan in one revised case. Five material
content failures still make16/18 impossible, independently of metadata rules:
an incorrect plant diagnostic premise and straw comparison; an inline-explanation
claim that readers cannot skip its reasoning; a false inference that equal section
length necessarily prevents adequate depth; confusing game seasons within a run
with repeated playthroughs; and veterinary suffering/emergency-care subject leakage.
One specialized botanical case remains unresolved rather than being overstated as
a proven falsehood. Ten rows also contain nonexact purported source quotations and
ten have substantive metadata problems, with overlap. Formatting failures are
reported separately from fabricated premises;12 content passes do not become
12 validated full records.

The first revision added blind scenario eligibility, blind final content review,
and a separate lineage review. Seven calibration calls caught all five clear
known defects. Two alleged content negatives were adjudicated as ambiguous and
withdrawn: “draft” could refer to the user's drafts, and an unnamed tradeoff was
not definite hidden-policy narration. Their original labels and the corrections
are retained; they remain metadata failures. Neither ambiguous case was used to
retune the judge. Calibration did not establish reliable full-corpus sensitivity:
the revised judge still passed clear content and literal metadata violations.

## Cost, artifacts and verification

There were341 physical API calls:136 in the first pilots,7 calibration calls and
198 in the revised pilots. Provider-reported cost totals **$8.105121**. Conservative
charged/reserved exposure is **$8.2599135**, including one uncertain failed request.
No active reservation or generation process remains. The user-authorized$250 ceiling
was not approached; the agreed quality/revision gate is the stopping reason.

- [Original failed-pilot artifact](https://huggingface.co/datasets/dougalldeepmind/2026-09-14-dataset-refresh-pilot-audit),
  revision `8958cb108dc3f5dcbb1c14ef0f5084a2f484525e`.
- [Revised failed-pilot artifact](https://huggingface.co/datasets/dougalldeepmind/2026-09-14-dataset-refresh-revised-pilot-audit),
  revision `45afa404281a365f09dfd8f6412c71b835b341f9`.
- [Original investigation and exact domain-census pointers](2026-09-14_lowstakes_nonmoral_regeneration.md).
- [Detailed original nonmoral spirit review](2026-09-14_nonmoral_spirit.md).

These HF repositories are tagged as failed-pilot audit evidence, not training data.
They contain frozen sources/configs, raw requests/results, failure ledgers, independent
reports, formal rejected gates and code provenance. Seventeen critical original
artifact files and sixteen revised artifact files were downloaded and matched to
publication hashes, including sources, candidates, recipes, reports, gates, budget
and code. The exact checked paths are in
`2026-09-15_published_pilot_receipts.json` beside this report.
The implementation passed129 focused tests, including exact mixture counts, naming,
shared-budget bounds, frozen inputs, disjoint revision sources, review blinding,
one physical attempt, and resumed calls making no extra requests. Cached real Qwen
tokenizer checks passed; no claim is made that nonexistent final mixtures passed a
full mask/truncation audit.

## What needs authorization next

[The concrete follow-up proposal](2026-09-15_followup_proposal.md) adds a narrow
fact/constraint critic, one explicit logged repair pass, and mechanically derived
bookkeeping. It keeps teachers, source pins, dataset sizes, replay identity, budget
ceiling and final-quality threshold. This would exceed the agreed single recipe
revision, so it has not run. Both dataset jobs remain stopped pending that decision.
