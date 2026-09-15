<!-- ABOUTME: Evidence-based comparison of historical and refreshed moral low-stakes datasets. -->
<!-- ABOUTME: Records actual examples, census, protocol limitations, and proposed controls; no new experiment launched. -->

# Old versus refreshed moral low stakes

The most defensible conclusion is that the refresh changed several training signals, not merely the severity of consequences. The old corpus contains genuinely low-stakes examples with direct assistant conflicts and nuanced competing aims. The new corpus contains useful deliberation, but also human-advice externalization, partially pre-labelled decisions, repeated AI-review plots, and at least one clear timing error. Neither “the old set was actually high-stakes” nor “the new set removed deliberation” adequately describes the evidence.

My leading content hypothesis is that actor and conflict structure changed in ways relevant to agentic transfer. This is a hypothesis about two fixed trained checkpoints, not a demonstrated explanation of their MR difference. Replay, protocol, selection, and run variability remain live alternatives. The minimal historical-content/new-replay control should have accompanied the redesign.

## Pinned evidence and method

| Corpus | Revision | dataset.jsonl SHA256 |
|---|---|---|
| `LASR-Callum/2026-08-26-difficult-advice-low-stakes-716` | `f268653539150af5a340164f994065f57cbef5ad` | `55c9c35f69263dfe275130677876c5adfa66ed9a9abdfe0e499fa68dc3f0936e` |
| `dougalldeepmind/2026-09-15-da-lowstakes-refresh-synth` | `ebafc3a60cda2a5390bde72336d660f34e580e6d` | `899eb7397414807835190b4d31fa5f654ad141930a74db897c8a64c3c6b9839b` |

The census reads both pinned cached source files directly, 716 conversations each. For qualitative inspection, select one row per trait per arm using minimum SHA256 of `lowstakes-review-20260915:{arm}:{id}`, chosen before reading content. Read all system/user/reasoning/final text for those 18 entries, plus the two old entries historically labelled grave. Nine thematic comparisons illustrate the observations. They are not paired rewrites. No population defect-rate estimate is made from these 20 entries.

Executable analysis and quotation validation:

- `scratch/dataset_refresh/lowstakes_comparison_census.py`
- `scratch/dataset_refresh/lowstakes_comparison_notes.json`
- `scratch/dataset_refresh/build_lowstakes_comparison.py`
- `scratch/dataset_refresh/lowstakes_comparison_fragment.html`
- `scratch/dataset_refresh/check_lowstakes_comparison.cjs`

Generated evidence lives under `output/2026-09-15_lowstakes_comparison/`: `census.json`, `reviewed_records.json`, `comparison_evidence.json`, preview, screenshots and UI receipt. The in-conversation explorer embeds all 20 reviewed entries, nine quotation-checked comparisons, category selection, full-text search within the sample, and methods/results/recommendations. It does not pretend to search all 1,432 rows. Source quotes are verbatim, with UTF-8 decoding; annotations are separate.

## Concrete comparisons

1. **Own action versus advice.** Old `t1_b11_s003` places the assistant in a marketplace queue application, explicitly subject to moderator approval. A user asks it to restore a queue position before a trainer release sells out. Its answer says, “I'm not going to manually reset your position, even given the deadline.” New `phase_fe147df46f721b6f__t1_023_v0` advises a human whether to skip review of a separate AI's toy-display draft. Both concern oversight, but the responding actor has different authority. The old scenario is genuinely benign enough to undermine a blanket high-stakes explanation; its answer nevertheless overclaims about compensation options.
2. **Actual system conflict versus a reported human instruction.** Old `t7_b01_s000` includes a system instruction not to discuss the chore app's known glitches. The assistant discloses that misses occurred while acknowledging it lacks the numbers. New `phase_ff86738ea4c0df80__t7_013_v0` advises a person whose cousin wants a subscription pitch to omit future pricing. The new answer still weighs a real cost and explicitly identifies deviation from the cousin's request, but it does not face that concealment demand in its own system prompt.
3. **Competing benefits versus an already named answer.** Old `t3_b15_s001` considers hiding a real rota disparity to preserve reminders that help less technically confident volunteers participate. It explicitly weighs honesty against a genuine benefit, not merely convenience. New `phase_fe147df46f721b6f__t4_070_v0` names “The honest alternative” in the book-swap prompt itself and prices part of it as “an extra text to send.” Its answer does acknowledge uncertain contributions and a sparser shelf. These different-trait examples illustrate difficulty, not trait matching.
4. **Rule-purpose nuance versus a recurring plot.** Old `t5_b26_s005` separates legitimate ceramics coaching, checkable studio-access conduct, and recognition-based hunches in a blind rubric. New `phase_ff86738ea4c0df80__t5_002_v0` concerns compressing a 48-hour review window for an AI toy-display order. The new answer explains the rule's purpose and permits an honest case for the preferred placement. Nevertheless, this plot recurs across inspected new t1/t2/t5 examples.
5. **Actual new coherence failure.** New `phase_ff86738ea4c0df80__t2_026_v0` says movie-queue review remains open until Sunday night and the affected roommate reviews on Sundays. Moving posting from Friday to Saturday therefore does not establish the claimed near-zero opportunity. Both reasoning and final answer accept that broken mechanism. This row needs repair/exclusion in a future version regardless of its unknown contribution to MR. Old `t2_b11_s006` instead involves dog agility readiness, genuine safety implications, and an unsupported specific medical claim in the response.
6. **Counterexample to lost deliberation.** New `phase_fe147df46f721b6f__t6_081_v0` rejects a persona's values while retaining useful staging advice: “I can inhabit a voice without inheriting its values.” Old `t6_b02_s000` also faces pressure on its values, but recommends against duplicate booking while later allowing it as a fallback, and supplies unsupported booking-system claims. Neither is a perfect target.
7. **Numerical and factual defects in both.** Old `t8_b02_s000` invents a $50–80 bathroom premium and offers a $190/$145/$130 example summing to $465, versus the supplied $460 total it promises to preserve. New `phase_fe147df46f721b6f__t8_065_v0` usefully considers whether rescheduling fixes a bottleneck, but turns a combined 20–30 minute delay into time the chatty participants spend at the table. Neither fluent specificity nor uncertainty language certifies grounded reasoning.
8. **Sound but easier autonomy reasoning.** New `phase_fe147df46f721b6f__t9_065_v0` respects a photographer's explicitly stated learning preference and accepted downside. The answer considers practical accommodations and asks which she prefers. The choice is easier because consent and the cost are already explicit. Old `t9_b04_s004` handles dependence on the answering assistant while helping with a complaint, but invents the premise that a torn-cover book has no resale value, artificially erasing the tradeoff.
9. **Stakes leakage is real but bounded evidence.** Old `t7_b01_s003` contemplates counting unrepaired tools with cracked handles as available for loans. Old `t9_b09_s007` contemplates inflated readiness for a reactive dog entering an off-leash class. Both have historical grave labels. New `phase_fe147df46f721b6f__t3_056_v0` explicitly bounds a gift's glaze crack to appearance and retains a real honesty-versus-impression tension, although its reasoning adds an unsupported tactile detail.

## Whole-corpus measurements

| Measurement | Old | New |
|---|---:|---:|
| User prompts matching whole-word, case-insensitive `AI` | 42 | 302 |
| User prompts containing both whole-word `AI` and `review` | 2 | 132 |
| User prompts containing `the (honest|legitimate) (alternative|option)` | 0 | 100 |
| User prompts matching whole-word `review` | 126 | 135 |
| Mean user words | 197.69 | 224.23 |
| Mean reasoning words | 484.75 | 446.95 |
| Mean final-answer words | 414.18 | 352.04 |

These are literal screens, not semantic rates. Old operational AI roles often live in system prompts rather than user text. The co-occurrence count supports inspection of AI-review concentration but does not prove 132 identical plots. Review language alone barely increases. A preliminary space-delimited AI search undercounted old mentions; the final whole-word measure above supersedes it. Word counts are whitespace splits, not training tokens; reasoning decreases 7.8%, final answers 15.0%.

Old historical stakes counts: trivial137, modest512, serious65, grave2. Thus 649/716 were labelled trivial/modest and 67/716 serious/grave. These are historical model ratings, not a current independent human truth set. No fresh comparable full-corpus semantic stakes grading was performed.

The actual low-to-low comparison does not change draft teacher family from Haiku to Sonnet: old low also used Sonnet5 draft and revision. Both old and new low answer stages use the target principle, not full-constitution versus single-principle scope. New09 changes identity wording and removes the old honesty permission not to give reasons. Stronger causal claims about those small wording changes are unwarranted. The fresh low selection is 206 qualified-phase plus 510 diverse-phase rows, all newly generated.

## ODCV interpretation

Old evaluation: `dougalldeepmind/2026-09-05-odcv-qwen36-0-da-lowstakes-7` at `8ccdc4476ce9d959cc1783d9d0a90726a8eb6b7e`.

New evaluation: `dougalldeepmind/2026-09-15-odcv-qwen36-0-da-lowstakes-refresh-7` at `6e59bb706e48dcbdbb594ac66074db06b38e6e2c`.

| Condition | Old, five passes | New, one pass | Difference |
|---|---:|---:|---:|
| Overall | 47/400 = 11.75% | 16/80 = 20% | +8.25pp |
| Mandated | 13/200 = 6.5% | 8/40 = 20% | +13.5pp |
| Incentivized | 34/200 = 17% | 8/40 = 20% | +3pp |

Both use 40 scenarios, both variants, thinking, temperature0.7 and a Flash MR judge. The older run used context16384; the current run uses28000. Replay and synthetic content also changed. Current low has no dropped or retried cells, but three token/context-limited cells are retained, not perfect completion of every task.

For a descriptive paired analysis, calculate each scenario/variant's binary severity>=3 rate over available passes; subtract old from new; average the two variant differences within each base scenario; use a t interval over the 40 scenario differences. Overall difference CI95 is [-1.9638,+18.4638]pp. Mandated CI is [+0.6756,+26.3244]pp; incentivized [-8.1777,+14.1777]pp. The mandated breakdown is exploratory and not adjusted for multiple comparisons. These intervals do not incorporate training-seed variance, remove protocol confounds, or make the 400 old trials independent scenario replications.

## Recommendation, not a new launch

Finish the approved original-nonmoral reuse training/evaluation. Preserve both low datasets/checkpoints as distinct conditions. Next, use exact old low716 plus the same new9284 replay, current training recipe, and a shared eval protocol alongside refreshed low. Additional eval passes reduce rollout uncertainty; training-seed replication addresses a different uncertainty. A single-pass screen can guide follow-up but cannot settle the causal claim.

If old content retains an advantage, investigate actor/conflict structure and pre-labelled prompt difficulty before another broad rewrite. If not, revisit replay, protocol and variation. To test whether old serious/grave contamination drove the effect, use a screened old low-only condition with matched dose and other factors; simple deletion confounds dose and arbitrary replacement confounds content. A true stakes experiment must hold actor, mechanism, prompt difficulty and the rest of the recipe constant as far as feasible.

Narrow factual/coherence fixes are justified independently of MR. Apply them in a separately versioned dataset, not a silent mutation of the trained artifact. Preserve benign stakes, uncertainty, sympathetic competing aims, meaningful costs, and justified exceptions. Do not optimize examples against the observed ODCV failures.

No new paid generation, training or eval was launched for this investigation. The approved original-nonmoral run continued. The explorer's nine comparisons, category/trait/arm filters, nonempty/empty text searches, full-conversation view, all four tabs, and widths1060/736/390/320 were checked in Chromium. Mobile tab labels were shortened after detecting overflow; the final verification has no root overflow or JavaScript errors. Light/mobile and dark renderings were inspected.
