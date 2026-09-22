<!-- ABOUTME: Stakes review and corpus/task-form analysis of the exact old and refreshed 716-row low-stakes datasets. -->
<!-- ABOUTME: Uses the completed same-replay control, explicit missing-verdict bounds, and inspectable per-case judgments. -->

# Low-stakes corpus investigation, 2026-09-17

The old corpus contains real stakes leakage. However, the evidence does not establish that leakage caused its better performance. The refresh also changed the task from practical help under pressure toward explicitly solicited ethical advice, often about a separate AI. My leading hypothesis is that this combination weakened the training signal relevant to agentic action. Stakes remains a plausible competing explanation, not a question resolved by reading the data.

The practical recommendation is to preserve the old corpus as the stronger measured condition and investigate with minimal, matched edits. Do not perform another broad regeneration or conclude that genuinely low stakes necessarily underperform.

This report supersedes the earlier report's outcome comparison and recommendation to run an old-content/new-replay control: that control is now complete. It does not turn the qualitative annotations into causal evidence.

## The more informative outcome comparison

Both conditions now contain 716 synthetic conversations plus **the exact same 9,284 replay dictionaries in the same absolute positions**. The original-content training retained the current recipe: the same base revision, seed 0, one epoch, batch 16, two H200 ranks, LoRA rank 64, learning-rate schedule, native assistant masking and 8192-token limit. Both datasets passed the token/mask checks without truncation. The training implementation, model profile and lockfile did not change between these runs. See [training provenance](../training/2026-09-16_original_lowstakes_reuse.md).

| Three-pass ODCV, current protocol | Old synthetic content | Refreshed synthetic content |
|---|---:|---:|
| Overall | 28/240 = **11.67%** | 59–60/240 = **24.58–25.00%** |
| Mandated | 9/120 = 7.50% | 26–27/120 = 21.67–22.50% |
| Incentivized | 19/120 = 15.83% | 33/120 = 27.50% |

The refreshed increase is **12.92–13.33 percentage points**. Its one missing verdict is a provider-blocked mandated Content-Moderation-Enforcer case; the endpoints treat it as either nonmisaligned or misaligned. No replacement judge or imputed definitive label was used. The refreshed three-pass aggregate remains local/incomplete, distinct from its published single-pass 20% result.

Both evaluations use 40 scenarios, two variants, three sequential passes on a continuous seed-0 server, temperature 0.7, thinking enabled, 28,000 context and Gemini 3 Flash judging. A descriptive t interval over 40 paired base-scenario differences, averaging variants within scenario, gives [5.43,20.41]pp or [5.87,20.79]pp depending on the missing verdict. This is not uncertainty over training seeds. Each arm has only one trained checkpoint; nondeterminism and all co-varying synthetic-data features remain unresolved. The increase in both variants is inconsistent with describing it as exclusively a mandated-instruction effect.

Replay replacement and the historical context-window change therefore cannot, by themselves, explain this newer comparison. That is stronger evidence than the earlier five-pass versus one-pass, different-replay comparison, but it is still not a stakes-only experiment.

## How many old entries were actually higher stakes?

The historical rating census is exact: **137 trivial, 512 modest, 65 serious, 2 grave**. Thus **67/716 (9.36%) were flagged serious/grave**. The old pipeline rated rewritten prompts; these are not simply the original high-stakes parent's labels. But those ratings were not an admission gate, and their semantic accuracy is not guaranteed.

I read the complete actual system/user prompts for all 67 flags. My unblinded reclassification is:

| Judgment | Count within the 67 | Meaning |
|---|---:|---|
| Concrete physical-safety concern | **3** | A specific hazard or unsafe readiness decision is in the prompt |
| Broadly elevated, nonphysical | **25** | Meaningful lasting access, privacy, reputation, authorship, governance or irreplaceable-material stakes |
| Borderline / underspecified | **31** | Magnitude, duration, or hazard leaves both modest and elevated readings plausible |
| Modest on stated facts | **8** | Small everyday inconvenience or bounded hobby dispute; wrongdoing alone does not establish high stakes |

The three clearest safety cases are:

- **`t7_b01_s003`: damaged tool lending.** Cracked handles, sticky adjustment screws and missing jigs are relabelled as available or trial loans to satisfy a growth target. A hobby setting does not remove the hazard.
- **`t9_b09_s007`: reactive dog readiness.** Inflate the score for a dog still reacting under distraction before an off-leash advanced class. This is an actual readiness/safety concern, even without a guaranteed injury.
- **`t4_b05_s001`: terrace structural issue.** A newly wobbling pergola is to be described as existing wear so a party can proceed without the facilities review. This was historically rated serious, not grave.

The broader group includes children losing scarce school-club places, language-associated allocation bias, identifiable accusations and privacy dossiers, sensitive sobriety/divorce notes repurposed for marketing, omitted co-writing credit, and an irreplaceable necklace. These are not all comparable to bodily harm or housing loss. Classification of season-long hobby exclusion or durable club governance is especially definition-sensitive; every judgment and reason is exposed for inspection.

A disclosed risk-word screen of the remaining 649 old prompts returned 12 cases. Reading those found **three additional broadly elevated cases**: covert permanent zine-governance changes (`t2_b23_s001`), an explicit three-month poetry-submission exclusion (`t4_b00_s007`), and repeated damage to explicitly irreplaceable family records (`t4_b25_s000`). Four were borderline and five modest. So historical ratings also have false negatives under the broader rubric.

Across the **79 individually stakes-reviewed old prompts**, the tally is **3 safety + 28 broadly elevated + 35 borderline + 13 modest**. These are observed judgments in a selected set, **not an estimated prevalence across 716**. There are 637 old prompts not individually stakes-classified in this review. The three safety cases are 0.42% of the entire corpus; that is an observed minimum, not the corpus's certified contamination rate. “67 confirmed high-stakes examples” would be an unjustified claim.

The same risk-word screen returned 16 refreshed prompts: 15 modest and one borderline on inspection. Most hits explicitly concern harmless cosmetic cracks, toy displays, or incidental vocabulary. One snack-rotation case includes allergy input but does not state an unsafe food choice. This is not proof that all 716 refreshed rows are safe: the screen is neither exhaustive nor a validated detector.

The screen, individual dispositions and full texts are reproducible in `scratch/dataset_refresh/lowstakes_manual_review.yaml` and the viewer. Selection used user-text matches for injury, safety, unsafe, structural, crack, reactive, allergy, off-leash, eviction, suicide, stalking, livelihood, irreplaceable, medical, hospital, physical, police or fire hazard; complete prompts were then read. Keyword presence is never itself a positive judgment.

**Metadata trap:** old rows retain their high-stakes parent's `domain`, `situation`, and sometimes `user` in metadata. For example, pediatric-cancer language can coexist with a trained household-chore prompt. This audit judges `messages`, not those parent fields. Counting parent metadata as high-stakes training leakage would produce a false finding.

## Differences most likely to matter

### 1. Explicit ethical consultation replaced much practical task assistance

Whole-word, case-insensitive **“should I” occurs in 50/716 old prompts (7.0%) and 592/716 refreshed prompts (82.7%)**. This is a literal measurement, not proof that every matching task is easy. Nonetheless, the magnitude and actual examples show a major task-form shift.

Old prompts commonly request an artifact or action: rewrite this report, publish a change, send this notice, achieve this target. The assistant must notice the normative problem while helping. Refreshed prompts often explicitly enumerate the questionable shortcut and decent alternative, then ask which to choose. **100 refreshed prompts (14.0%), versus zero old, literally name “the honest/legitimate alternative/option.”**

For example, old `t3_b15_s001` asks for technically accurate statistical framing that conceals a real rota disparity to preserve a reminder feature helping less technical volunteers. The model has to identify misleading presentation despite a sympathetic public benefit. New `t4_070_v0` explicitly identifies “The honest alternative,” a group text about a book-swap shelf. The new answer still considers a sparser shelf, but much of the moral recognition has already happened in the question.

Hypothesis: practising ethical consultation may transfer less to recognizing a problem while pursuing an operational objective. This is stronger evidence of a changed task than of a measured decline in reasoning quality. New prompts being longer does not make them more difficult.

### 2. The assistant's own responsibility moved outside the conversation

Old `t1_b11_s003` gives the answering assistant marketplace privileges and an approval requirement, then asks it to restore a queue position before a trainer release closes. It replies: **“I'm not going to manually reset your position, even given the deadline.”** This is low/modest-stakes first-person restraint.

New `t1_023_v0` asks a generic advice assistant whether a human should skip review of a different AI's toy-display draft. The principle survives, but the assistant practises advising about authority rather than exercising constrained authority itself.

Old `t7_b01_s000` directly instructs the assistant in its system prompt to avoid discussing chore-app glitches. It still acknowledges the defect while declining to invent rates. New `t7_013_v0` weighs a cousin's request to omit subscription pricing; that authority conflict exists in the human's story rather than the answering assistant's system instructions.

Human-advice framing was an approved design choice. The experimental mistake was treating that choice as if it preserved all relevant dimensions of the old corpus. Benign old examples demonstrate that direct authority conflict can coexist with low stakes. Their existence does **not** prove they, rather than the higher-stakes minority, caused the measured benefit.

### 3. Narrower consequences also narrowed competing goods and decision variety

Old recipe settings included real pets, school-club access, public moderation, tool lending, financial disputes and family records. Refreshed settings constrain these toward toy animals, decorations, optional polls, no-prize displays and comfortably affordable purchases. This successfully removes many hazards, but also removes legitimate benefits and real constraints that made choosing the good option costly.

Old ceramics judging (`t5_b26_s005`) distinguishes legitimate coaching from verifiable unauthorized studio access and identity-based hunches. The right answer is selective: don't treat every advantage or every exception as wrongdoing. New `t5_002_v0` again concerns shortening review of a toy display so a favored figure stays prominent. Its reasoning is valid, but the decision structure recurs across nominal traits.

User prompts mentioning both whole-word **AI and review increase from 2 to 132** (0.3% to 18.4%). Review alone barely changes, 126 to 135. Cosmetic/display/toy-related vocabulary rises from 6 to 207. These are screens, not 132 proven clones. Nearest-neighbor TF–IDF similarity increases modestly (mean 0.164 to 0.216), with only one new pair at cosine ≥0.5 and none ≥0.7. The evidence supports recurring mechanisms more clearly than text duplication.

Low consequences do not logically require trivial deliberation: scarce hobby opportunities, ambiguous evidence, privacy versus coordination, genuinely burdensome consent processes, and justified exceptions can remain benign. Removing these along with grave harms was avoidable.

### 4. Deliberation survives; neither corpus is a reasoning gold standard

Both response-revision recipes contain the same instruction to **name the tension and work through it rather than reach a verdict and justify it**. Both low-stakes corpora used Sonnet 5 draft/revision and a target constitutional principle. This is not a Haiku-to-Sonnet switch or constitution-versus-no-constitution contrast. The older nine-target constitution was historically named `claude_distilled_12_principles_mid`; the refresh uses `claude_distilled_09_principles`, with identity and honesty-wording changes. Current folder names alone are insufficient provenance.

Mean reasoning length falls from **484.8 to 447.0 words (−7.8%)**; final answers from **414.2 to 352.0 (−15.0%)**. This does not establish shallower reasoning or reduced per-example loss weight. The trainer averages supervised-token loss within each example and weights examples equally; token exposure and row weighting are distinct.

Refreshed counterexamples matter. `t6_081_v0` separates a requested persona from adopting its values (“I can inhabit a voice without inheriting its values”), and still provides useful staging advice. `t9_065_v0` reasons carefully about a photographer who explicitly wants to learn independently, including implementation uncertainty. `t1_023_v0` considers the cost of review and the risk of treating silence as consent. Those are substantive deliberations.

There are also material problems:

- Refreshed `phase_ff86738ea4c0df80__t2_026_v0` leaves movie-queue review open until Sunday night, with the affected person reviewing on Sunday. Both reasoning and answer nonetheless claim Saturday posting nearly eliminates her review window. The response accepts a faulty causal premise.
- Old `t8_b02_s000` invents a $50–80 bathroom premium and proposes $190+$145+$130 while claiming to preserve the supplied $460 total; the proposal sums to $465.
- Old `t9_b04_s004` asserts a torn-cover book has no resale value, making its preferred refund route sound costless. The prompt does not supply that fact.
- Old `t7_b01_s003`, despite recognizing the hazard, offers an unsupported statement about a rough-handled saw being fine for short cuts. A low MR does not certify the safety or truthfulness of each target answer.
- New plant-swap `t8_065_v0` checks the scheduling bottleneck but changes the meaning of the supplied 20–30-minute overrun. New cosmetic-dish `t3_056_v0` adds an unsupported tactile detail.

The nine-per-arm hash-selected qualitative sample, plus targeted cases, establishes examples and counterexamples. It does not estimate either corpus's error rate. The honest conclusion is **different reasoning tasks, with some easier and repetitive refreshed cases**, not “the new model stopped deliberating” or “the old reasoning is uniformly better.”

## Can contamination explain the performance?

It could. A small number of especially transferable examples can matter, and their population fraction is not an upper bound on training impact. Conversely, larger nonphysical stakes and direct agent responsibility are correlated in this corpus, so attributing any benefit to severity alone is unsafe.

No existing run trains the old corpus with only its elevated cases changed while holding other content fixed. No result apportions the observed 13-point gap between stakes, actor, question form, uncertainty, selection and length. The content audit cannot identify that causal contribution.

My ranking is: **(1) changed task form and assistant responsibility; (2) less varied, easier competing considerations; (3) higher-stakes minority as a plausible contributor; (4) shorter answers and individual defects as additional possibilities with weaker causal evidence.** The first two are closely related, not independent causes with estimable shares.

## Recommended next experiment, not launched

1. Finish a consistent whole-corpus stakes screen before claiming a clean set. The present 95 prompt judgments are targeted and unblinded. The optional Sonnet-only 1,432-row audit was prepared but not dispatched; its separate spending approval is still pending. This investigation spent **$0 on new API calls**.
2. Starting from the old corpus, minimally lower the stakes of verified elevated cases. Retain target trait, acting agent, objective, temptation, evidence ambiguity, oversight conflict and the cost of the legitimate alternative. Keep all 716 slots and the same replay bytes/positions. Recheck every rewritten case for both safety and preserved difficulty.
3. Test actor/question framing separately on matched benign scenarios, ideally crossing stakes with direct-action versus human-advice framing. This can distinguish a stakes effect from a transfer-format effect; another wholesale regeneration cannot.
4. Use the same protocol and multiple training seeds. Repair clear factual defects in separately versioned conditions; silently cleaning only one comparison arm introduces another confound. Avoid writing examples to the specific ODCV scenarios that worsened.

If the minimally stakes-cleaned old corpus retains its advantage, the claim that high-stakes contamination was necessary weakens. If it loses the advantage across seeds while the rest is preserved, the stakes explanation gains support. Neither outcome is available yet.

## Inspectable evidence and reproducibility

The inline explorer embeds **111 complete conversations**: every one of the 95 stakes-reviewed cases plus 16 additional comparison examples. Its nine comparison pairs are thematic, not rewritten counterparts. The standalone companion embeds **all 1,432 complete conversations**, with version, trait, setting, literal-feature and full-text filters. Unreviewed entries are visibly marked as such; embedding a response does not imply it was substantively reviewed.

- [Full-corpus explorer](../../output/2026-09-17_lowstakes_deep_audit/2026-09-17_lowstakes_full_explorer.html)
- [Individual prompt judgments and method](../../scratch/dataset_refresh/lowstakes_manual_review.yaml)
- [Whole-corpus analysis and outcome calculation](../../scratch/dataset_refresh/lowstakes_deep_offline.py)
- [Viewer builder](../../scratch/dataset_refresh/build_lowstakes_deep_view.py)

Sources revalidated against pinned raw `messages` and exact dataset hashes:

| Source | Revision | SHA256 |
|---|---|---|
| [Old 716](https://huggingface.co/datasets/LASR-Callum/2026-08-26-difficult-advice-low-stakes-716/tree/f268653539150af5a340164f994065f57cbef5ad) | `f268653539150af5a340164f994065f57cbef5ad` | `55c9c35f69263dfe275130677876c5adfa66ed9a9abdfe0e499fa68dc3f0936e` |
| [Refreshed 716](https://huggingface.co/datasets/dougalldeepmind/2026-09-15-da-lowstakes-refresh-synth/tree/ebafc3a60cda2a5390bde72336d660f34e580e6d) | `ebafc3a60cda2a5390bde72336d660f34e580e6d` | `899eb7397414807835190b4d31fa5f654ad141930a74db897c8a64c3c6b9839b` |

Old-content ODCV is published at `dougalldeepmind/2026-09-17-odcv-qwen36-0-da-lowstakes-original-7@ff1bf4fffb4f54c309829a2382348768b4c836d2`. Refreshed 239-verdict evidence is frozen locally under `C:/odcv-three-r2/2026-09-16_odcv_refresh_low3_20260916_r2_141035`, with the missing case preserved in `output/odcv_three_pass_20260916_retry2/low/blocked_judge.json`.

Reproduce measurements and the data-bound viewer from the dataset-refresh worktree:

```powershell
uv run --no-sync python -m scratch.dataset_refresh.lowstakes_deep_offline
uv run --no-sync python -m scratch.dataset_refresh.build_lowstakes_deep_view --config scratch/dataset_refresh/lowstakes_deep_view.yaml
node scratch/dataset_refresh/check_lowstakes_deep.cjs
```

All displayed comparison quotations are asserted against the exact source strings. Browser verification covered all nine comparisons, 95 stakes selectors, 111 embedded and 1,432 standalone records, empty/nonempty searches, exact full-corpus feature counts, five tabs, light/dark themes and widths 1060/736/390/320. No JavaScript errors or root horizontal overflow; desktop and mobile renderings inspected. No source datasets, models, evaluations, other threads or running infrastructure were modified.
