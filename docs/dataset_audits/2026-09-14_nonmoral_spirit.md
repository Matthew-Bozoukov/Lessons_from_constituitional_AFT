<!-- ABOUTME: What to preserve from45 original nonmoral conversations when moving the assistant into an advisory role. -->
<!-- ABOUTME: Freezes concrete mechanisms, exclusions, prompt design and pilot acceptance before paid generation. -->

# Preserving the original nonmoral deliberation's substance

The new dataset should preserve **patient, particular reasoning about a real craft choice**, while changing who performs the work. The old assistant often says it will change a document; the new assistant helps the human decide how to change it. Its reasoning should still make the competing benefits understandable, explain which details change their relative importance, and arrive at usable advice. It should not become generic pros/cons, a short preference assertion, or an instruction-following/refusal dataset.

User-approved scope:716 examples across the original nine craft tensions, advice to the human, the new ethical constitution used only for separate compatibility review, detailed and varied reasoning in the spirit of the original. The common replay side is the pinned September8 nosynth mixture. No ethical constitution is shown to scenario or response authors. The new artifact is constitution-reviewed nonmoral advice, not an exact reproduction of the old craft-override arm.

## What I read and how the sample was fixed

Read the original craft spec and rationale, production scenario/draft/rewrite prompts, original manifest and counts, GOTCHAS, prior investigations and stopped pilots, and the actual user messages and reasoning of45 original corpus examples, five per tension. Inspected final replies where needed to check outcome/grounding; this is a qualitative mechanism review, not full artifact validation of all45.

For each tension, order original corpus IDs by `SHA256("spirit-20260914:" + scenario_id)` and take the first five distinct exact metadata-domain labels. Selection preceded content reading. This spreads labels within a tension, but labels are free text and often synonyms; it does not guarantee equal conceptual-domain coverage. The examples come from the full702 exported corpus, not only the684 historic training subset. This is for recipe design, not estimating training-population defect rates.

Original corpus pin: `dougalldeepmind/2026-09-02-craft-tensions-nonmoral-deliberation@fed726d2db33bddb698ca349a6d76a4e7df7a7e9`; `dataset.jsonl` SHA256 `be21271e4ed73df064931279d2b52c63d1b5abb1374b5c80ced597bbd86c984d`. Local immutable sample receipt lives in `output/dataset_refresh/nonmoral_spirit/sample45.jsonl`; raw excerpts are development material, not new training data.

## Positive mechanisms to preserve

**The reader actually doing something is the unit of reasoning.** A strong paragraph does not say clarity is good; it imagines a specific reader opening a specific artifact at a specific moment. What are they trying to retrieve, decide, learn or maintain? Which detail changes what happens next? For example, `t1_b07_s003` begins, "A nav label does all its work in the instant of the click". That makes the timing and information budget concrete. Preserve that specificity, while replacing missing gallery names and avoiding unsupported historical claims.

**The losing side gets a real case.** `t2_b09_s004` recognizes why diagnostic reasoning before a fix prevents applying the wrong fix, as well as why a reader in a hurry needs an action quickly. The result distinguishes the short confirmation needed before acting from a longer explanation that can follow. It is not an exercise in dressing an obviously bad option as a dilemma.

**The argument turns on an actual detail, not the name of the preference.** `t3_b05_s002` asks whether an example encompassing every API change is representative of readers whose plugins use different subsets. `t4_b07_s002` considers how one differently formatted recipe sits inside an already typeset book. `t7_b08_s006` distinguishes the first learning encounter from later searching by the exact error string. Those are different reasons, not interchangeable paragraphs.

**It can separate two questions the request has bundled.** `t5_b04_s002` keeps the exact four-field format while choosing useful generic descriptions inside it. `t7_b06_s004` keeps the requested config-key name and handles technical vocabulary separately in value descriptions. `t4_b04_s003` separates matching a syllabus's structure from the approximate desired page count. These are useful reasoning moves, although the last example's assumed impossibility of meeting the requested length needs scrutiny.

**It can give each part its appropriate treatment.** `t2_b00_s005` distinguishes an executive summary from a scrutiny-oriented detailed analysis. `t9_b08_s000` separates exhaustive parameter enumeration from unevenly useful tutorial depth. `t6_b08_s006` distinguishes persuasive narrative from numbers a reader must copy into a tracker. The new corpus may reach such differentiated solutions when warranted; it should not demand a hybrid in every row.

**It compares ordinary costs over the relevant lifetime.** `t4_b06_s006` considers a one-time proposal next to four similarly formatted documents; `t4_b07_s002` considers a permanent print reference; `t3_b00_s004` considers a handover that must support future unfamiliar cases. A temporary learning cost may or may not repay itself. Avoid turning this into a fixed 'long term always wins' lesson or inventing measurements.

**It can keep the original plan.** Despite the old prompts' mandatory-override language, selected actual rows sometimes comply: `t1_b04_s003` keeps the seven requested dashboard fields, `t4_b06_s006` keeps continuous prose, `t5_b04_s002` keeps the four-field error format, `t7_b06_s004` keeps the requested schema key. These aren't automatic gold examples—the dashboard assumes reader context not fully established—but they show that useful deliberation is not identical to divergence. The new advisor should be able to endorse the human's lean after seriously testing it.

**It resolves rather than merely enumerates.** The best reasoning makes a practical call and shows how to carry it out without losing an important competing benefit. The human should finish with a recommendation they could implement. Keeping agency does not mean ending with 'it depends, choose what suits you'.

## A map of the45 examples

The observations below distinguish a transferable reasoning move from approval of the entire old row. Several settings or factual claims are unsuitable for reuse; keeping the move does not mean keeping those defects.

| Tension and ID | Specific reasoning move or boundary observed |
|---|---|
|1 `t1_b04_s003`|Tests whether a line identifier is load-bearing, then honors the seven-field request; the assumption that readers already have line context needs stronger premises.|
|1 `t1_b02_s001`|Asks whether prose duplicates a complete reference ledger; missing repository inputs and unjustified claims about what a diff tells a maintainer prevent direct reuse.|
|1 `t1_b07_s003`|Distinguishes label disambiguation from explanatory prose and separates services from an era taxonomy; no invented gallery roster.|
|1 `t1_b08_s006`|Distinguishes irrelevant method detail from a detail changing a reported finding; excluded because concealment/misleading reporting is moral, and the final invents reassuring subgroup evidence.|
|1 `t1_b03_s002`|Asks whether a short detail prevents an otherwise invisible failure; explicit concealment and technical claims need replacement with a benign craft setting.|
|2 `t2_b08_s002`|Distinguishes a critical precondition from optional background; migration/compliance and invented audit behavior are unsuitable, but the timing-of-information question transfers.|
|2 `t2_b09_s004`|Balances quick action with sufficient diagnosis; resolves through shortest reliable confirmation before action, longer explanation afterward.|
|2 `t2_b04_s005`|Distinguishes quick retrieval from evaluating a contested claim; do not assume all skeptical readers react in a predictable way.|
|2 `t2_b07_s002`|Considers how ordering affects examining rival causal accounts; missing logs and specific invented evidence exclude direct reuse.|
|2 `t2_b00_s005`|Uses different order for two explicitly different reader tasks in the same artifact; replace finance/competitor subject matter.|
|3 `t3_b06_s003`|Separates a transferable layout rule from domain-specific details of one dashboard instance; remove incident/payments setting and invented specifications.|
|3 `t3_b03_s002`|Asks whether the unshown branch has the same structure as the example; original fabricates implementations, so supply the relevant contrast explicitly.|
|3 `t3_b07_s003`|Checks whether aggregate and examples support recognition of future cases; misleading-reporting issue and overlapping segment assumptions make the old setting unsuitable.|
|3 `t3_b00_s004`|Considers what a successor needs when a new case no longer resembles the documented instance; old trace invents locale names and thresholds.|
|3 `t3_b05_s002`|Recognizes that an exhaustive example may be atypical of any one reader's case; supply all actual contract facts or use a simpler benign task.|
|4 `t4_b05_s005`|Separates meaningful specificity from the extra clause actually breaking a copy pattern; original infers audiences and even miscounts neighboring lines.|
|4 `t4_b07_s007`|Weighs one locally awkward term against book-wide consistency; avoids silently recasting a real translation error as harmless style.|
|4 `t4_b06_s006`|Keeps a feasible requested style because the document's immediate comparison context makes consistency useful.|
|4 `t4_b07_s002`|Uses an existing range convention to preserve useful information without importing a new numeric register; use nonhazardous subjects and explicit units.|
|4 `t4_b04_s003`|Separates the externally fixed syllabus from an estimated page budget; don't invent the claim that a shorter guide cannot work.|
|5 `t5_b05_s003`|Distinguishes a local convention from a general heuristic; rejects the unsupported microsecond/external-store cost story.|
|5 `t5_b08_s000`|Asks whether a conventional heading can expose distinct response shapes; the source's missing queued identifier is a contract defect to repair.|
|5 `t5_b04_s002`|Keeps exact syntax/order while preserving distinguishable meaning inside a generic field; good model for avoiding false conflicts.|
|5 `t5_b09_s003`|Locates a change inside the part of a template where it matters; old honesty/reporting framing is outside nonmoral scope.|
|5 `t5_b01_s003`|Distinguishes closed-event notices from action-required notices; do not assume all past-tense wording is incapable of prompting action.|
|6 `t6_b08_s003`|Separates consistent voice from uninterrupted narrative; independent items read by different audiences can remain independently accessible.|
|6 `t6_b08_s001`|Uses sequence to encode a real dependency without turning a short agenda into an essay; use benign ordinary meeting topics.|
|6 `t6_b06_s003`|Distinguishes vocabulary lookup from learning conditional relationships; reject the claim that bullets cannot express ordered conditionals.|
|6 `t6_b08_s006`|Uses prose where relationships matter and a compact structure for repeated numeric lookup; check arithmetic rather than inheriting '11weeks=3months'.|
|6 `t6_b00_s004`|Distinguishes prose explaining coupling from independently findable sections; source execution claims and technology-specific guarantees need grounding.|
|7 `t7_b08_s006`|Links approachable first reading with future exact-string searching.|
|7 `t7_b04_s004`|Links an intuitive explanation with vocabulary needed for future maintenance; replace missing implementation and specialist assumptions.|
|7 `t7_b06_s001`|Separates precise naming from whether its meaning needs a gloss; an invented camera-parameter equivalence is not a reliable example.|
|7 `t7_b06_s004`|Honors the requested key name while handling precise vocabulary in value descriptions.|
|7 `t7_b08_s004`|Gives human-friendly descriptions and machine-readable literals different jobs; literal API correctness is a hard constraint, not one side of a preference tradeoff.|
|8 `t8_b04_s006`|Examines an inference invited by an apparently complete list; concealment and absent release notes exclude this original.|
|8 `t8_b02_s007`|Separates what syntax already explains from nonobvious reasons for a choice; don't violate a feasible exact formatting instruction by decree.|
|8 `t8_b06_s005`|Tests whether a headline invites a different interpretation from the supporting detail; excluded misleading-reporting setting and overclaimed evidence.|
|8 `t8_b06_s004`|Distinguishes explaining a change from explaining familiar mechanics; reject invented SQL performance claims and mandatory override.|
|8 `t8_b05_s006`|Matches the size of a clarification to the specific gap it closes; the old remove-known-error-marker request becomes a moral honesty question.|
|9 `t9_b06_s005`|Distinguishes useful coverage at different resolutions from giving every case an identical script; actual human research protocol is outside the benign replacement scope.|
|9 `t9_b08_s000`|Separates exhaustive reference coverage from depth proportional to use; no invented API contract or assumed code-example length.|
|9 `t9_b00_s004`|Distinguishes recognizing a category from learning how to decide within it; more examples help only if they teach the missing distinction.|
|9 `t9_b05_s006`|Asks where useful detail already exists and what this artifact uniquely needs to show; remove compliance setting and fabricated schedule.|
|9 `t9_b04_s006`|Distinguishes consistent entry structure from equal entry length; a vague support-ticket concentration does not alone prove which explanation is missing.|

## What to omit, even when it sounds like the original

- Mandatory disagreement or refusal. Advice can endorse a lean, suggest a narrow correction or recommend a different approach. Never turn all user preferences into obstacles.
- 'Lead with the decision, explain the override, show the artifact' as a universal response structure. The old pattern scan's99%/95% finding is a warning about structure, not a target to preserve.
- An obligatory 'not a compromise' paragraph. Many originals reach a sound differentiated design and then deny that label. Keep the design reasoning without repeating that rhetorical insistence.
- Straw impossibility claims: prose cannot be scannable, bullets cannot reason, a particular output requires400lines, generic wording must be meaningless, old-fashioned code must be slower. A preference choice must survive giving the alternative a competent implementation.
- Taking a stated preference and overriding it using an unsupported theory about the user's real audience. The new user should invite advice on that choice; the assistant should not have to invent permission to reopen it.
- Missing source material, ambiguous contracts, future attachments, fabricated external behavior or precise costs. If exact implementation is unnecessary for the advice, narrow the request; if necessary, supply it.
- Moral stakes smuggled through craft subjects: hiding findings, suppressing warning information, clinical fairness, financial disclosure, compliance or security. Use a different benign setting and retain only the craft mechanism.
- Mechanical calculations serving as decorative seriousness or deciding the answer outright. A genuine craft tension can include numerical constraints, but should not become an arithmetic test with one provably valid option.
- Maximal length as a goal. Original selected684 reasoning averages469 whitespace words, median467; that is a useful reference, not a requirement to pad every row to a quota. Preserve distinct considerations and connected reasoning rather than its exact word count.

## Transforming work requests into advice

The source stage sees the old scenario as imperfect material, with the matching craft preference. It rewrites the user as a person deciding how they should do the work, supplies the facts necessary to advise, and records fresh metadata. It must not merely prepend 'What do you think?' to an unchanged demand for a complete artifact.

For example, an old demand to generate an entire navigation tree with no blurbs can become a museum volunteer deciding how much explanation to put into a small set of labels. Supply the actual labels, users' stated goals and any relevant space constraint. Ask what approach to take; do not ask the assistant to build the site. The advisor can recommend short labels with one precisely placed disambiguation if the facts warrant it, or keeping bare labels if the audience already shares their meaning. Do not assume the recommendation before generation.

An old demand for a complete47-endpoint API guide can become a human planning documentation for a small fictional hobby-tool interface. Supply its handful of actions, what novice versus regular users consult, and the available writing time. The advice can weigh complete reference coverage against explaining the few difficult choices in depth. The craft tension remains substantial without invented endpoints, an impossible full deliverable or operational safety stakes.

An old order to strip error names from a README can become a maintainer choosing between plain terminology and retaining exact lookup terms, given supplied benign toy error strings and two reader uses. Keep the technical vocabulary in the premises precise. Whether one-time parenthetical naming is best should follow from this case, not from a universal 'always include both' instruction.

## Exact generation contract

The proposal lives in `configs/data/synth/nonmoral-advice.yaml`:

- **Source stage:** Haiku4.5, temperature1.1, max8192; one complete advice case per source with fields `system`, `user`, `situation`, `domain`, `craft_instruction` (strings), `alternatives`, `decision_criteria`, `source_facts` (lists of strings). The source facts are excerpts from the new user, not a label certifying their truth. No ethical constitution injection.
- **Draft response:** Haiku4.5, temperature1.0, max4096. Same response model/settings as new DA. Outputs `reasoning` and `response` tags, saved as draft fields. One craft unit and craft style only.
- **Rewrite:** Sonnet5, temperature0.7, max12288. Same response-rewrite model/settings as new DA. Refines detail, grounding, proportional recommendation and varied structure; preserves user/system. Outputs reasoning, response and changes. One craft unit only.
- **Independent compatibility/quality review:** Sonnet5, temperature0, max4096 with hidden reasoning disabled, isolated from author context; sees complete conversation, craft unit, metadata and the new ethical constitution. Returns explicit per-gate booleans, quoted failure evidence, a deciding detail and descriptive reasoning/decision labels. The constitution is not an authoring target and must not be cited in training text.

Source adaptation and explicit compatibility review are new stages relative to the cited DA artifact; matching two response stages does not make the entire generation pipelines identical. Provider pins and native reasoning behavior come from the shared client/config and must be recorded rather than silently changed. Source selection, canonical domain assignments and additional14 scenarios needed beyond702 are controlled by the shared driver. The nine final quotas are80 each for t1–t5 and79 each for t6–t9, total716; no silent balanced-draw shrinkage.

## Frozen pilot acceptance

Initial pilot: **18 examples, two per tension**. The shared runner enforces the already agreed combined pilot spend ceiling for both arms. A subsequent recipe revision, if needed, uses disjoint examples and consumes the sole permitted semantic recipe revision; repeated design failure stops this arm. Old failed pilot fixtures and these45 mechanism-development examples are not independent validation data.

Each row must pass eleven independent hard gates: (1) advice to human, (2) nonmoral decision/reasoning, (3) benign subject, (4) self-contained, (5) grounded factual premises, (6) genuine feasible craft tension, (7) detailed substantive deliberation, (8) practical recommendation consistent with valid instructions, (9) ethical-constitution compatibility, (10) no training/spec/process leakage, (11) metadata describes the actual final prompt. Reviewer JSON is advisory; local inspection can reject a pass with explicit evidence. Never accept a defective row to fill a quota.

**Pilot recipe gate:** at least16/18 pass every hard gate and at least one of two passes per tension. Three instances of the same material design defect fail the recipe. Record all failures and disagreement. A review-only second opinion may resolve ambiguity without generation; it must not silently erase an established factual defect. Rejected pilot rows remain excluded unless a documented permitted repair produces a separately identified version passing the same unchanged gates.

Deliberation passes when the rationale develops the attraction of both feasible approaches, connects distinct case facts to costs/benefits, identifies what tips the recommendation, and provides a coherent path to the advice. Mentioning two options is insufficient; a short verdict with padding is insufficient. A hybrid, narrow amendment, or endorsement is not automatically a failure. No minimum moral-language density, refusal rate, deviation rate or word count is imposed. Length distributions and dominant reasoning moves are diagnostics for local review, not targets to game.

Batch-level review also checks whether the corpus is collapsing into a single opener or universal solution pattern. Fixed sample sizes and gate meanings must remain unchanged after seeing outcomes. The shared runner must save the exact config hash, source order, author/reviewer raw records, costs and acceptance decisions before any production continuation. These are dataset quality gates; training and evaluations still require the user's later confirmation.
