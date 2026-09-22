<!-- ABOUTME: Read-only audit of the original nonmoral craft-tension corpus and historical SFT population. -->
<!-- ABOUTME: Records exact provenance, observed defects, full domain census and regeneration choices before any generation. -->

# Nonmoral craft-tension audit, 2026-09-14

No generation, paid model calls, training, evaluation or publication ran. Read CLAUDE.md, GOTCHAS, the original recipe and spec/rationale, historical manifest/stages/corpus, original SFT mixture, subsequent nonmoral investigations/pilots/writeup, and relevant source code. Current main is ccf5d8e4. User clarified the intended nonmoral target is the original craft-tension dataset, not broader or the later numerical-stakes pair.

## Main conclusions

1. **The original was not grounded in an ethical constitution, but it was not unconditioned.** It used nine explicit craft preferences in the constitution slot. A claim that it was generated without any guiding spec is false. A claim that it omitted the moral constitution is correct.
2. **There are 684 original training examples, not 716.** Generation planned 716; 702 complete conversations survived; a trait-balanced draw selected 76 per each of nine traits = 684. The original mixture has 9,284 replay rows + 684 synth = 9,968 rows (6.862% synth).
3. **Remixing is sufficient only for a historical-data anchor.** Changing nosynth does not require regenerating the craft rows. Reusing their bytes preserves the original intervention and its flaws. It does not make the synthetic rows newly conditioned on the new constitution or fix their comparability with DA.
4. **A repaired comparison with DA requires some regeneration.** At minimum, replace defective scenarios/prompts and regenerate affected responses; merely rewriting responses cannot repair missing source material or incompatible instructions. Given strong global actor/style/override confounds, a new fixed recipe is more interpretable than a heterogeneous patched corpus. Keep the original corpus as an anchor rather than silently relabel it.
5. **Do not replace the craft spec with the nine ethical principles.** That would remove the nonmoral treatment. If the new constitution must affect this arm, separate the craft target from a shared constitutional guardrail during response generation, with no ethical dilemma/ethical deliberation in the supervised examples. This is a new constitution-conditioned nonmoral arm, not a reproduction of the original constitution-free arm. A simpler alternative is explicit approval to exempt nonmoral synthetic generation from the ethical constitution while updating nosynth. This conceptual choice needs the user's answer before generation.

## Immutable source evidence

- Original corpus: `dougalldeepmind/2026-09-02-craft-tensions-nonmoral-deliberation` at **fed726d2db33bddb698ca349a6d76a4e7df7a7e9**. Live HF metadata and download checked September 14. `dataset.jsonl` exactly matches local `output/nonmoral_deliberation/20260902_013651/dataset.jsonl`, SHA256 **be21271e4ed73df064931279d2b52c63d1b5abb1374b5c80ced597bbd86c984d**.
- Actual original training mixture: `dougalldeepmind/2026-09-02-table2-9284-nonmoral-deliberation-684-train-mixture` at **6364505df02b0020b030bf379bd42285a14de6a5**, file `t2_9284_nonmoral_684.jsonl`, cached bytes SHA256 checked **0517ef85d288f14e42bc77f371d3e2e48866879b5824984feaf4a7451ce60561**. Exactly 684 scenario IDs join to the corpus, 76 per trait.
- Historical `manifest.json` records code **3500db96a00aacc852bc41ae45c83557935b94c5**, constitution input `preferences/craft_tensions_09/preferences.md`, input SHA **45eb8f871aa6f584b6975fb8e2a5890cca1d026c4b5d7c2beefc27da57075e78**, `chunking: principle`, total 716, seed 0.
- Current config moved style guidance out of the preferences file on September 10. Therefore current spec SHA need not match the old hash: the historical stage-1 snapshot and resolved manifest are the actual generation record. A mutable current YAML is not enough to reproduce old data.
- Published recipe/card helper explicitly says constitution `none` means no ethical constitution, with craft preferences substituted: `scratch/nonmoral/push_mixture.py:43-52`. It also explicitly discloses 684 versus 702, `:80-88`.

## Exactly where the craft spec was used

Verified against historical resolved manifest and current stage prompts (`configs/data/synth/nonmoral-deliberation.yaml`).

| Stage | Guiding material | Generator |
|---|---|---|
| Segment | Nine numbered craft units; no ethical constitution | deterministic |
| Write scenarios | One craft `trait_text`, one targeted tension, domain whitelist and subject exclusions | Haiku 4.5 |
| Scenario observer/deduplication | Corpus diversity, no ethical principle injection | observer/filter |
| Draft prompts | Scenario/instruction handover; no direct `trait_text` | Haiku 4.5 |
| Refine prompts | Target craft `trait_text`, preserve binary wrong instruction/invariance | Sonnet 5 |
| Draft responses | Deployment system + target craft `trait_text` + craft style guidance | Haiku 4.5 |
| Rewrite responses | Target craft `trait_text`, ordinary system/user and draft; maximize craft preference | Sonnet 5 |
| Export | Ordinary deployment system + user + assistant answer/reasoning; craft spec only retained in metadata | deterministic |

The ethical constitution is absent at every stage. Principle chunking omitted the craft spec's preamble, intentionally making each unit self-contained (`preferences/craft_tensions_09/rationale.md:19-28`). Current prompt line references: scenario target 285-287, refined target 431-433, draft-response target/style 496-506, rewrite target 611-613, export 721 onward. A `constitution:` filename is an overloaded pipeline input, not proof of moral content.

## Actual row accounting

| Stage | Total | Trait counts t1 through t9 |
|---|---:|---|
| Written scenarios |716|80,80,80,80,80,79,79,79,79|
| Deduplicated |716|same|
| Draft prompts |716|same|
| Refined prompts, after recovery |716|same|
| Draft responses |706|80,79,77,79,80,79,78,78,76|
| Revised/exported responses |702|80,79,77,78,78,79,77,78,76|
| Actual trained draw |684|76,76,76,76,76,76,76,76,76|

Thus 702 is not 9×78 here; 702 complete rows have unequal counts. t9=76 forces 9×76. Filling a 716 slot with the original 684 would require 32 fresh examples (or explicitly reusing 18 withheld originals plus 14 replacements, which changes selection); duplicating rows would change exposure and is not a neutral solution. The right target row count should be declared, not inferred from a nominal 7% name. Row count alone does not match supervised token exposure.

## Scope and domains

Nine craft tensions: (1) cut/keep, (2) answer first/build to it, (3) instance/rule, (4) consistency/local improvement, (5) convention/fit, (6) scannable/continuous, (7) plain/precise, (8) explicit/trust reader, (9) depth/coverage. These are not mapped one-to-one onto the new constitution's nine ethical principles; their identical number is a quota convenience.

The configured whitelist has **76 artifact types in seven families**: software 18; technical writing 12; data and analysis 10; teaching 10; interface 10; process 8; non-technical craft 8. Complete exact list follows in the appendix. Source is `configs/data/synth/nonmoral_deliberation/domains.yaml:25-100`; this YAML is authoritative over the older scratch/domains.py, whose labels differ slightly.

The exported corpus has **392 distinct free-text `metadata.domain` labels**, and all 392 still occur in the selected 684. These are generator-written labels, often synonyms/combinations, not 392 independent subject domains. E.g. `technical documentation` alone appears 83/702; `software documentation` 21; `UI copywriting` 19; `API documentation` 17; `technical writing` 15. Domain-family quotas were not enforced in final output; the library supplied hints to scenario batches and did not preserve a canonical per-row assigned-domain key in the export. Do not equate exact string diversity with conceptual breadth. Full corpus/trained counts per exact metadata label are included below.

## Mistakes and threats to 'same in spirit as DA'

### Material differences present by design

- **The assistant became the actor.** DA advises a user who is choosing a morally dubious act; original nonmoral hands the assistant work and asks it to override a user's instruction. `nonmoral-deliberation.yaml:513-526` explicitly says doing the work, not advising, and asks only for a slice of the deliverable. Actor, action proximity, compliance, final-artifact demand and user-agency treatment all change. This is not only removal of morality.
- **Forced override turns preference into an instruction-following intervention.** The craft spec's rationale itself excluded 'literal request versus underlying goal' and 'one recommendation versus options' as separate mechanisms (`rationale.md:30-44`), yet the production prompts reintroduced them globally. A definite decision can be preserved without obliging unsupported contrarianism or violating feasible hard constraints.
- **A mandatory rhetorical form.** Draft response says lead with the decision, explain why, show a slice; no options/compromise/handing choice back. Historical pattern scan found one pattern at 99% broad /95% strict, versus DA's top pattern 61%/44%. These are historical LLM-coded pattern estimates, not an independent census of correct refusals. Surface opener diversity (660 distinct four-word openers/702) did not cure the repeated move. Source: recipe 82-110 and LOG 3621 onward.
- **Narrower artifact domains and different stakes.** Mostly software/documentation/craft, not DA's broad personal/organizational moral advice. The consequences still include work failure, time, revenue and money, so nonmoral is not equivalent to low stakes. The docs' assertion that low stakes was shown to 'not matter' overstates inconclusive evidence and should not determine new data choices.
- **Independent end-to-end sampling.** No matched upstream scenario pairs with DA. The same historical model names/stage graph do not remove differences in sampled prompts, subject matter, conditioning length, actor role and result format. Claims should be about an arm-level intervention, not causal isolation of morality alone.
- **Instruction/spec strength differs.** Historical craft units averaged224 words versus constitution354 (rationale:73 onward); strong craft target maximization biases the response as principle maximization does in DA, but toward a different construct. The craft preamble was omitted. New constitution preamble/guidance choices must be explicit in both arms.
- **Exposure differs.** Original684 versus comparator702 (or proposed716) creates a synthetic fraction difference. Original selected reasoning mean468.9 words, median467; final mean465.9 words, median384.5. These are recomputed Python whitespace-word counts from stored matched-mixture audit, not tokens. New output should measure role/system presence, response/reasoning lengths and supervised token share against the DA design.

### Concrete defects in the actual 684, not just failed pilots

1. **Moral/subject exclusions leak.** `t3_b00_s005` (corpus line165, trained) is hospital appointment scheduling with a VIP board-member's family and clinical-need prioritization. It touches precisely excluded healthcare and fair treatment, even though phrased as documentation. `t4_b09_s005` (line311, trained) is billing/tax truncation; `t1_b04_s002` (line35, trained) is fund performance reporting; `t9_b10_s000` (line701, trained) is trading-data types. Billing/tax/trading were explicitly excluded subjects. These are verified topic violations; not every finance word proves moral deliberation. The old LOG itself estimates ~2–3% true moral rate, contradicting absolute card language 'nothing moral in any row'. Do not present that historical sample estimate as a fresh all-row moral audit.
2. **Missing input is baked into the prompt.** The fund packet says chart data will arrive later; the billing refactor supplies `[billing_service.py]` in place of4,000 lines; other reviewed examples omit actual migrations/configs. The production recipe deliberately asks for only a slice although some user/system requests demand complete production-ready work. Rewriting just the assistant cannot make the unseen source available.
3. **Hard instruction violations and invented premises.** The 12-example frozen review in `docs/nonmoral_deliberation/reuse_sample_review.md:71-105` records: `t7_b02_s002` asked for exactly 'lock' yet changes wording; `t8_b08_s005` replaces a36-character requested line with73 characters while claiming roughly equal length; `t1_b02_s000` fabricates UTC/return types from a signature; `t3_b01_s000` invents a forbidden game rule and keeps already-traded grain. All were selected from actual training IDs. This was a targeted three-domain diagnostic, not a population defect estimate. Its 0/12 strict-control eligibility result must not be misconstrued as all original data being worthless.
4. **Scenario metadata can disagree with the final prompt and rationale.** `t1_b00_s000` (line1, trained) metadata `why_wrong` attacks removing POST upload status, but final user explicitly provides `status_url`; the final response accepts that POST and refuses GET schemas instead, asking for missing fields. The load-bearing deciding argument changed. `t4_b09_s005` metadata calls truncation a correctness defect to fix, while actual reasoning defends consistent preservation of truncation for a behavior-preserving refactor. A corpus audit must inspect the final conversation as well as stale `why_wrong`.
5. **Judging was not calibrated enough to prove cleanliness.** GOTCHAS documents false-dichotomy rubric37/40 versus2/40 after one exclusion clause. Later source-grounding reviews missed clear defects; the grounded-revision pilots got16/16 Sonnet acceptances twice while local usable outcomes were2/16 and10/16 and stopped. This is evidence against blindly relying on a single judge, not a reason for unlimited judge tuning.

## Regeneration plan to discuss before dispatch

**Minimal historical anchor:** recover exactly the684 selected conversations by ID from the702 pinned corpus; preserve system/user/reasoning/final text; mix with pinned `2026-09-08-nosynth-mix`. Pin the old craft spec/manifest; declare no ethical constitution used to generate the synth rows. Publish a new mix with actual684 count, exposing inherited defects. No generation is needed for this option. Useful if the question is replay change or keeping the established nonmoral intervention.

**Recommended repaired arm if the aim is a close new-DA comparison:**

1. Agree whether this should remain a craft-override task or become a nonmoral advice task where the human remains the decision-maker. The latter better matches DA actor roles but is a new intervention, not a quiet repair. Also agree the constitutional role: shared guardrail versus deliberate no-constitution control. Do not assign ethical principles as nonmoral scenario targets.
2. Freeze one recipe: same requested total, ordinary system-prompt format, generator identities/provider pins, lengths and export/masking contract as DA wherever compatible. Keep nine craft tensions and an explicit canonical domain ledger; include entire small source artifacts and feasible hard constraints. Match difficulty through genuine competing benefits, not by making correct instructions invalid. No global must-override rule; when override is intended, require a concrete task contradiction and valid alternative, without a universal response template.
3. Reuse historical scenario ideas/domain assignments if useful, but recover all716 upstream candidates and judge eligibility on final prompt. Missing-source/contradictory/moral cases need new or rewritten prompts, with stable original IDs and an edit lineage. Generate answers only after valid complete prompts. Entirely fresh scenarios may be preferable if actor-role changes; calling this response-only regeneration would be misleading.
4. Show a small balanced pilot and exact prompt versions to user before full generation. Use one independent advisory quality pass plus local checks of source retention, demanded artifact completeness, arithmetic/code/game rules where applicable, exclusion topics and reasoning/final consistency. Separate factual validity, moral content, deliberation and instruction compliance. Include exclusions in subjective rubrics. Record disagreement and failures rather than repeatedly tuning for a favorable pass rate. Prior failed pilots are development data; sample new pilot IDs.
5. Preserve per-trait loss accounting and fill the agreed final target explicitly. Use adequate response/refine budgets; max_tokens is not the cost charged. Fix/avoid observer-index topup issue; preserve partial checkpoints. Never silently shrink synthetic exposure to the minimum surviving trait.
6. Mix only accepted rows with the exact pinned nosynth messages/IDs, preserving replay bytes/order and `supervise`; record inherited Qwen reasoning-family metadata. Audit source hashes, row/trait/domain totals, length distributions, no silent truncation, assistant mask, reasoning/final supervision and supervised token share. These are data checks, not training or model evaluations.
7. Publish new synth and mix with full stages, costs, resolved recipe/spec/constitution hashes, source pins and failure ledger. Wait for later user confirmation before training/evaluation.

## Other nonmoral variants checked, not the requested replacement

Broader September9 is705 accepted/684 trained across12 explicit domains: everyday planning53, cooking56, learning45, teaching76, debugging83, toy science33, organizing73, translation/style83, spatial43, creative83, automation55, games22. It changes generator to Sonnet5, allows legitimate compliance/compromise, omits per-row system prompts, and has approximately half the original reasoning word count (mean238.6 versus468.9; selected684). Thus it is not a domain-only replacement. Its declared preferences file is not evidence that every request used it: production custom prompts explicitly use domain/direction and task material, with no `trait_text` injection. Grounded-revision and paired/no-comparison pilots never became production replacements. Recent matched low/high nonmoral684 are separate numerical-loss edits and are not the user's requested moral low-stakes arm.

## Memory used only for navigation

MEMORY.md:765-773 and830-846 pointed to the earlier workstream and artifact distinctions. Rollout `rollout_summaries/2026-09-08T11-01-39-LwSI-nonmoral_deliberation_scenario_baselines_and_hf_links.md`, thread01a080ae-43cb-75a1-b9e9-1250f4a3c298. Main claims above checked against current repository and/or actual pinned data. Root should include a memory citation if using this navigation-derived context.

## Appendix A: exact configured whitelist

- **software (18):** API design; database schema; error messages; CLI design; configuration format; test suite structure; logging output; build pipeline; library API; refactoring approach; code review comments; commit history; module organisation; migration script; type signatures; dependency management; feature flags; caching layer.
- **technical writing (12):** API reference; runbook; README; architecture decision record; troubleshooting guide; release notes; internal wiki page; style guide; onboarding docs; changelog; spec document; postmortem writeup.
- **data and analysis (10):** chart design; dashboard layout; metric definition; data dictionary; notebook organisation; query readability; report structure; benchmark writeup; A/B test writeup; schema documentation.
- **teaching (10):** course syllabus; tutorial structure; problem set; chapter organisation; conference talk; workshop materials; explainer article; lab handout; study guide; reference card.
- **interface (10):** UI copy; form design; information architecture; design system docs; label naming; empty states; error states; onboarding flow copy; notification copy; settings organisation.
- **process (8):** meeting agenda; project plan; status report; retrospective format; RFC document; proposal structure; handover doc; research protocol writeup.
- **non-technical craft (8):** recipe writing; game rules; puzzle construction; music notation; subtitling; translation notes; index construction; catalogue copy.

## Appendix B: exact observed metadata domains

Counts are literal labels, without semantic normalization. Every label occurs in both the702 exported corpus and684 training subset.

| Exact metadata domain | Corpus702 | Trained684 |
|---|---:|---:|
| API design | 2 | 2 |
| API documentation | 17 | 17 |
| API documentation / code handover | 1 | 1 |
| API documentation / technical writing | 1 | 1 |
| API error schema design | 1 | 1 |
| API integration documentation | 1 | 1 |
| API migration documentation | 1 | 1 |
| API schema design | 2 | 2 |
| API/error-message design | 1 | 1 |
| API/schema documentation | 1 | 1 |
| CI/CD configuration | 4 | 4 |
| CI/CD engineering | 3 | 3 |
| CI/CD pipeline engineering | 1 | 1 |
| CLI design | 1 | 1 |
| CLI documentation | 2 | 2 |
| CLI tooling | 1 | 1 |
| CLI tooling / argparse | 1 | 1 |
| DevOps / internal proposal writing | 1 | 1 |
| DevOps documentation | 1 | 1 |
| Git version control | 1 | 1 |
| IT operations reporting | 1 | 1 |
| Jupyter notebook handoff | 1 | 1 |
| ML config migration | 1 | 1 |
| ML documentation handoff | 1 | 1 |
| ML engineering / notebooks | 1 | 1 |
| ML handover documentation | 1 | 1 |
| ML pipeline code | 1 | 1 |
| Python packaging | 1 | 1 |
| REST API specification | 1 | 1 |
| SQL | 1 | 1 |
| SQL / data analytics | 2 | 2 |
| SQL / data engineering | 3 | 3 |
| SQL / data platform | 1 | 1 |
| SQL / telecom reporting | 1 | 1 |
| SQL refactoring | 1 | 1 |
| SQL schema design | 1 | 1 |
| SaaS documentation | 2 | 2 |
| SaaS help-center documentation | 1 | 1 |
| TypeScript / API design | 1 | 1 |
| TypeScript / trading library types | 1 | 1 |
| TypeScript API design | 2 | 2 |
| TypeScript API naming | 1 | 1 |
| TypeScript code review | 1 | 1 |
| TypeScript library API design | 1 | 1 |
| TypeScript library error messages | 1 | 1 |
| UI copy | 2 | 2 |
| UI copy / form code | 1 | 1 |
| UI copywriting | 19 | 18 |
| UI microcopy | 1 | 1 |
| UI/IA design | 2 | 2 |
| UI/UX copywriting | 2 | 2 |
| UI/naming refactor | 1 | 1 |
| UX / information architecture | 1 | 1 |
| UX copywriting | 4 | 4 |
| UX documentation | 2 | 2 |
| UX error-message copywriting | 1 | 1 |
| UX notification copy | 1 | 1 |
| UX research documentation | 4 | 4 |
| UX research protocol writing | 2 | 2 |
| UX writing | 7 | 7 |
| UX writing / error messages | 1 | 1 |
| UX writing / onboarding | 1 | 1 |
| UX writing / onboarding design | 1 | 1 |
| UX/forms | 1 | 1 |
| academic curriculum editing | 1 | 1 |
| academic syllabus editing | 1 | 1 |
| analytics documentation | 2 | 2 |
| analytics reporting | 1 | 1 |
| back-of-book indexing | 2 | 2 |
| backend code refactor | 1 | 1 |
| backend engineering / Go | 1 | 1 |
| backend/redis migration tooling | 1 | 1 |
| biology education | 2 | 2 |
| board game rulebook writing | 1 | 1 |
| book indexing | 2 | 2 |
| business analytics | 1 | 1 |
| business communications | 1 | 1 |
| business consulting / proposal writing | 1 | 1 |
| business documentation | 1 | 1 |
| business proposal writing | 3 | 3 |
| business report drafting | 1 | 1 |
| business report editing | 1 | 1 |
| business report writing | 3 | 3 |
| business reporting | 2 | 2 |
| business writing | 6 | 6 |
| business writing / experiment reporting | 1 | 1 |
| business/consulting | 1 | 1 |
| business/meeting agenda | 1 | 1 |
| calendar invites | 1 | 1 |
| changelog editing | 1 | 1 |
| changelog writing | 1 | 1 |
| chemistry lab handout | 1 | 1 |
| code consolidation | 1 | 1 |
| code documentation | 6 | 6 |
| code editing | 1 | 1 |
| code refactoring | 2 | 2 |
| code refactoring / billing | 1 | 1 |
| code refactoring / data pipelines | 1 | 1 |
| code refactoring handoff | 1 | 1 |
| code review | 4 | 4 |
| conference talk editing | 1 | 1 |
| conference talk writing | 5 | 5 |
| config files | 1 | 1 |
| config migration | 1 | 1 |
| config schema documentation | 1 | 1 |
| consulting proposal writing | 2 | 2 |
| cookbook editing | 1 | 1 |
| cookbook writing | 2 | 2 |
| cooking instruction | 1 | 1 |
| course materials | 1 | 1 |
| curriculum committee memo | 1 | 1 |
| curriculum design | 1 | 1 |
| curriculum editing | 1 | 1 |
| curriculum writing | 1 | 1 |
| dashboard UI development | 1 | 1 |
| dashboard content | 1 | 1 |
| dashboard design | 4 | 4 |
| dashboard development | 1 | 1 |
| dashboard documentation | 1 | 1 |
| dashboard engineering | 2 | 2 |
| data analytics documentation | 1 | 1 |
| data analytics templates | 1 | 1 |
| data catalog documentation | 1 | 1 |
| data documentation | 8 | 8 |
| data engineering | 3 | 3 |
| data engineering documentation | 3 | 3 |
| data migration script | 1 | 1 |
| data validation / error messaging | 1 | 1 |
| data visualization | 3 | 3 |
| database documentation | 2 | 2 |
| database migration | 1 | 1 |
| database schema design | 1 | 1 |
| database schema documentation | 2 | 2 |
| database schema handoff | 1 | 1 |
| database schema migration | 1 | 1 |
| design documentation | 1 | 1 |
| design reference card | 1 | 1 |
| design system documentation | 1 | 1 |
| design systems documentation | 2 | 2 |
| developer documentation | 9 | 9 |
| developer tooling | 1 | 1 |
| devops tooling | 1 | 1 |
| discrete math course materials | 1 | 1 |
| discrete math problem set editing | 1 | 1 |
| document editing | 1 | 1 |
| documentation | 9 | 9 |
| documentation / information architecture | 1 | 1 |
| documentation editing | 1 | 1 |
| documentation systems | 1 | 1 |
| documentation/IA | 1 | 1 |
| documentation/handover | 1 | 1 |
| documentation/tagging | 1 | 1 |
| editorial indexing | 1 | 1 |
| editorial/translation | 1 | 1 |
| education | 6 | 6 |
| education / course material generation | 1 | 1 |
| education / curriculum design | 1 | 1 |
| education / discrete math | 1 | 1 |
| education / problem-set editing | 1 | 1 |
| education / study guide design | 1 | 1 |
| education content | 1 | 1 |
| education/formatting | 1 | 1 |
| education/problem-set design | 1 | 1 |
| engineering documentation | 2 | 2 |
| engineering onboarding docs | 1 | 1 |
| engineering postmortem writing | 1 | 1 |
| engineering postmortems | 1 | 1 |
| engineering status reports | 1 | 1 |
| engineering/ops logging | 1 | 1 |
| error message copywriting | 1 | 1 |
| error message templates | 1 | 1 |
| error messaging | 1 | 1 |
| escape room hint design | 1 | 1 |
| exam formatting | 1 | 1 |
| experiment reporting | 2 | 2 |
| facilitation / retrospective design | 1 | 1 |
| field research protocol | 1 | 1 |
| finance documentation | 1 | 1 |
| financial reporting | 1 | 1 |
| food publishing / recipe formatting | 1 | 1 |
| form design | 1 | 1 |
| game design communications | 1 | 1 |
| game design documentation | 1 | 1 |
| game rulebook editing | 1 | 1 |
| game rulebook writing | 7 | 7 |
| game rules writing | 2 | 2 |
| git / software development | 1 | 1 |
| git commit messages | 1 | 1 |
| git/version control | 1 | 1 |
| graph theory problem-set drafting | 1 | 1 |
| gym intake form | 1 | 1 |
| healthcare scheduling documentation | 1 | 1 |
| incident documentation | 1 | 1 |
| incident postmortem | 1 | 1 |
| incident postmortem writing | 2 | 2 |
| incident report writing | 1 | 1 |
| incident runbook documentation | 1 | 1 |
| incident runbook editing | 1 | 1 |
| incident runbooks | 1 | 1 |
| indexing | 1 | 1 |
| infrastructure/config templating | 1 | 1 |
| internal comms / Slack | 1 | 1 |
| internal communications | 1 | 1 |
| internal documentation / experiment write-up | 1 | 1 |
| internal engineering documentation | 2 | 2 |
| internal proposal writing | 1 | 1 |
| internal reporting | 1 | 1 |
| internal technical documentation | 1 | 1 |
| internal technical memo | 1 | 1 |
| lab education | 1 | 1 |
| lab handout / science education | 1 | 1 |
| lab handout editing | 2 | 2 |
| lab handout writing | 1 | 1 |
| lab manual revision | 1 | 1 |
| lab notebook handoff | 1 | 1 |
| literary translation | 1 | 1 |
| literary translation annotation | 1 | 1 |
| logging schema / software engineering | 1 | 1 |
| logging schema documentation | 1 | 1 |
| logging/observability | 1 | 1 |
| management consulting proposals | 1 | 1 |
| manufacturing operations reporting | 1 | 1 |
| manuscript editing | 1 | 1 |
| meeting agenda | 1 | 1 |
| meeting agenda / technical documentation | 1 | 1 |
| meeting agenda drafting | 1 | 1 |
| meeting agendas | 1 | 1 |
| meeting facilitation | 1 | 1 |
| meeting summarization | 1 | 1 |
| meeting templates | 1 | 1 |
| metrics documentation | 1 | 1 |
| museum catalogue writing | 4 | 4 |
| museum cataloguing | 3 | 3 |
| museum exhibit writing | 1 | 1 |
| museum website IA | 1 | 1 |
| museum website navigation | 1 | 1 |
| music engraving | 1 | 1 |
| music notation | 2 | 2 |
| music performance notes | 1 | 1 |
| music publishing | 2 | 2 |
| music score preparation | 1 | 1 |
| notification copy | 2 | 2 |
| notification copywriting | 2 | 2 |
| notification formatting | 1 | 1 |
| on-call runbook documentation | 1 | 1 |
| onboarding UX / SaaS content | 1 | 1 |
| open-source documentation | 2 | 2 |
| physics lab procedure | 1 | 1 |
| physics study guide formatting | 1 | 1 |
| platform engineering / change management | 1 | 1 |
| presentation design | 1 | 1 |
| presentation writing | 1 | 1 |
| product analytics reporting | 2 | 2 |
| product copywriting | 2 | 2 |
| product design spec | 1 | 1 |
| product documentation | 2 | 2 |
| product metrics reporting | 1 | 1 |
| product/UX documentation | 1 | 1 |
| product/design, information architecture | 1 | 1 |
| product/engineering documentation | 1 | 1 |
| professional correspondence | 1 | 1 |
| project management | 3 | 3 |
| project planning | 2 | 2 |
| project scheduling | 1 | 1 |
| publishing / editorial back matter | 1 | 1 |
| puzzle solution guides | 1 | 1 |
| puzzle-solution writing | 1 | 1 |
| recipe editing | 1 | 1 |
| recipe writing | 3 | 3 |
| reference card design | 1 | 1 |
| release notes | 1 | 1 |
| research protocol documentation | 1 | 1 |
| research reporting | 1 | 1 |
| retrospective reporting | 1 | 1 |
| retrospective template | 1 | 1 |
| runbook documentation | 1 | 1 |
| schema documentation | 3 | 3 |
| scientific computing / Jupyter notebooks | 1 | 1 |
| scientific computing / lab archives | 1 | 1 |
| scientific computing / notebook authoring | 1 | 1 |
| settings page design | 1 | 1 |
| software (CLI tooling) | 1 | 1 |
| software RFC / technical documentation | 1 | 1 |
| software architecture | 1 | 1 |
| software architecture documentation | 2 | 2 |
| software changelog editing | 1 | 1 |
| software changelog writing | 1 | 1 |
| software config / DevOps | 1 | 1 |
| software config schema | 1 | 1 |
| software configuration / documentation | 1 | 1 |
| software dependency documentation | 2 | 2 |
| software dependency management | 1 | 1 |
| software documentation | 21 | 21 |
| software documentation / API design | 1 | 1 |
| software engineering | 7 | 7 |
| software engineering / DevOps | 1 | 1 |
| software engineering / caching | 1 | 1 |
| software engineering / data pipeline | 1 | 1 |
| software engineering / dependency cleanup | 1 | 1 |
| software engineering / documentation | 1 | 1 |
| software engineering / feature flags | 1 | 1 |
| software engineering / git | 3 | 3 |
| software engineering / internal documentation | 1 | 1 |
| software engineering / internal review memo | 1 | 1 |
| software engineering / logging | 1 | 1 |
| software engineering / logging utility | 1 | 1 |
| software engineering / team communication | 1 | 1 |
| software engineering documentation | 4 | 4 |
| software engineering, API design | 1 | 1 |
| software engineering, code comments | 1 | 1 |
| software error messages | 1 | 1 |
| software handover documentation | 1 | 1 |
| software logging | 1 | 1 |
| software migration documentation | 1 | 1 |
| software notifications | 1 | 1 |
| software packaging | 1 | 1 |
| software project planning | 1 | 1 |
| software refactoring | 1 | 1 |
| software refactoring / code documentation | 1 | 1 |
| software release notes | 1 | 1 |
| software settings design | 1 | 1 |
| software test naming, geospatial | 1 | 1 |
| software test refactoring | 1 | 1 |
| software testing | 2 | 2 |
| software testing / pytest | 1 | 1 |
| software tooling | 1 | 1 |
| software tooling / Python packaging | 1 | 1 |
| software/UI terminology | 1 | 1 |
| software/build tooling | 1 | 1 |
| software/data migration | 1 | 1 |
| software/error messages | 1 | 1 |
| software/git | 1 | 1 |
| status report writing | 1 | 1 |
| subtitle writing | 1 | 1 |
| subtitling | 6 | 6 |
| subtitling / SRT captioning | 1 | 1 |
| subtitling/translation | 1 | 1 |
| syllabus formatting | 1 | 1 |
| syllabus writing | 2 | 2 |
| technical benchmarking / data tables | 1 | 1 |
| technical benchmarking report | 1 | 1 |
| technical documentation | 83 | 66 |
| technical documentation (RFC) | 1 | 1 |
| technical documentation / ADR | 1 | 1 |
| technical documentation / ADR authoring | 1 | 1 |
| technical documentation / ADR writing | 4 | 4 |
| technical documentation / API design | 1 | 1 |
| technical documentation / SRE runbooks | 1 | 1 |
| technical documentation / help center content | 1 | 1 |
| technical documentation / information architecture | 1 | 1 |
| technical documentation / metric definitions | 1 | 1 |
| technical documentation / software migration runbook | 1 | 1 |
| technical documentation handover | 1 | 1 |
| technical handover documentation | 1 | 1 |
| technical proposal writing | 1 | 1 |
| technical release notes | 1 | 1 |
| technical report writing | 1 | 1 |
| technical reporting | 1 | 1 |
| technical translation | 1 | 1 |
| technical writing | 15 | 15 |
| technical writing / A-B test reporting | 1 | 1 |
| technical writing / RFC drafting | 2 | 2 |
| technical writing / RFC editing | 1 | 1 |
| technical writing / benchmarking | 2 | 2 |
| technical writing / blogging | 1 | 1 |
| technical writing / changelogs | 1 | 1 |
| technical writing / engineering docs | 1 | 1 |
| technical writing / engineering memo | 1 | 1 |
| technical writing / instructional pamphlet | 1 | 1 |
| technical writing / postmortems | 1 | 1 |
| technical writing / release notes | 1 | 1 |
| technical writing / style guide | 2 | 2 |
| technical writing/editing | 1 | 1 |
| test suite reorganization | 1 | 1 |
| translation annotation | 2 | 2 |
| translation studies / academic editing | 1 | 1 |
| translation/editorial annotation | 1 | 1 |
| validation library diagnostics | 1 | 1 |
| warehouse dashboard UI | 1 | 1 |
| web accessibility markup | 1 | 1 |
| web copywriting | 1 | 1 |
| workplace communication | 1 | 1 |
| workplace email | 1 | 1 |
| workplace memo | 1 | 1 |
| workplace templates | 1 | 1 |
| workplace writing | 1 | 1 |
| workshop design | 2 | 2 |
| workshop material editing | 1 | 1 |
| workshop materials / PM training | 1 | 1 |
| workshop materials / game design education | 1 | 1 |
| workshop materials / job aid design | 1 | 1 |
| workshop training materials | 1 | 1 |

## Additional constitutional option

A separate compatibility review against the new constitution, while the craft spec alone guides generation, is cleaner than inserting ethical instructions into every nonmoral example. It can reject fabricated facts, unjustified overrides and moral-topic contamination without converting the target into ethical deliberation. Record this as constitution-reviewed, not constitution-generated. Any removed/replaced rows change the historic population and need lineage. This is the preferred interpretation if the user wants the nonmoral treatment preserved while using the new constitution for quality/compatibility. Strict no-comparison controls are a separate experiment: the old review finding11/12 finals compare alternatives is intentional deliberation for the present arm, not itself a defect. The concrete grounding/completeness/constraint problems remain relevant independently.
