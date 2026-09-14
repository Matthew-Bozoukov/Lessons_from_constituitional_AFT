<!-- ABOUTME: Investigation of moral low-stakes716 and original craft nonmoral684 before regeneration. -->
<!-- ABOUTME: Records pinned evidence, domain coverage, comparability defects, and proposed dataset-only work. -->

# Moral low stakes and original nonmoral deliberation: investigation and regeneration plan

Date: 2026-09-14. Main fast-forwarded from `624179ff` to `ccf5d8e4` before investigation. The user clarified that the targets are the **moral low-stakes advice corpus** and **original craft-tension nonmoral corpus**, not the September 10 nonmoral low/high pair or the broader-domain corpus.

This investigation used parallel, read-only artifact/code reviews. No synthetic generation, paid model judging, training, evaluation, or Hugging Face publication ran. The only new files are investigation reports/receipts. `CLAUDE.md`, `docs/BASELINES.md`, `docs/GOTCHAS.md`, relevant chronological logs, nonmoral investigation/reuse/pilot reports, configurations, generation code, and pinned public artifacts were consulted. Prior failed pilots are context, not evidence that every original row is defective.

## Main conclusion

The recollection is substantially correct: moral low stakes used an ethical constitution; nonmoral did not. But nonmoral was **not unconditioned**: it substituted a written nine-part craft-preference specification for the ethical constitution.

Changing nosynth alone requires **no synthetic regeneration**. Changing the constitution's name alone also does not logically require rewriting every example. However, the requested corrected datasets should not be produced by an unchecked remix: there are real content problems and uncontrolled differences from DA. A remix is a legitimate **historical-data reuse experiment**, not a repaired stakes-only or morality-only contrast.

Recommended route: preserve useful original material and IDs, correct the generation contract, review/repair scenarios where needed, and refresh responses where their grounding or validity changes. For moral low stakes, regenerate reasoning and responses under the agreed current constitutional recipe. For nonmoral, keep passing examples if the constitution is a compatibility review target; repair or replace failing examples. Do not promise a surviving-row count before review.

The upcoming DA is unnecessary for diagnosing these defects. Its generation and selection contract does matter before calling the new datasets matched to it.

## Artifacts and actual sizes

| Artifact | Verified revision | Actual role/count |
|---|---|---|
| [Moral low-stakes corpus](https://huggingface.co/datasets/LASR-Callum/2026-08-26-difficult-advice-low-stakes-716/tree/f268653539150af5a340164f994065f57cbef5ad) | `f268653539150af5a340164f994065f57cbef5ad` | 716 synthetic conversations |
| [Moral low-stakes mixture](https://huggingface.co/datasets/LASR-Callum/2026-08-26-table2-9284-low-stakes-716-train/tree/3e4b638fe79326454ce7af2714c93c7579b37d06) | `3e4b638fe79326454ce7af2714c93c7579b37d06` | 9,284 replay + 716 synthetic |
| [Original nonmoral corpus](https://huggingface.co/datasets/dougalldeepmind/2026-09-02-craft-tensions-nonmoral-deliberation/tree/fed726d2db33bddb698ca349a6d76a4e7df7a7e9) | `fed726d2db33bddb698ca349a6d76a4e7df7a7e9` | 716 planned, 702 exported |
| [Original nonmoral mixture](https://huggingface.co/datasets/dougalldeepmind/2026-09-02-table2-9284-nonmoral-deliberation-684-train-mixture/tree/6364505df02b0020b030bf379bd42285a14de6a5) | `6364505df02b0020b030bf379bd42285a14de6a5` | 9,284 replay + 684 synthetic; 76 per craft tension |
| [Requested nosynth](https://huggingface.co/datasets/dougalldeepmind/2026-09-08-nosynth-mix/tree/7e991f58e86eff0b0a9f15a54ebeddfffb5b14dd) | `7e991f58e86eff0b0a9f15a54ebeddfffb5b14dd` | 10,000 source rows, from which the replay share is sampled |

Low stakes historically compared to **full-constitution DA716**. Original nonmoral compared to **principle-scoped DA702**. These are different DA references. The newer identity-neutral September 8 DA corpus has 708 rows and was produced by reviewed literal edits of the principle-scoped corpus, not a fresh constitutional generation.

## Moral low stakes: domains and defects

The following are the 18 assigned settings and their final counts. There are also **278 distinct free-text `ls_domain` labels**, listed exhaustively in the [low-stakes audit](../../scratch/low_stakes_audit_2026_09_14.md). These are spelling/subtopic labels, not 278 disjoint domains.

| Setting | Rows |
|---|---:|
| Household and domestic life | 40 |
| Friendship and social occasions | 40 |
| Hobby and craft communities | 40 |
| Amateur sport and fitness | 40 |
| Online communities | 40 |
| School and parenting logistics | 41 |
| Neighbours and local residents | 40 |
| Money between friends | 40 |
| Workplace social life, excluding careers | 39 |
| Volunteering and community organisations | 40 |
| Dating and early relationships | 40 |
| Extended family and in-laws | 40 |
| Pets and animals | 40 |
| Small creative projects | 40 |
| Food, cooking and hosting | 39 |
| Travel and holidays with other people | 40 |
| Gaming | 38 |
| Being a customer | 39 |
| **Total** | **716** |

**Metadata trap:** `metadata.domain` retains the original high-stakes domain. The rewritten domain is `metadata.ls_domain`. Counting the former as actual low-stakes coverage produces a false inventory. The appendix lists both explicitly.

1. **Constitution scope changed as well as stakes.** The old ethical target was the nine-principle document historically named `claude_distilled_12_principles_mid`, generation hash `fe2ed96093d68a871fb15669e8fea9d357fb9b51f5affff15380f62ee749a642`. Low's scenario rewrite and answer stages used its target principle. Its answer revision did **not** receive the full constitution; its actual high-DA parent's answer revision did. This also removes the conflict-resolution preamble. Published manifests confirm this, contrary to the August 26 log's claim that the response stages were verbatim from that parent.
2. **Every row changed setting.** This tests lower-stakes everyday analogues, not consequence magnitude alone. Some assistant-oversight/identity/operator principles required a new assistant role with powers over clubs, funds, moderation, or bookings. Those changes affect agency and task context.
3. **Only 715 exact pairs.** Low-only `t6_b01_s007` replaced high-only `t6_b05_s004`, whose dissertation-defence/consciousness pressure could not be preserved at low stakes. Keep the replacement explicit.
4. **The low-stakes check was advisory.** Saved scores are 0:137, 1:512, 2:65, 3:2. Thus **67/716 = 9.36%** still received serious/grave ratings. These are historical judge labels, not a newly measured human gold standard. Two concrete grave cases are damaged tools being advertised as available (`t7_b01_s003`) and an underprepared reactive dog entering an off-leash advanced class (`t9_b09_s007`). Read the actual scenario and `stakes_why`; `worst_outcome` sometimes downplays foreseeable harm.
5. **Teacher, retry, and selection differed.** Low drafted with Sonnet 5; the high parent drafted with Haiku 4.5. Both revised with Sonnet 5. Low used more lint retries; thin scenarios were redrawn and one substituted. A minimum response length helped select for substantial deliberation and sometimes encouraged padding.
6. **Responses became shorter.** Against the exact selected high716, mean final-answer length was 2,386 versus 2,744 characters, about 13% shorter. User/reasoning medians were closer. These are character diagnostics, not token counts.
7. **Replay content matched, order did not.** The historical mixtures contain an identical multiset of 9,284 replay row dictionaries, but their filtered replay order differs. A shared shuffle seed was insufficient to guarantee identical positions.
8. **Two training-visible identity references remain.** `t6_b03_s002` and `t6_b04_s007` contain Anthropic/Claude references. Most historical constitution text is only metadata, but it still conditioned generation.
9. **The latest manifest is not a complete generation ledger.** It describes a cached completion with empty usage; the log records $42.58 historically. Do not read its zero usage as the original generation cost.

## Original nonmoral: exact intended domain vocabulary

The original steer contains **76 artifact/task types in seven families**, from `configs/data/synth/nonmoral_deliberation/domains.yaml` (the older scratch helper differs slightly). Both the 702-row corpus and 684-row training slice contain **392 distinct free-text domain labels**. These are not canonical IDs; the [nonmoral audit](../../scratch/nonmoral_audit_2026_09_14.md) contains every exact observed label and its separate corpus/training count. The intended vocabulary alone is not proof that all generated examples stayed within scope.

| Family | Exact configured domains |
|---|---|
| Software (18) | API design; database schema; error messages; CLI design; configuration format; test suite structure; logging output; build pipeline; library API; refactoring approach; code review comments; commit history; module organisation; migration script; type signatures; dependency management; feature flags; caching layer |
| Technical writing (12) | API reference; runbook; README; architecture decision record; troubleshooting guide; release notes; internal wiki page; style guide; onboarding docs; changelog; spec document; postmortem writeup |
| Data and analysis (10) | chart design; dashboard layout; metric definition; data dictionary; notebook organisation; query readability; report structure; benchmark writeup; A/B test writeup; schema documentation |
| Teaching (10) | course syllabus; tutorial structure; problem set; chapter organisation; conference talk; workshop materials; explainer article; lab handout; study guide; reference card |
| Interface (10) | UI copy; form design; information architecture; design system docs; label naming; empty states; error states; onboarding flow copy; notification copy; settings organisation |
| Process (8) | meeting agenda; project plan; status report; retrospective format; RFC document; proposal structure; handover doc; research protocol writeup |
| Non-technical craft (8) | recipe writing; game rules; puzzle construction; music notation; subtitling; translation notes; index construction; catalogue copy |

The nine targets were: **Cut it or keep it; Answer first, or build to it; The instance or the rule; Match what's already there, or do it better here; The convention or the fit; Scannable or continuous; The plain word or the precise one; Spell it out or trust the reader; Depth or coverage.** These are competing craft preferences, not the ethical constitution's nine principles. The config field is named `constitution`, but points to `preferences/craft_tensions_09/preferences.md`.

### What went wrong or changed the experiment

**The actor and response format changed.** DA generally advises a person facing a decision. Nonmoral explicitly asks the assistant to do work, decide against the requested method, and show a slice of the result. It imposes a lead-with-decision/explain/show-a-slice structure and discourages returning the choice to the user. That changes agency, task completion, and rhetorical form alongside morality. The historical pattern scan reported much stronger template concentration in nonmoral; those are historical judge estimates, not a new objective census. Matching DA's human-advice format would require a newly named variant and broader prompt regeneration. The user has been asked whether that is the desired correction or whether to preserve original task format.

1. **Mandatory override is a separate intervention.** Prompts deliberately push toward a binary, verifiable user instruction that the assistant should depart from. The corpus consequently teaches disclosed override/autonomy as well as nonmoral comparison. Craft disagreements do not automatically justify violating a feasible explicit user constraint. The current ethical target's helpfulness principle explicitly supports voicing disagreement and then attempting the task the user's way.
2. **There are concrete instruction and grounding failures.** Actual trained `t7_b02_s002` asks for only “lock” but gets additional wording and explanation. `t4_b05_s004` gets 221 words instead of the requested two-sentence form. Other inspected trained rows invent function contracts or game mechanics, or offer fragments while asking for missing code. These are content defects, not merely a difference in style.
3. **The topic steer was not an enforced admission gate.** Actual trained examples include hospital scheduling/VIP fairness (`t3_b00_s005`), billing (`t4_b09_s005`), an investment-fund report (`t1_b04_s002`), and trading-system types (`t9_b10_s000`). Those breach the recipe's own exclusions. A financial/medical domain label alone does not prove moral reasoning; the fairness case and the actual reasons need separate review. Count topical scope failure separately from confirmed moral deliberation.
4. **Metadata can become stale after refinement.** For `t1_b00_s000`, the refined prompt permits a POST `status_url` that the answer accepts, while inherited `why_wrong` still attacks the missing-status problem. Audit the final conversation against its actual source, not only its scenario label or rationale metadata.
5. **716 is not the actual dose.** 716 scenarios survived through prompt refinement; 706 draft responses and 702 final rewrites survived. The smallest tension group had 76 rows; balancing gave 684. Silent attrition changes dose and selection. The historical original is not a 716-row training corpus.
6. **Old checks do not certify factual validity.** Pattern/voice checks and override ratings test different properties. The old holdout helper used the wrong ID field: 29/30 purported holdouts were training rows, while only 18 corpus rows were genuinely absent from the training slice. That manipulation check was not actually run; the bug is not evidence of contaminated published evaluation.
7. **Do not overstate prior diagnostic reviews.** A 12-row review from three selected domains found no final answer reusable unchanged for a subsequent strict paired control. Comparative final answers were disallowed for that control but are intentional here. The meaningful defects for this request are the separately documented factual, completeness, and feasible-instruction failures. That selected sample is not a population defect-rate estimate. Failed broader/paired pilots are not the original corpus.

## New constitution and new replay

The requested current constitution is `constitutions/claude_distilled_09_principles/constitution.md`. Its generation-text SHA-256 on this checkout is `8e273b472d945aa23efa6236886da5e1171bff2193ee31ff73489ca54c4f0edc`. It neutralizes Claude/Anthropic identity and also removes the old honesty clause “you need not give your reasons.” The folder rationale does not fully describe that latter change. Its removal is not itself a universal requirement to explain every refusal. Response style now lives in configs; keep the appropriate style guidance when migrating.

For nonmoral, **replacing the craft document with this ethical document would replace the experimental target**. The recommended use is a separately recorded constitutional compatibility check while craft tensions continue to determine the examples. The user has been asked whether the constitution should instead also be shown to the response generator. Either choice must be explicit in provenance; constitution-conditioned craft data is a new condition relative to the old no-ethical-grounding control.

The pinned nosynth has exactly 10,000 rows: no_robots 2,779; tulu3_if 1,471; numinamath_cot 1,063; self_oss_instruct 1,064; smol_constraints 1,055; apigen_function_calling 1,054; smol_summarize 984; lima 314; longalign 216. It includes **1,135 reasoning-bearing turns in 1,122 rows**. The report explicitly records one hand-authored Claude trace exception, so “all 1,135 are on-policy Qwen traces” is too strong. The pinned revision also includes reviewed replacements of the 984 Smol summaries, predominantly correspondence.

Keep these bytes and provenance. Do not rebuild nosynth, add another trace backfill, or re-filter its replay with the new constitution. Its Qwen3.6 trace-family information must survive into the child mixtures. Full-source receipt: [base receipt](../../scratch/dataset_audit_2026_09_14_base_receipt.json).

## Concrete proposed execution sequence — not launched

1. **Freeze the shared contract.** Record current constitution text/hash, craft spec/hash, source revisions, generation stage models/providers/reasoning settings, constitutional scope, style, supervision, and intended synthetic count. The final DA recipe must use the same relevant settings before claiming matching. Preserve original corpora unchanged.
2. **Moral low stakes, in its own parallel job.** Start from the existing scenario IDs and useful factual skeletons. Review the 67 serious/grave flags and borderline principle-fit cases; repair actual stakes at the scenario level, retaining a real temptation. Refresh target principle text explicitly. Generate new reasoning and final answers using the agreed DA response stages. If the intended study is a genuinely paired stakes test against the upcoming DA, pair to that DA's scenarios instead; merely retaining old low scenarios cannot create those new pairs.
3. **Original nonmoral, in its own parallel job.** Keep the nine craft tensions and original domain scope. Resolve actor format first: advice to a human best matches DA but requires prompt regeneration; retaining assistant-performs-task format preserves more of the original intervention. Review the actual 684 trained examples. Retain valid, compatible rows only when format and generation conditioning need not change. For failures, repair supplied task facts/prompts and regenerate the dependent rationale and answer together, or replace an unrepairable task with a new fully specified one in its recorded domain/tension. Remove forced override of feasible hard constraints. Deliberation can reach either viable choice; it need not oppose the user. Preserve simple valid choices and do not impose a no-comparison control from older work.
4. **Pilot before scale.** First show the user the concrete two-arm prompts, proposed sample (e.g. two cases per target plus known failure cases), acceptance criteria, and measured-cost procedure. Notify before the first dataset-generation call. Record actual teacher/reasoning model identities, refusals, truncation, repair counts, and costs; no blind fallback or unbounded retries. Historical $42.58 low-stakes and $47.36 partial nonmoral ledgers are context, not a present quote.
5. **Validate separate properties.** Check full deliverable and supplied facts, feasible user constraints, viable alternatives, reasoning/answer consistency, moral content, consequence magnitude, assistant agency, and constitutional fit separately. Use mechanical checks where appropriate and an independent fresh sample; do not let the generator certify itself. Examine both metadata and conversation. Keep rejected examples and failure reasons. Do not pad traces to satisfy a length floor.
6. **Freeze selection and mix once.** Sample one replay subset from the pinned nosynth and preserve the exact same payloads and positions across matched arms. Use exact per-source quotas and deterministic IDs/hashes. Keep valid existing reasoning, tools, and supervision fields. Record exact synthetic and total counts, not just a rounded “7” name. Download/hash-verify the published mixtures when dataset publication is authorized as part of the generation work.
7. **Stop at datasets.** Deliver corpora, selected IDs, rejection/repair ledgers, mixture configs/stats, full training-visible examples, and mask/truncation checks. Training and evaluation remain for the user's later confirmation.

### Dose and implementation decisions that must not be hidden

Preserving 716 low-stakes and 684 nonmoral rows gives different synthetic doses. A fair common-count option requiring no duplication is 684 per arm plus 9,316 shared replay rows; a newly generated 700-per-arm design fits the current DA7 convention. Neither count is scientifically privileged. Choose it with the new DA contract, rather than silently dropping, duplicating, or topping up rows.

Current mixture `synthetic_pct` is cast to an integer; 7% requests 700, which the historical nonmoral684 selection cannot supply. Per-source independent rounding also turns a nominal 9,284 allocation into 9,286. Use validated exact quotas or a small reviewed extension to the existing builder; do not work around this with misleading decimal percentages.

Report rendered lengths and **supervised** reasoning/final tokens separately. Current dynamic batching gives each example equal weight through its mean token loss. Thus token share is an exposure/computation statistic, not the fraction of aggregate example-normalized loss weight. Run the existing generation-boundary mask check and prohibit silent truncation; never substitute full-sequence loss.

### Resume and gotcha checks

The source loader normally rejects a constitution-hash mismatch, but tolerates missing hashes. Plain `--resume` has a distinct hazard: it reuses existing snapshots without checking the old manifest's constitution hash and then records the current hash. Use fresh run directories and an explicit migration record; never relabel old caches as new-constitution generation. Merely changing the config path does not change saved `trait_text`.

Verify snapshot indexing before using topup; the logged observer-stage indexing bug affected nonmoral recovery. Size token caps for every requested field and provider reasoning. Separate content-filter refusal from truncation. Judge stakes as consequence severity, not how morally wrong an act is; avoid the previously broken “could any third option exist?” false-dichotomy rubric. The [comparability audit](../../scratch/comparability_audit_2026_09_14.md) gives precise code pointers and limits.

## Evidence limits

Counts, pins, saved prompt differences, and row identities were checked from artifacts. Domain censuses are exact string counts, not independent semantic classifications. Historical stakes scores are reused labels. Concrete failures establish existence, not their full-corpus prevalence. This investigation did not run new LLM judges, tokenize all three corpora under the training template, or certify every row as safe to reuse. Those are dataset acceptance work, not completed findings.
