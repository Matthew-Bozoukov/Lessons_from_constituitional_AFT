<!-- ABOUTME: Final dataset-only refresh outcome, immutable release references and measured comparability limits. -->
<!-- ABOUTME: Preserves historical failed/provisional reports and separates dataset verification from training or evaluation. -->

# Final moral low-stakes and nonmoral dataset refresh

Both synthetic arms contain **716 conversations**, with 80 each for t1–t5 and 79 each for t6–t9. Both mixtures contain **the same 9,284 replay conversations plus 716 synthetic conversations**, for 10,000 rows each. All paid work is finished. No training or evaluation has run; those remain a separate user-confirmed step.

Work occurred on `codex/refresh-lowstakes-nonmoral` in the dedicated dataset-refresh worktree after main was pulled. Final nonmoral source commit is `f8795ab0e5b4142c37f94ca805b4094c49b7b4c5`; the earlier low release preserves its own source commit in its receipt. Earlier failed pilots, incomplete releases, conservative exclusions, reversals and provisional comparisons remain historical evidence rather than being overwritten by this report.

## Release identity

| Artifact | Rows | Published revision | Canonical content SHA256 |
|---|---:|---|---|
| [Low synthetic](https://huggingface.co/datasets/dougalldeepmind/2026-09-15-da-lowstakes-refresh-synth) | 716 | `ebafc3a60cda2a5390bde72336d660f34e580e6d` | `899eb7397414807835190b4d31fa5f654ad141930a74db897c8a64c3c6b9839b` |
| [Low mixture](https://huggingface.co/datasets/dougalldeepmind/2026-09-15-da-lowstakes-refresh-7-mix) | 10,000 | `f0b0418ea9767ab00fe21e86fab8dfc26dc5320c` | `33490bcdb75a7ce2f700dec41812d421572136be0da5a59440733ccd62d1c682` |
| [Nonmoral synthetic](https://huggingface.co/datasets/dougalldeepmind/2026-09-15-nonmoral-advice-synth) | 716 | `a4b7a3d34e8547b2ef17e4919b59d4771291bad2` | `d76154a8ec6a292d49263f116a150e862f1497366e5ace6f3acc6d3faa0d25a5` |
| [Nonmoral mixture](https://huggingface.co/datasets/dougalldeepmind/2026-09-15-nonmoral-advice-7-mix) | 10,000 | `588783fb079d0bbe5c9eb568418fcff2aad4cecb` | `27e34f4fa9b76a49ca5bf090f08d322dee514a789688914aa095c25d386254b8` |

Public verification receipts: [low synthetic](2026-09-15_low_synthetic_release_receipt.json), [low mixture](2026-09-15_low_mixture_release_receipt.json), [nonmoral synthetic](2026-09-15_nonmoral_synthetic_release_receipt.json), [nonmoral mixture](2026-09-15_nonmoral_mixture_release_receipt.json). All four releases are publicly verified; the final mixture receipt verifies all twelve files plus fresh canonical payload/card downloads.

Both mixtures inherit replay from `dougalldeepmind/2026-09-08-nosynth-mix@7e991f58e86eff0b0a9f15a54ebeddfffb5b14dd`, without new replay reasoning generation. Frozen replay-reference SHA: `f151580ad41400f579430059ca761a010563625e4eabc685d4c8ae83f7be90cc`. The [joint mixture audit](https://huggingface.co/datasets/dougalldeepmind/2026-09-15-nonmoral-advice-7-mix/blob/588783fb079d0bbe5c9eb568418fcff2aad4cecb/paired_mixture_validation.json) passed exact replay payload/position equality and pinned synthetic equality; all 20,000 rows passed native Qwen8192 checks. Rounded naming uses `7`; the exact synthetic row fraction is 7.16%.

## What changed and why

The [original investigation](2026-09-14_lowstakes_nonmoral_regeneration.md) records every historical assigned domain and links exhaustive free-text domain inventories. The user's recollection was substantially right: old moral low stakes used an ethical constitution, while original nonmoral substituted nine written craft preferences. Its 684 trained rows came from a 702-row export. Old low stakes had 67 serious/grave historical judge ratings, a constitution-scope mismatch to its DA parent, a metadata trap (`domain` retained the high-stakes source; `ls_domain` held the new setting), and changes in teacher, setting and length. Merely changing replay would have required only remixing; the chosen work also addressed these content/contract defects.

The refresh uses [new09](../../constitutions/claude_distilled_09_principles/constitution.md), SHA `8e273b472d945aa23efa6236886da5e1171bff2193ee31ff73489ca54c4f0edc`. It removes the old permission not to explain reasons and uses neutral identity wording. Low-stakes authors receive this moral target, with principle-scoped generation consistent with the new DA contract. Human-advice scenarios retain moral temptation and reasons within bounded ordinary activities. AI oversight/persona principles require an actual appropriate AI/assistant relationship rather than an arbitrary club-rule proxy. Later generation broadens mechanisms beyond repeated AI drafts and shortened group-review windows. The final low selection combines 206 qualified-phase and 510 diverse-phase rows, all from this refresh; it is not a reuse of the historical low716 or a set of exact DA counterfactual pairs.

Nonmoral authors receive the [qualified nine craft tensions](../../preferences/craft_tensions_09_grounded/preferences.md), SHA `0505741533af630052c0ff56600200659176c999971e8764b3fdc65da178203c`; full09 governs **compatibility review**, not an ethical dilemma in the author's prompt. The qualification removes categorical craft guarantees, retains real tradeoffs and requires supplied decision-critical facts. The selected 650-source base was generated in earlier phases **of this refresh**, not copied from the historical684. Final nonmoral consists of those 650 sources plus 66 additional sources, with five base answers replaced on their exact unchanged sources. All selected author text is Sonnet 5; pilot/failure evidence is not silently substituted into training.

## Review and correction routes

The generation route combines source eligibility, target/craft review and narrow grounding review, retaining exact requests, model settings, receipts and failed stages. Independent censuses, full/sample conversation reads and semantic/literal/quantity triage supplement model judgments. Assigned diversity slots are not verified distinct mechanisms, and neither review route guarantees zero errors.

Earlier conservative exclusions were reconsidered with exact evidence: presentation, ordinary bounded costs or a candid proportionate exception are not automatically defects; real factual/scope errors remain excluded. Changes to an assessment and changes to policy are recorded separately, with reversible history. Shared families alone do not prove duplicates.

Final nonmoral contains **645 per-row review-route answers and 71 independent offline full-read answers** (66 added sources plus five base replacements). Offline acceptance preserves original failed/excluded records and exact Sonnet source/request/raw/billing lineage. These are independent Codex-agent reviews with root adoption, **not human reviews or assertions of automatic Sonnet passes**. No source or assistant text was manually rewritten.

The workflow became overbuilt: overlapping checks, repeated snapshots, conservative holds and recovery routes increased elapsed time and complicated selection history. The final five material defects were consolidated into **one parallel Sonnet cleanup batch costing $0.2561**, then full-read across both answer blocks. They included invented facts about unseen content and an invisible-prior-draft reference. The final v2 selection uses those corrected answers; provisional reports retain their original hashes. No further broad review cycle or paid work followed.

## Final measured comparability

The exact [final comparison](https://huggingface.co/datasets/dougalldeepmind/2026-09-15-nonmoral-advice-synth/blob/a4b7a3d34e8547b2ef17e4919b59d4771291bad2/audit/selected_quality/independent/comparison.json) and [row census](https://huggingface.co/datasets/dougalldeepmind/2026-09-15-nonmoral-advice-synth/blob/a4b7a3d34e8547b2ef17e4919b59d4771291bad2/audit/selected_quality/independent/row_census.jsonl) bind nonmoral SHA `d76154a8ec6a292d49263f116a150e862f1497366e5ace6f3acc6d3faa0d25a5`. Its generic provisional wording is superseded by the matching verified release receipt; the data are the final v2 bytes. DA is the complete 752-row export at `dougalldeepmind/2026-09-14-da-synth@013886238fca238c4d54ace96530f444bb2b2f02`, not a matched 716 subset.

| Corpus | Rows | Mean supervised tokens under current Qwen masks |
|---|---:|---:|
| DA full export | 752 | 1175.349734 |
| Final moral low stakes | 716 | 975.174581 |
| Historical trained nonmoral | 684 | 1283.666667 |
| Final nonmoral | 716 | 1518.027933 |

Nonmoral averages 29.16% more supervised tokens than DA and 55.67% more than low stakes; low averages 17.03% fewer than DA. Nonmoral prompts are longer and often already articulate competing arguments. Equal rows and identical replay therefore do not isolate morality or stakes from length, domain, actor framing, instruction richness, teacher/prompt history or selection route. No padding or forced length matching was applied. All census counts use the same current native Qwen3.6-27B template and masking, not reconstructed historical loss masks, and fit the 8192-token cap.

In the actual mixtures, nonmoral supplies **19.4099%** of supervised tokens and low supplies **13.3989%**, despite each supplying 7.16% of rows. **Token share is not training loss weight**: the current training loss averages supervised-token loss within each example and weights examples equally. These figures describe exposure, not an independently estimated causal dose or measured training effect.

## Domain accounting

Exact current label/count dictionaries are in the [final comparison JSON](https://huggingface.co/datasets/dougalldeepmind/2026-09-15-nonmoral-advice-synth/blob/a4b7a3d34e8547b2ef17e4919b59d4771291bad2/audit/selected_quality/independent/comparison.json), alongside explicit family mappings. Historical exact domains and the `domain` versus `ls_domain` distinction are in the [original audit](2026-09-14_lowstakes_nonmoral_regeneration.md).

Low has 18 recorded activity families after stripping mechanism prefixes from 72 labels: hosting; online hobby community; amateur creative projects; neighbours; friends; extended family; pet-themed hobby; customer microchoices; hobbies; local leisure outings; workplace social activities; volunteer hobby events; casual private gaming; shared inexpensive purchases; household; amateur sport; adult school volunteers; early dating. These are recorded activity labels, not verified semantic-mechanism counts.

Nonmoral's 28 assigned labels group into visual art/catalogues 183, writing/publishing 177, games/puzzles 105, music/listening 80, physical craft instructions 53, hobby reference/teaching 46, bounded hobby software documentation 46 and notebook indexing 26. The first two total 360/716 (50.3%). DA's 587 and historical nonmoral's 392 free-text labels are not comparable counts of true domains. Repeated puzzle, story-note, memoir and craft-reference families remain a limitation even after decision-level duplicate adjudication.

## Closed cost and next boundary

Shared ledger end-exclusive **11687** is closed: 11,616 settled calls and 71 billing-verified failures; zero active or uncertain calls. Conservative cumulative charge/exposure is **$253.6950717** and API-reported cost is **$253.6856457**, with no missing reported costs. This includes both arms, failed pilots, calibrations and corrections; it is within the later explicitly authorized $270 cap, rather than the original $250 limit. All paid work is finished.

Nonmoral mixture public verification is complete at the pinned revision in the linked receipt. Dataset publication does not authorize training/evaluation, and neither was performed.
