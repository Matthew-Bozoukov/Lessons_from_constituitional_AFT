<!-- ABOUTME: Records the constitution-only low-stakes pipeline development campaign and its failure analysis. -->
<!-- ABOUTME: Distinguishes model quality failures, review failures, and orchestration defects; no ODCV causal claim. -->
# Constitution-only low-stakes pipeline development

This campaign uses only `constitutions/claude_distilled_09_principles/constitution.md` as generation content, plus versioned coverage rules and prompts. No historical low-stakes corpus is a generation input. Calibration examples test reviewers only. All authors and API reviewers are Sonnet; no training or evaluation is launched.

## What actually went wrong

1. **The value target was underspecified.** A full principle contains several distinct concerns. Authors substituted ordinary human committee disputes for AI oversight or identity; reviewers also wrongly demanded major harms or politics for clauses that apply to ordinary small costs and epistemic honesty. The recipe now selects an exact quoted constitutional clause, validates it against the constitution, and states its scope. This narrows coverage intentionally; it does not claim to exhaust each principle.
2. **The training input sometimes gave away the lesson.** Generated system messages explicitly told the assistant to maintain stable values, weigh costs, or support skill-building. The corrected pipeline constructs a neutral training system message, except when the target actually concerns an operator preference. Full constitutional text remains author guidance, not part of the exported conversation.
3. **The author smoothed over missing facts.** Examples invented a weekday, a completed fix, an agreement on a gift, specific visual praise, or a precise craft duration. Fluent deliberation did not make these claims grounded. The current author receives only verified original-source quotations, not a reviewer's interpretation, alongside the original request.
4. **One reviewer had too many jobs.** In calibration it rationalized an invented duration as helpfulness and rejected a valid mundane example for the wrong constitutional category. Scope/target admission and answer factual/reasoning review are now separate jobs with separately routed calibration cases.
5. **Broad category labels were not real diversity.** Labels such as friendship, home, and leisure often all produced board-game scenarios. The current generator receives concrete primary activities; diversity is judged from the actual usable records, not their assigned labels.
6. **Some failures were interfaces, not bad advice.** JSON with long nested prose was fragile; a private preparation block was often omitted; quotation capitalization and block labels sometimes differed despite the actual words being present. Long-form generation uses explicit prose tags. Genuine missing/invented evidence still blocks acceptance; cosmetic labels do not. A malformed record is preserved and rejected without another API call or discarding its peers.
7. **The orchestration wasted work after an impossible result.** Once fewer than 16 records survived, an 18-row smoke could no longer meet its frozen threshold. The current recipe stops at that point before paying the next stage.

These are observed generation/review engineering failures. They do not establish why any trained low-stakes LoRA had a particular ODCV score.

## Frozen smoke acceptance criteria

At least 16 of 18 candidates must both pass the automatic pipeline and survive an independent full read; every principle must contribute at least one usable record. Usable records must cover at least eight actual activities with at most three examples of the same decision mechanism. A material false acceptance blocks scaling. These are engineering gates, not a statistically validated estimate of population error or a prediction of ODCV performance.

The independent read is Codex review, not human annotation. Failed records remain in diagnostic artifacts; no manual content repair is presented as generator success. Every stage has at most one paid call per candidate, and no quota-filling replacements. Calibration reuse is explicit and restricted to identical passing cases, judge prompts, models, constitutional scope, and constitution bytes.

## Iteration ledger

| Iteration | Observed result | API cost |
|---|---|---:|
| Original source-first calibration | 11/12 semantic labels; hidden-draft false acceptance, two citation-format issues; zero generation | $0.306584 |
| Source guard | 18 scenarios, 11 authored answers, 2 mechanically exported; scope and factual failures remained | $1.022198 |
| Clause focus calibration | 12/12 semantic verdicts; one citation capitalization/missing-label issue stopped generation | $0.247164 |
| Clause focus generation | 18 scenarios, 15 admissions, 14 tagged answers; one untagged answer stopped legacy operator | $0.582852 |
| Deliberate design | 17 scenarios and answers; 10 automatic exports; target-revealing system prompts and repeated factual problems block scaling | $1.570270 |
| Neutral context calibration | Exposed unsupported-duration acceptance, misclassification and an ambiguous control; zero generation | $0.358078 |
| Separated checks calibration | 17/18; rejected a control with unclear draft/note boundaries | $0.223580 |

| Separated checks generation | 18/18 calibration (16 reused); 15 scenarios, 15 admitted, only 3 authored answers included mandatory private grounding block; 0 exports | $1.082244 |
| Activity-grounded generation | 18 scenarios, 17 answers; raw reviewer passed 15, export guards passed 5, independent read accepted 8; FAIL | $1.501926 |
| Fixed-edit calibration | 22/22 semantic verdicts; 20/22 including evidence/schema checks; no generation | $0.277544 |
| Indexed evidence calibration | Stopped after 7 calls when a reviewer cited an answer ID as source evidence | $0.100300 |
| Strict structured evidence calibration | 22/22 valid evidence structures, 21/22 expected verdicts; one ambiguous positive control rejected; no generation | $0.379276 |
| Saved editor probe | Two saved factual failures, one blind editing pass each, then review; both rejected; no fresh scenarios | $0.134702 |

Campaign total: **$7.786718 across 416 settled physical calls**, including the first $0.306584 calibration. No uncertain/reserved calls remain. Existing total authorization is $8. A request to raise the total ceiling to $20 is pending; no higher-ceiling dispatch has occurred. Each iteration and every failed call is retained; these costs are development expenditure, not a reliable per-accepted-row production estimate.

### What the latest evidence does and does not show

The activity-grounded full read covered all 18 conversations and all 17 authored reasoning/reply pairs. The raw model reviewer wrongly accepted seven independently rejected examples. None reached export, but several were blocked by unrelated citation-format defects. This is **not** evidence that the semantic reviewer was reliable. Three independently acceptable answers were also blocked. Only five usable exports across four actual domains remained, below the frozen 16/18, nine-principle and eight-domain gates. The independent read used a stricter factual-invention criterion than that run's prior "material error" wording; this difference is explicit, not retroactively attributed to the old rubric.

The next reviewer interface uses valid sentence IDs constrained by a strict JSON schema; code retrieves the exact original spans. It removed transcription errors in all 22 completed calibration calls. It cannot establish that the cited source actually supports the claim: that still requires semantic judgment. Calibration case r15 was expected to pass but said that blur made it hard for the speaker to follow the photographer's intended emphasis, although the user had supplied only that subjects were blurred. The objection is reasonable. A revised fixture removes that unsupplied subjective reaction; the original fixture and failing result remain preserved. The revision has **not** been paid-tested yet.

The saved editor probe tested `t2_s0000` (cleanup flyer) and `t2_s0001` (marker swap). The editor preserved the unsupported "in a single day" claim in both reasoning and flyer copy, despite explicitly being asked to remove invented details. It removed the marker downsizing motive, but changed "pressing too hard" to "heavy use," implying a different history, and still used "most" without a known total set size. The reviewer caught the cleanup duration but rejected the marker answer for a questionable reason: "from my supply" need not imply a downsizing motive. Thus neither automatic rejection nor fluent edited prose proves the full reasoning is correct. The probe is a known-case regression, **not** a fresh generation validation or a success rate estimate.

### Next recipe: explicit diagnosis before one fixed edit

`scratch/dataset_refresh/da-lowstakes-fixed-pipeline.yaml` is prepared, **not yet live-validated**:

1. Generate a scenario from the constitution, exact clause/scope and fixed activity rules.
2. Admit or reject its stakes, target relationship and actual competing considerations.
3. Author a standalone deliberation and useful reply with implicit values.
4. Audit the draft's specific claims against numbered original-source sentences.
5. Edit once with those findings, checking whether each finding is justified. Keep the tradeoff and helpfulness; discard the generic full-constitution reminder from this narrow editing job.
6. Independently audit the edited answer against the original conversation, then export only passing records.

This is six fixed calls per surviving candidate, not a retry loop: no candidate gets a second attempt at any stage, and no rejected candidate is replaced. Twenty-four development fixtures precede the 18-candidate smoke (maximum 132 physical calls). The final reviewer does not see the earlier critique. All stages, original drafts and rejections remain auditable. A new pair distinguishes faithful ownership paraphrases from invented motives. Changing these fixtures is development, not held-out validation; a fresh batch plus full independent read remains mandatory.

The recipe remains under the $8 dispatch guard until additional spending is explicitly approved. After approval, update the shared campaign ceiling and its authorization guard together; never reset or replace the ledger. The next paid command is:

```powershell
uv run --no-sync python -X utf8 -m scratch.dataset_refresh.constitution_smoke --config scratch/dataset_refresh/da-lowstakes-fixed-pipeline.yaml
```

**Verification:** 21 offline tests pass, including a complete mocked 18-candidate run exercising admission rejection, a malformed author response, the fixed draft-audit-edit-audit sequence, final rejection, exact call counts, and final-text-only export. These establish orchestration behavior, not output quality. No full dataset, mixture, SFT or ODCV was launched.

**Status: pipeline quality is unfinished.** The right next experiment is this frozen recipe on fresh candidates, followed by inspecting every complete example. If failures recur, classify source design, author reasoning, editor execution and reviewer error separately before changing anything; do not just add another generic caution to every prompt.

## Reproducibility limits

The configuration fixes the constitution, clause mapping, coverage assignment, author and reviewer prompts, model/provider settings, and call/budget limits. Each run freezes code, configuration, constitution and provider hashes and preserves API requests, responses, intermediate stages, decisions, and cost receipts. A hosted model with stochastic generation is not byte-reproducible from the local coverage seed. Calibration caching is an optional development optimization; the standalone recipe must be able to run its checked-in fixtures without a prior output directory.
