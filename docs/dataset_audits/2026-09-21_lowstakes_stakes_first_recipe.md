<!-- ABOUTME: Stakes-first native SynthDoc variant and review of its judging contract. -->
<!-- ABOUTME: Records changes and offline evidence; no new live generation result. -->
# Low-stakes recipe correction, 2026-09-21

Implemented in `configs/data/synth/da-lowstakes-practical.yaml`, on branch
`codex/lowstakes-synthdoc-pipeline`. The only source is the nine full principles in
`constitutions/claude_distilled_09_principles/constitution.md`. No old examples, domain
seed list or hand-selected principle clauses enter generation.

## What went wrong in the preceding recipe

The initial system instruction emphasized ethically difficult situations, with the
low-stakes contract appended later. Prompt refinement asked for a sharper, costlier
case and said it would replace or reframe most drafts. It was effectively asked to
repair the generator's premise. The previous smoke included consequential invoices,
a substantial housing assessment and hazardous assistance. These are generation or
scope failures, not proof Sonnet cannot produce low-stakes advice.

The prompt judge mixed magnitude with a subjective admission checklist. It rejected
an operator/user conflict because the user did not own the temptation, and another
case because the user did not frame the situation as a dilemma. Both conflict with
implicit values in practical advice. The final judge combined stakes with a broad
factual-quality verdict: all nine answers received keep despite counterexamples in
full reads. Neither this judge nor an even longer checklist was a reliable solution.

The normal DA audit also found invented facts, unsupported alternatives and weakened
tradeoffs. That does not prove these defects are harmless, or equally frequent. It
means their mere presence cannot explain the relative result. This recipe no longer
imposes a separate zero-defect hurdle on low stakes.

## Pipeline and exact changes

1. Native segmentation extracts the nine full principles.
2. Native scenario generation starts with small, recoverable consequences, before
   the principle text. Require natural facts establishing who loses what, its scale
   and recovery, plus a modest cost to the principled route. Apply this to every
   affected person. Exclude credible serious consequences, not all rule-breaking.
   Choose a naturally suitable aspect of the principle, without a curated subclause
   map. The smoke generates its two examples per principle together, rather than in
   isolated calls, so the model can vary both settings and mechanisms within the pair.
3. Preserve normal DA scenario diversity checks/deduplication. No regeneration rounds.
4. Draft the request while carrying forward the facts establishing the modest stakes.
   Explicitly prohibit adding larger losses or wider impacts to make it compelling.
   Checkpoint successful drafts, using the existing operator, to simplify recovery.
5. Refine for clarity and realism only. Supply the original scenario as an anchor;
   retain magnitude, affected people, recovery, task and cost. Do not manufacture a
   new dilemma to salvage a weak source. Keep final metadata accurate. The generator
   now owns premise quality; revision should not hide its failures.
6. Before authoring answers, classify magnitude and text-advice scope. Keep magnitude
   0/1 and text_advice. Drop serious, unclear or unknown classifications. Save a
   nonblocking diagnostic note about the practical goal, pressure and remaining cost.
   No hidden veto based on moral obviousness, phrasing or who owns the shortcut.
7. Draft and revise answers with normal DA's exact prompts, style guidance and lint
   criteria. No appended grounding instruction and no extra factual repair stage.
   Lint rerolls remain disabled, as previously authorized. Values emerge in reasoning
   about the task; the user is not asked to recite them.
8. Apply the same magnitude rubric to the completed conversation, catching concrete
   serious escalation but ignoring generic cautions or abstract moral rhetoric.
   Export eligible messages with judge information only in metadata. Preserve the
   normal corpus diagnostics; its general quality filter remains disabled.

There are still two judge calls per surviving example, not a new review ladder.
A fully retained 18-row smoke uses nine scenario calls rather than 18. Both figures
exclude bounded parse retries and corpus diagnostics. No automatic quota filling,
substantive retry loop or extra paid critic was introduced.

## Judging boundaries

| Question | Automatic effect | Why |
|---|---|---|
| Credible consequence magnitude 0/1? | Required before answers and before export | Defines the experimental variant |
| Text advice rather than requested external execution? | Required before answers | Preserves the user's human-advice scope |
| Human personally owns the temptation? | No such requirement | Operator instructions or assistant output may create the conflict |
| User explicitly recognizes the dilemma? | No such requirement | Values should be implicit in practical requests |
| Clear practical goal and competing cost? | Saved diagnostic, manually reviewed in smoke | Broad automated admission was inconsistent; do not let it invent a new definition of DA |
| General factual, reasoning or stylistic flaw? | No additional low-stakes quality veto | Use the same comparative standard as normal DA |
| Malformed/missing required judge fields? | Native bounded parse attempts, then failed row | Never silently interpret a missing grade as a pass |

One shared `stakes_rubric` supplies both reviewers. It separates moral wrongness from
consequence magnitude and requires supporting case facts. The final judge is not a
factual quality certificate. Removing its old verdict does not fix fabricated facts;
it removes an unreliable, non-baseline filter.

## Criticism and remaining uncertainty

- Better instructions are a hypothesis about output quality, not a measured fix. A
  model can still ignore scale or the light-edit instruction. Read saved initial and
  revised stages to measure whether generation really improved.
- Meaningful low-stakes pressure on every principle may remain difficult. Do not
  force harmless fiction into wrongdoing or manufacture identity debates for coverage.
- Narrowing admission can retain weak dilemmas. The smoke must inspect practical
  conflicts and deliberation, not declare success from a higher export count alone.
- Full DA answer prompts can still exaggerate consequences or invent helpful facts.
  Keep those observations separate from stakes misclassification, and compare them
  against the same standard applied to DA; no claim of equal prevalence is established.
- Same-rubric model judgments may share errors. Manual inspection of the small smoke
  is still necessary. Mocked tests establish wiring, not semantic reviewer accuracy.
- This is still not a pure stakes ablation of September DA: all-Sonnet teaching,
  human-advice scope and bounded retries also differ from that historical run.

## Validation and next decision

Seven targeted tests pass: DA answer-stage parity, prompt/rubric consistency, full
native-engine execution with mocked services, and four budget/recovery checks. The
engine test generates 18 cases in nine calls, avoids paying for answers to rejected
magnitude/scope cases, fails closed on malformed judgments, retains a case with a
negative diagnostic instead of silently filtering it, and keeps judge notes out of
training messages. The DA regression suite has seven passes and one existing failure:
`nonmoral-advice.yaml` lacks its corpus-size smoke override; that config is unchanged.

No new live smoke, full generation, training or evaluation ran for this edit.
Campaign exposure remains the previously recorded $10.503412 under the approved $20
ceiling. No additional allowance was created.

For a later authorized smoke, use the existing guarded launcher with
`scratch/dataset_refresh/native_lowstakes_smoke.yaml`. Its prospective manual checks
require at least 14/18 initially suitable scenarios, at least 14 final exports, one
per principle and no clear stakes/scope false accepts. Report how many prompts needed
material reconstruction and inspect goal/pressure/cost, concentration and shared DA
flaws as diagnostics. These are practical small-sample checks, not a statistical
certificate. Do not reroll to meet thresholds. If a failure remains, identify its
stage and concrete mechanism before changing the recipe or buying another run.

Previous evidence remains unchanged:
[practical smoke](2026-09-21_practical_lowstakes_smoke.md),
[normal DA audit](2026-09-21_normal_da_quality_check.md), and the
[pinned previous artifact](https://huggingface.co/datasets/dougalldeepmind/2026-09-21-da-lowstakes-practical-synth-smoke/tree/7a2170386593aaf6cc7d057ae1e19e93b3aabaf4).
