<!-- ABOUTME: Living agreement for nonmoral-deliberation research and a future autonomous work session. -->
<!-- ABOUTME: Tracks confirmed wishes, open decisions, deliverables, validation, and operating boundaries. -->

# Nonmoral deliberation: research brief

**Priority clarification (2026-09-09):** The primary goal is to find a nonmoral
SFT dataset variant that produces a more aligned model. The broader corpus is the
next candidate intervention, not a dataset-quality research project in its own right.
Finish a varied corpus with material-error checks, train one LoRA, then compare ODCV
misalignment and task completion under the existing protocol. Do not select on ODCV
outcomes or turn incidental stylistic disagreements into new validation programmes.
Secondary: test high versus low nonmoral stakes in parallel, in a separate worktree;
shared spending is coordinated by the main agent within the same $300 ceiling.
One LoRA per condition, formal capability testing deferred as already agreed.

**Current state (2026-09-09):** The broader dataset is complete: 705 accepted
conversations, 684 selected into the frozen 9968-row mixture. Full token-length and
new-example mask checks passed; corpus and mixture are public under dougalldeepmind
with exact revisions in the [experiment card](experiment_card.md). The authorized
one-seed SFT is now starting on 2xH200 RunPod with dynamic batching; matched ODCV
follows. Total ceiling remains $300; no additional taste approval is needed.

The selected conversations include actual Opus4.8 and Sonnet5 authorship, all reviewed
independently by Sonnet5. One documented literal correction pass was permitted;
substantive errors and remaining defects were excluded. Original prompts are intact.
The earlier paired-arm and identical-answer constraints remain abandoned. Recipe and
model changes are provenance, not causal comparisons. Stakes work is currently paid-paused.
Earlier pilot decisions below are historical and do not override this current authorization.

**User requirement: preserve results locally before GPU teardown.** Normal teardown
must follow a verified local copy of necessary outputs, not merely a completion flag
or a successful HF push. ODCV transcripts, Docker logs, scores and metadata are
produced locally by the driver; all 720 completed rollouts were rechecked locally.
For future SFT, fetch full training output trees (adapters, tokenizer, resolved configs,
provenance, loss history, logs and saved resume checkpoints), verify the remote/local
archive hash and completed-arm identities, and record a local backup receipt first.
Failed downloads block ordinary teardown and retry within a reserved recovery window.
The independently enforced budget/lifetime ceiling remains an emergency limit; alert
explicitly if recovery cannot succeed before it. The retention requirement itself is not a launch authorization; the current broader
SFT has separate standing user authorization after its completed dataset checks.

**Earlier direction: user approved the bounded postmortem reset.** The
[postmortem and proposed reset](2026-09-08_postmortem.md) identifies the mismatch between
historical reasoned overrides and our new compliant paired controls. Tagged joint
drafting, one targeted correction and controls for accepted cases are now authorized
within the cumulative $3 pilot cap. Original prompts remain unchanged. Existing-model
evaluation preparation proceeds independently; rental/evaluation remain unlaunched.

**Reset outcome:** eight joint tagged drafts, seven one-time corrections, five controls,
three locally accepted complete B/C pairs. Cost $0.412058; shared ledger $0.599406,
all settled. No further correction loop. Accepted examples: drawing error, model handoff,
scheduling proposal. Controls total about 37% longer than comparative CoT, so length
remains a confound. See the [current card](experiment_card.md) for exact results and full
examples. The baseline lane has a separate three-checkpoint/720-rollout proposal; its
quote/spend-control preflight remains, and no evaluation is authorized or running.

Earlier pilot: **Authorized eight-task pilot stopped at answer gate: 1 pass, 5 failures,
1 unresolved, 1 generation failure. Cost $0.187348 across 12 requests; no traces or training.**
Start at the [short experiment card](experiment_card.md).
The tested recipe has eight shared answers, local review, then eight paired
traces with a cumulative $3 cap; the trace gate did not pass. Its inputs span UI copy, technical documentation and
one business proposal; 25 offline driver tests pass. The exact current config is
`configs/data/synth/nonmoral-paired-reuse.yaml`. This is a feasibility pilot, not approval
for training, evaluation or uploads. User's latest "go ahead" authorizes this exact
pilot under its $3 cap, with local answer review before generating paired reasoning.
No retry or revised paid recipe has launched. The failed config and all raw responses
are preserved in `output/nonmoral_paired_reuse_pilot/20260908_210135` and its parent.
V1 recorded
1/24 individual passes under its original review ($1.469684 exposure). The subsequent Sonnet calibration consumed
$0.210076 across 14 requests; its final six-fixture attempt matched only 1/6 provisional
expected labels. That is agreement with our labels, not established judge accuracy.
No fresh v2 scenarios were generated. No training or new ODCV evaluation ran.
See [pilot results](2026-09-08_pilot_results.md) and [v2 protocol/outcome](protocol_v2.md).
HF destination update from the user: **`dougalldeepmind`**, public artifacts. The old
publication attempt to `LASR-Callum` was denied (403); that does not establish access
to the new namespace. Historical source IDs are preserved until redirects/revisions
are verified. `.env` HF_ORG and the effective repository namespace resolver now return
`dougalldeepmind`. No new publication was attempted during this local review.
Started 2026-09-08. Full-session feasibility depends on the pilot.
Working branch: `codex/nonmoral-deliberation-controls`.

This is the canonical record of what the user wants the next research session to do.
The dated [research plan](2026-09-08_research_plan.md) contains the evidence review and
candidate experiments. A proposal there is not an agreed requirement here. Revise this
brief in place as decisions settle; do not create a new plan for each conversation turn.

The [prelaunch investigation](2026-09-08_investigation.md) now records verified HF
provenance, a holdout-split bug, evaluation drift, a proposed $290 stage envelope and
overnight prerequisites. [Six complete paired examples](paired_examples.md) make the
proposed CoT contrast reviewable. In the subsequent Sept 8 reply, the user accepted
testing this contrast, agreed to respecting valid hard user constraints, and tentatively
accepted the two-new-LoRA package plus reevaluation of the historical reference. They
asked what the experiment would deliver; this is not an instruction to launch paid work.
Detailed budget allocations and the evaluation/audit protocol remain proposed.

Earlier executable proposal (superseded): [protocol v1](protocol.md). The user's then-latest "go ahead"
authorized the local checks and protocol preparation offered immediately before it.
It did not authorize paid calls: the assistant explicitly retained that boundary.

Subsequent explicit approval: after the assistant explained the 24-pair pilot, its audit
and gates, $10 maximum, and exclusion of GPU/training work, the user replied "go ahead".
This authorizes that paid pilot and review. Full generation, training and ODCV remain
pending the pilot findings and a separate full-run decision. Protocol v1's original
awaiting-approval wording is retained as the frozen prelaunch record.

## Confirmed by the user

### Latest decisions: strict control, original result first

The user answered the four scope questions explicitly:

1. Remove comparison from **both CoT and the final assistant response** in the control.
   Retain substantive reasoning, a defensible decision and the full requested output.
   The paired comparative arm may compare in CoT; both arms share a comparison-free
   final answer. This concerns the experimental synthetic rows; preserve the common
   historical replay data rather than silently changing the entire training mixture.
2. First investigate the existing nonmoral result using reusable original tasks;
   then develop the broader improved recipe.
3. Start with a few clearly checkable domains. Record exclusions and limits on scope.
4. Next deliver verified baseline evidence, inspected reusable examples and a concrete
   experiment ready for review. **No paid generation, training or evaluation.**

This supersedes the earlier allowance for comparative rationale in the common final
answer and the original broad twelve-domain pilot as the immediate path. Prompts that
require a comparative final explanation cannot enter the strict pair by dropping that
requirement. Original data and pilot results stay intact. Free retrieval of existing
public artifacts supports provenance work; it is not a new model evaluation.

The prior workstreams are continuing in parallel on these bounded deliverables.

Subsequent clarification: keep every original prompt unchanged and exclude incomplete
cases. Do not supply missing code or otherwise repair prompts. Correcting a final
answer identically in both arms is allowed and must be recorded.

### Completed local deep work

- Retrieved existing public math/nonmoral artifacts with pinned revisions and verified
  namespace redirects. Nonmoral submission accounting is **398/400 = 99.5%**, using
  the historical marker rule; absent markers are in the two reconstructed transcripts.
  This is submission evidence, not proof of task success; progress remains ungraded.
- Math reference is **98/240 = 40.83%**. Same cells/base revision, but 28,000 context
  and a newer transcript-budget harness versus nonmoral's 16,384: no matched effect
  estimate or ranking follows from that difference.
- A frozen 12-case review of actual original training rows found 1 ready prompt,
  2 conditional and 9 excluded; zero final answers reusable unchanged. A separate
  additional 12-prompt UI screen found 2 more candidates. Keep denominators and review
  depth separate; these are selected-domain diagnostics, not corpus error-rate estimates.
- Three strict UI pairs are now available as locally checked illustrations: drawing
  errors, shape tooltips, dashboard tooltips. Original prompts match training bytes;
  both arms have the same corrected complete final answer. Their CoT lengths are
  91/90, 95/90 and 87/100 tokens, without padding. This is feasibility evidence, not
  a ready training corpus or demonstrated multi-domain coverage.
- [Experiment card](experiment_card.md) supplies the question, fixed/changed fields,
  remaining preparation, validity limits and oversight point. No new paid requests,
  GPU rentals, training, evaluations or uploads occurred in this work block.

### Scope checkpoint after the judge calibration

The user explicitly asked to pause and reconnect this work to baseline, math, naturally
short reasoning, no-deliberation and stakes experiments, questioning sequential work
and time use. Further API work is paused. A proposed next step is parallel **design and
existing-artifact analysis**, not parallel paid runs or another judge-tuning loop.

The paired comparative-versus-verification pilot addresses only the contribution of
alternative-comparison in CoT. It is not a complete no-deliberation baseline because
the shared answer can still deliberate. Inclusion of math among twelve domains is not
a math-only ablation; concise trace prompts are not a short-reasoning experiment.
Neither stakes nor alignment effects have been tested by the current pilot.

The user's subsequent "ok do all 3 in parallel" authorizes these independent local
work packages, with separate agents and files. It does not restart paid work:

1. Baselines and existing artifacts: identify exact reference models, training mixtures,
   and comparable evaluation results; assess reuse before generating replacements.
2. Mechanism controls: make concrete examples and confound tables for comparative,
   non-comparative verification, naturally short reasoning and math-only conditions.
   Keep final-answer deliberation explicit; keep masking experiments distinct.
3. Data validity: inspect existing or locally authored examples using task constraints
   and executable checks where possible. Judge output is advisory until trustworthy;
   stop treating six-fixture prompt tuning as a prerequisite for every research branch.

Merge these into one small experiment matrix and spending decision before further paid
work. Preserve the later stakes workstream and the user's predicted null effect.
All three agents completed their bounded reviews. No paid experiments were dispatched.

### Integrated findings and recommended sequence

| Workstream | Delivered | Decision-relevant finding |
|---|---|---|
| Baselines | [Artifact inventory](baseline_inventory.md) | Exact historical 684 synthetic + 9,284 replay rows and a Numina math-control adapter exist. Cached nonmoral MR is 73/400 (18.25%); principle-scoped DA is 43/400 (10.75%) under matching recorded evaluation settings. This is a descriptive checkpoint comparison. The old 320-rollout submission summary is superseded locally by the recovered 398/400 count; progress remains missing. |
| Ablations | [Experiment matrix and complete examples](ablation_matrix.md) | Explicit comparison in CoT, mathematical reasoning, naturally shorter tasks and stakes are distinct interventions. The shared final answer determines how narrow a no-deliberation claim must be. |
| Validity | [Independent review](data_validity_review.md), `scratch/nonmoral/validity_checks.py` | All three seeded grounding defects are confirmed; the judge accepts all three. Aggregate 1/6 agreement also depends on disputed construct labels. The mathematical proof is correct but its comparison overstates what competing methods require. Local checks passed; originals remain unchanged. |

The previously recommended math-provenance, submission-accounting and small-sample
reuse work is complete above. Next review the actual strict examples; then establish
the eligible original-prompt pool, exact row count and held-out families before a paid
proposal. Do not inspect ODCV content to choose training examples.

Then choose one new training contrast: one rewritten-CoT LoRA against the old checkpoint
is an inexpensive exploratory package comparison if the exact original rows/answers
remain usable. Two newly authored matched CoT conditions reduce writer/style asymmetry
and permit common exclusions or answer corrections. Recommend the latter when the
claim is about the comparative-reasoning ingredient. Both use one seed per condition.
Neither requires first generating a new twelve-domain corpus; broader recipe development
remains a separate workstream, with its own usefulness criterion.

Existing math supplies a category baseline without new training. Naturally short tasks
come next as a complexity/extent intervention, not a pure length effect. Keep stakes
deferred with the user's predicted null. No full factorial is proposed. Exact training
or evaluation launch scope remains undecided; no automatic restart of the failed pilot.

The user is currently available for questions and requests occasional extremely brief
updates from each workstream. Ask only when an unresolved choice materially changes
the experiment; report completed work and concrete uncertainties, not generic plans.

- Use Sonnet for new scenario generation; drop Haiku and the proposed Haiku/Sonnet
  comparison. Every extra judgement, rewrite and reasoning pass has a cost, which must
  be included explicitly rather than treated as free validation.
- Communication should be short, high-signal and direct. Lead with the answer; put
  supporting detail in artifacts. Surface meaningful findings, decisions and blockers.
- First experiment direction accepted: comparative CoT versus execution/verification CoT
  with the same prompts and full, comparison-free final answers. The control has no
  explicit comparison in either assistant-text region. This varies supervised text;
  it cannot establish the absence of internal deliberation by the trained model.
- Respect valid explicit hard constraints. Departures require genuine contradiction or
  impossibility, or discretion explicitly granted by the prompt; better craft alone is
  insufficient reason to override. Do not perpetuate systematic override as a goal.
- The first package is tentatively two paired new datasets/LoRAs plus reevaluation of the
  original nonmoral checkpoint. Shorter reasoning, stakes and new math training follow
  later. One LoRA per new condition remains the rule.
- Extend the project's nonmoral-deliberation work with more defensible dataset selection
  and useful baselines/ablations. The unexpected alignment improvement needs explanation.
- The desired outcome combines all three: make nonmoral deliberation as effective as we
  can, understand which ingredients matter, and have confidence in the results. The user
  tentatively favours more baselines as the first step, not as a replacement for the other
  objectives. The exact sequence and resource allocation remain open.
- Reporting should centre on concrete charts, clear numbers, exact observed results,
  precise answers, and specific questions for further work. Long, generic prose is an
  explicit failure mode. Keep the user-facing synthesis short and evidence accessible.
- Improving alignment is desirable, but do not directly optimize selection against ODCV.
- Decreased capabilities or increased over-refusal disqualify a candidate. The user
  accepts no trade-off on either. This is a substantive constraint, not a positive
  regression allowance to select later. Relevant benchmarks, comparator, precision and
  decision rules remain to be specified; inconclusive preservation evidence is not a pass.
- Capability tests must NOT run without the user yet; they are deferred to a later joint
  stage. For now apply common-sense dataset quality checks to avoid training undesirable
  behaviour. ODCV task completion is the team's current rough proxy for over-refusal.
  Preserve this scope: do not add capability or dedicated over-refusal evaluations to the
  overnight package. Neither data review nor task completion proves capability preservation
  or measures over-refusal directly; qualify any eventual best-recipe claim accordingly.
- There are no additional nonmoral training-seed results beyond the known checkpoint.
- For now, train one LoRA per experimental condition, with no training-seed replication.
  This interprets the user's "always just do 1 LoRA for now" in the discussion of
  three seeds per condition. It does not authorize any run now. Repeated ODCV rollouts
  remain a separate design choice; report that training-seed variability is unmeasured.
  Earlier proposals to reserve funding for multiple SFT seeds are deferred, not launch
  requirements for the current phase.
- Broader nonmoral judgement is allowed; justified override need not define every example.
- The user agrees to preserve the original nine craft preferences as the foundation of
  the historical arm, while allowing the expanded corpus additional explicitly documented
  tensions. Do not force the broader domains into the original nine categories. A shared
  quality rubric should require relevant considerations, defensible reasoning, a decision
  and full execution across domains. Treat the expanded corpus as a documented new recipe,
  not a reproduction of the historical arm or a clean one-variable comparison against it.
- What constitutes a good nonmoral-deliberation example is still highly uncertain. Spend
  time brainstorming this jointly before settling the construct or the experiment matrix.
- The user likes all four initial example families: weighing competing qualities,
  choosing under practical uncertainty, reconsidering an initial approach, and aesthetic
  judgement. They want a broader exploration beyond these, not a corpus restricted to them.
- The user also accepts all twelve families in the broader brainstorming slate as
  candidates, provided they avoid moral deliberation. Acceptance of the domains does not
  establish that every generated example in them is nonmoral or approve generation runs.
- Morality and stakes are separate axes. In the user's terms, morality asks whether it
  is wrong to pick an option; stakes ask what can be lost by picking an option. Do not
  treat potential loss, substantial consequences, or personal cost as sufficient evidence
  of moral deliberation. Conversely, small consequences do not make an ethical conflict
  nonmoral. The present task is to ensure the situation and deliberation are nonmoral,
  not to enforce low stakes. Borderline-case adjudication still needs concrete examples.
- Cases where competent people could reasonably choose differently are acceptable,
  provided the selected option is reasonably argued for. Do not inherit the old recipe's
  requirement that practitioners agree on one choice or that the instruction must be wrong.
  This permits pluralism, not unsupported assertions or arguments that ignore constraints.
- Conversations must be single-turn: one user request and one assistant response, with
  an ordinary system prompt permitted and CoT represented within that assistant turn.
  Do not introduce follow-up user turns to resolve ambiguity. Every example must end in
  a decision rather than only discussion or a request for clarification.
- The assistant must provide the FULL requested output. A selected approach plus a
  demonstration fragment is insufficient when the user asked for a complete artifact.
  Size tasks to permit complete single-turn answers rather than silently truncating the
  deliverable. Preserve completeness across experimental variants.
- For genuinely ambiguous cases, the user tentatively permits a conditional decision:
  if the missing preference/condition is one way, do X; otherwise do Y. This is subject
  to preserving the experiment's purpose, not blanket permission for vague hedging.
  Its frequency and acceptance criteria remain to be settled.
- Keep explicit track of what changes and why, and actively prevent implementation sprawl.
- The next reasoning-content control is distinct from the already-tested CoT-masked,
  empty-CoT/empty-marker-masked, and CoT-only supervision conditions. Retaining CoT while
  changing deliberation is a candidate; its operational definition remains unsettled.
- Math and naturally shorter reasoning remain candidate directions, not an approved
  list of experiments or a settled order.
- A low-versus-high-stakes experiment within nonmoral deliberation is explicitly wanted
  later. Defer its design/execution now; focus first on nonmoral validity. On 2026-09-08
  the user recorded the prediction that stakes will not matter, while explicitly allowing
  that this could be wrong. Preserve this prospective prediction; an exact endpoint,
  equivalence criterion, sample size and analysis plan have not yet been registered.
- Spend time extracting intentions, constraints, success criteria, tests and health checks
  before a future multi-hour autonomous session, potentially involving parallel agents.
- Continue asking questions across rounds. Ask in ordinary chat: the earlier question
  widgets disappeared for the user, so do not rely on them to carry the conversation.
- Budget: plan below a provisional USD 300 ceiling across the first autonomous session.
  Minimize spend. The user recalls roughly USD 100–150 for generation, USD 20 per SFT,
  and roughly USD 20 per ODCV run; these are planning inputs, not current verified prices
  or a specification of pass counts. More spending can be justified by sufficient useful
  experiments, but requires a new explicit budget decision. No automatic increase.
- An informative null is an acceptable outcome if the process is legitimate. Establish
  and check manipulation validity, controls, provenance and analysis before treating a
  null as a scientific answer; an invalid or inadequately resolved comparison is not one.
- Datasets, LoRA adapters, ODCV results and comparable research artifacts go to Hugging
  Face publicly under `dougalldeepmind` (user update supersedes `LASR-Callum`). This settles destination and visibility for authorized
  future research work; it does not start generation/training/evaluation now. Credentials
  and other secrets never belong in those uploads. Git merge/publishing boundaries were
  proposed but not explicitly settled in this reply.
- Availability: on 2026-09-08 the user expects to be available for another 4–5 hours,
  then wants to launch an overnight session around 19:00 Europe/London, generally without
  questions requiring an answer before morning. These are approximate planning targets,
  not an automation already created or authorization to start the run now. They can
  check remotely for exceptionally urgent issues.
- Decision fallback: after asking a question, if the user has not responded within
  30 minutes, choose the best available option and proceed when the decision can be
  made within the agreed authorization, budget and research constraints. Record the
  choice and reason. This is explicit authorization to exercise judgement within those
  boundaries, not permission to exceed them or treat silence as approval for new scope.
  If no valid choice exists, park the affected work and continue independent tasks where
  possible. During the wait, continue useful work that does not depend on the answer.
- For now, discuss and prepare locally. No GPU rental, paid inference, or autonomous launch
  has been authorized. A provisional ceiling and decision fallback are now agreed;
  the costed work package and launch instruction remain outstanding.

## Decisions to elicit

| Area | What must become clear | Status |
|---|---|---|
| Scientific priority | Improve the recipe, identify important ingredients, and establish trustworthy results; baselines tentatively first | Goals confirmed; ordering provisional |
| Research construct | Broad candidate families accepted; reasonable disagreement allowed; morality distinct from stakes | Principles confirmed; case-level rubric open |
| Initial scope | Required workstreams versus optional follow-ups and exclusions | Open |
| End state | Charts, exact results, precise answers and specific next questions; minimal prose | Format confirmed; deliverable set open |
| Evidence standard | Claims sought, important effect sizes, acceptable uncertainty, handling of nulls | Open |
| Null outcomes | Valid, informative null is acceptable; legitimacy of the process is essential | Confirmed; exact evidence thresholds open |
| Disqualifying effects | No capability loss or increased over-refusal accepted; capability tests deferred until user involvement; ODCV completion used as rough proxy for now | Constraint confirmed; broader validation is later |
| Dataset policy | Broad domains; judge moral content independently of stakes | Coverage and case-level acceptance rubric open |
| Source preferences | Preserve historical nine; document additional tensions for expanded corpus; shared quality rubric | Agreed; actual tension set and rubric need validation |
| Conversation structure | Single-turn; decision and full requested output | Confirmed; conditional decisions tentatively allowed |
| Later stakes experiment | Vary stakes within nonmoral deliberation; predicted no effect | Required follow-up, deferred; formal protocol open |
| Training replication | One LoRA/training seed per condition for now | Confirmed in context; ODCV pass count remains open |
| Resources | Plan below USD 300; minimize cost; increase only by explicit decision | Ceiling provisional; costed allocation/concurrency open |
| Autonomy | Ask; after 30 minutes without reply choose within agreed boundaries; otherwise park affected work | Fallback agreed; urgent triggers and remaining permissions open |
| Collaboration | Agent roles, review independence, shared-file ownership and integration | Open |
| Publishing | Public research artifacts on Hugging Face under dougalldeepmind | Updated by user; historical source IDs preserved; new namespace access not yet checked |
| Communication | User available initially; overnight from around 19:00 London; urgent remote access possible | Overnight update cadence/urgent triggers open |

## Proposed translation into measurable deliverables

The following operationalizes the user's output preferences; the specific charts and
statistical thresholds will be agreed with the experiment design.

- A compact comparison table with exact counts/denominators, per-seed results, aggregate
  effects and uncertainty; identify the dataset, model and evaluation revisions.
- Actual chart files accompanied by the underlying data and regeneration commands.
  Show replication spread and uncertainty rather than only a winning point estimate.
- A short answer to each research question, distinguishing supported, unsupported and
  unresolved conclusions, with the evidence that determines that classification.
- The strongest validated recipe among those tested, with its scope and remaining
  uncertainty stated. Do not claim global optimality from a finite experiment set.
- A small, ranked list of further experiments, each tied to a specific unresolved
  observation and explaining what the experiment could distinguish.

“Exact results” means exact observed measurements and provenance, not certainty about
the true population effect. More output volume, many agent-hours, or a low unreplicated
ODCV score are not substitutes for these deliverables.

## Proposed budget discipline and overnight continuity

Illustrative arithmetic, NOT an agreed work package or measured estimate: the assistant's
candidate 2×2 has four datasets/conditions (short/long comparative reasoning and short/long
procedural reasoning). Training each with three independent SFT seeds would make 12
adapters. Assuming USD 20 to train EACH adapter and USD 20 to evaluate EACH adapter yields
USD 240 training + USD 240 ODCV = USD 480 before generation. The per-adapter eval assumption
may not match what the user's historical USD 20 quote covered; pass count and serving/judge
cost must be established. Three SFT seeds are not three sampled rollout passes of one
adapter. Neither the four-condition grid nor exhaustive replication has been approved.
Do not use this hypothetical figure as a quote or a reason to increase the budget.
The user subsequently directed one LoRA per condition for now, so the ×3 training-seed
factor does not apply to the current phase. Cost the actual
protocol, inventory reusable artifacts, and reserve evaluation/contingency funding before
committing to corpus generation. Do not promise conclusive answers to all questions in
one budget. Numerical stage allocations remain to be agreed from estimates.

Proposed accounting: count accrued spend, active billing exposure and estimated remaining
cost of committed jobs before launching another stage; keep a contingency allowance and
stop launching early enough that cleanup stays within the ceiling. Allocate a shared
budget centrally so parallel workers cannot each spend the entire allowance. Do not
charge unrelated users' existing resources to this session or terminate them.

Proposed continuity: use a durable task/dependency ledger, artifact paths and revisions,
cached/resumable stages, timestamps for pending questions, recorded fallback decisions,
and verified cleanup of owned resources. An individual worker stopping must not erase
the scientific record or leave rented GPUs billing without supervision. These mechanisms
need to be prepared and checked before overnight launch; writing this plan does not
create them. Completion should stop the session rather than filling the night with
unnecessary work.

## Broader brainstorming slate — accepted candidate families, not a corpus specification

The following expands beyond the first four families. The user accepts all as candidate
directions if moral deliberation can be avoided. They are not final domain quotas,
separate experimental arms, or a claim that each isolates a distinct cognitive mechanism.

| Candidate | Illustrative situation | What might be deliberated |
|---|---|---|
| Everyday planning | Plan a free afternoon around several enjoyable activities | Variety, immersion, energy and transitions |
| Cooking | Adapt a dish to ingredients already available | Preserve its defining texture/flavour or create a coherent different dish |
| Learning strategy | Practise a difficult musical passage | Isolated technique versus whole-piece fluency and motivation |
| Teaching | Introduce an unfamiliar concept to a specified audience | Concrete intuition versus formal structure; what to explain first |
| Information gathering | Diagnose a colour problem in a hobby image editor | Which check is worth doing before committing to a fix |
| Scientific judgement | Explore a toy simulation's parameter space | Broad coverage versus resolving one promising region carefully |
| Representation | Organize personal notes that span several subjects | Categories, tags and cross-links; retrieval versus maintenance |
| Translation | Translate a playful line with wordplay | Literal meaning, tone, rhythm and the function of the joke |
| Spatial design | Arrange a small hobby workspace | Light, usable surfaces, movement and storage access |
| Stopping/revision | Revise an essay that is already coherent | Which edits add value and when further revision erodes its character |
| Revisiting assumptions | Reconsider a plan to automate a small recurring task | Whether the task is stable enough to justify the proposed automation |
| Mathematical strategy | Choose a route through a nontrivial proof problem | When to seek a construction, exploit symmetry or change representation |

Additional candidate raised by the user: choosing a move in chess or another game.
Proposed inclusion is conditional on the actual reasoning: weighing plans, initiative,
position, resources or uncertainty can instantiate nonmoral deliberation. Ordinary
stakes are position, points or winning/losing the game; later versions could vary their
magnitude without making the choice morally wrong. Mechanical or forced-tactic tasks
can supply contrast examples but are not non-deliberative merely because the move has
an objectively correct answer. Use a complete supplied position/rules/information state,
validate legality and factual claims, and require a concrete move/action as the full
single-turn deliverable. Candidate status only; no game corpus or separate arm approved.

Some candidates involve instrumental reasoning as well as judgement. Do not classify
all math/debugging as non-deliberative by domain alone: these boundary cases help test
the definition. Everyday personal goals also broaden the old artifact-only constraint.
Moral wrongness and magnitude of loss must be assessed separately, following the user's
definition above; concretely classifying borderline situations remains part of the pilot.

### Proposed decision-completion check

An unconditional decision is the default: the assistant selects an approach and explains
why it is defensible from the supplied facts, goals and constraints. A conditional
decision must name the genuinely missing condition and specify an actionable choice
for each relevant branch within the same response. Merely listing pros/cons, saying
“it depends,” or asking a question and waiting is not completion.

The response must also deliver the complete requested work. Choosing a structure for
an essay does not complete a request to write the essay. A complete plan does complete
a request for a plan. Conditional decisions must not conceal missing deliverables;
select tasks where the necessary conditional output can be provided within one turn.

Do not invent a user preference to force certainty, or manufacture ambiguity to justify
a conditional answer. Tag conditional examples separately and inspect their prevalence
and reasoning; the permitted share is not yet set. If all variants of a scenario are
used in a paired ablation, preserve its decision structure across the variants.
Conditional responses could train excessive caveating, so their effects on decisiveness
and task completion should be checked rather than presumed harmless.

## Proposed working method — not yet an approved experiment plan

Post-pilot model/cost proposal (2026-09-08): Sonnet 5 generates scenarios and serves as
the primary dataset-quality judge. The original nonmoral scenario/manipulation audit
scripts already use Sonnet 5, as do configured DA/PAR/PC quality checks. Flash's use in
recent ODCV runs does not establish its validity for this different audit. No blanket
change to benchmark judges is proposed; those choices affect historical comparability.

Use mechanical checks before further generation and one Sonnet audit per completed
pair, with separate quality and contrast fields in the same call. Do not add automatic
scenario-judge, answer-judge, trace-judge and rewrite/rejudge loops on every candidate.
Any calibration or repair pass is a separately counted, bounded cost. At the completed
pilot audit's exact token volume, verified provider rates imply $0.10361 for 15 Sonnet
audits versus $0.028055 for Flash, or $4.724616 per 684 pairs with identical token use.
This excludes extra reasoning, retries, rejected candidates and repairs; it is a
token-volume illustration, not a full-corpus cost forecast. Existing synth configs
also document explicit hidden-reasoning controls/output headroom: verify and budget
these when building v2. This inspection made no paid model calls.

Elicit requirements in small rounds, summarizing the resulting decisions after each.
Use concrete examples and counterexamples to resolve scientific ambiguity rather than
agreeing on labels that different agents might interpret differently. Before launch,
walk through hypothetical outcomes and failures to verify that the planned response
matches the user's wishes.

The current 2×2 proposal (comparative/procedural reasoning × short/long) is a candidate,
not the default merely because the assistant wrote it down. Its validity must be examined
alongside the user's other possible baselines. Do not begin by treating it as settled.

For each selected workstream, fill one compact task specification:

| Field | Required content |
|---|---|
| Purpose | Scientific question, why it matters, and why this task can address it |
| Inputs | Exact source artifacts/revisions and dependencies on other workstreams |
| Intervention | What changes and what must stay fixed |
| Deliverable | Concrete files, datasets, code, results or recommendations |
| Completion test | Observable acceptance criteria, including human judgement where needed |
| Validity checks | Leakage, confounds, correctness, uncertainty and missing-data checks |
| Limits | Compute/spend/time/retry limits and forbidden shortcuts |
| Failure response | Repair, stop, ask, or continue on another independent task |
| Ownership | Agent role, owned files, reviewer and integration point |

A completed task and a supported hypothesis are different outcomes. An adequately tested
null may complete a task; a lower ODCV number alone does not validate a mechanism or a
selection policy. Exact evidential criteria remain to be agreed rather than invented.

## Checks to specify before a long run

These are categories for the discussion, not claims that numeric thresholds are settled.

- **Data validity:** split integrity, genuine reasoning-content differences, moral-content
  boundary, factual correctness, reasoning/action consistency, coverage and attrition.
- **Training validity:** data/render/mask provenance, supervision and length accounting,
  matched recipe, seed replication and healthy optimization.
- **Evaluation validity:** frozen comparable ODCV protocol, task completion/progress as
  well as MR, truncation/failure accounting, calibrated judging and uncertainty at the
  correct level. Capability tests remain outside the present authorized scope.
- **Operational health:** spend against budget, owned GPU status, stalled processes,
  retry storms, durable checkpoints, recoverability and eventual teardown.
- **Research discipline:** no selection tuning on ODCV outcomes; label exploratory changes;
  do not reinterpret a failed manipulation as a negative scientific result.
- **Integration:** independently review consequential conclusions and changes, prevent
  overlapping edits, and verify the combined deliverables rather than only separate tasks.

## Launch readiness

Before dispatch, the brief should answer: what is the priority; what precisely will be
done; what counts as valid completion; what may be decided autonomously; how resources
are bounded; and what happens on failure or uncertainty. No unresolved item should be
hidden behind “use best judgement” when the answer could change the scientific question
or exceed the agreed scope. Routine implementation choices can remain delegated.

The user's future instruction to start will activate the agreed scope. This brief does
not itself schedule a task, start an agent, rent infrastructure or authorize spending.
