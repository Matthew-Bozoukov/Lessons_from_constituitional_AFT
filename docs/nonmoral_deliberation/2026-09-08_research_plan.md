<!-- ABOUTME: Codex's evidence review and proposed nonmoral-deliberation experiments, before new runs. -->
<!-- ABOUTME: Separates local observations, historical results, hypotheses, and decisions still open. -->

# Nonmoral deliberation: understanding and proposed next experiments

Written 2026-09-08. Discussion draft, not a registered protocol or authorization to run.

Updated after discussion: the user confirmed there are no additional nonmoral training
seeds, permitted broader judgement tasks, and clarified that the new control should retain
CoT while varying deliberative content. CoT masking, empty-CoT, and CoT-only supervision
have already been tested and are not the proposed experiment. Section 5 replaces the
initial suggestion to repeat those interventions.

Subsequent elicitation is authoritative in [research_brief.md](research_brief.md).
The user requires full single-turn deliverables, accepts reasonable disagreement, and
distinguishes moral wrongness from stakes (what can be lost). A stakes ablation is wanted
later, with a prospective prediction of no effect; it is deferred during the present
nonmoral-validity work. These decisions supersede narrower initial suggestions below.

## 1. What we are trying to learn

The project is improving constitutional SFT methods: identify which properties of synthetic
training data change model behaviour, then build a recipe from measured effects. Difficult
advice is the empirical reference. The causal chain is generation procedure → properties of
the resulting data → learned behaviour. Changing a generator instruction does not establish
that only the intended property changed.

Here the question is whether the difficult-advice effect requires moral content, or whether
some more general habit of reasoning and deciding transfers. Lower ODCV misalignment would
be useful evidence, but the selection rule must be justified without asking which examples
would improve ODCV. A null result from a cleaner dataset would still answer the question.

The existing nonmoral arm is more specific than its name: it asks the assistant to produce
an artifact, confront a bad binary instruction, weigh competing craft considerations, and
commit to a justified departure. It changes morality, domain, actor, response form, and
generation samples together relative to difficult advice. It is an informative intervention,
but not a paired test of morality alone.

I would operationalize deliberation as comparing viable options using situation-specific
considerations and explaining why those considerations favour the chosen action. The
handoff's distinction between instrumental CoT and judgement under incommensurables is a
useful hypothesis, not an established partition of reasoning. In particular, the recipe
also demands that competent practitioners agree and that the facts decide. That can
select fairly straightforward technical corrections despite the intended judgement task.

## 2. Evidence and provenance checked

Initial review used main `732c8b4c` and the then-unmerged nonmoral branch at `c984e2cf`.
After the user's merge, `git pull --ff-only origin main` advanced main to `42b6acff`;
`c984e2cf` is a verified ancestor. Work now uses branch
`codex/nonmoral-deliberation-controls`, created from that updated main. The recipe is
`configs/data/synth/nonmoral-deliberation.yaml`; the craft spec and scratch tools are
also present in main. Claude's handoff and inspected data remain under
`.claude/worktrees/nonmoral-deliberation/`. The prior draft was preserved through the pull.
No source-code edit was performed.

Later Sept 8 location update: the handoff and corpus are now under root `output/`;
the old worktree paths above are historical. The pinned HF training mixture was retrieved
again and its SHA256 matched the earlier inspection. See the
[prelaunch investigation](2026-09-08_investigation.md) for current evidence and proposals.

Read: root `CLAUDE.md`, README, `docs/BASELINES.md`, relevant LOG and TODO entries,
`docs/error_bars.md`, `docs/corpus_checks.md`, synth architecture, the nonmoral recipe,
craft spec/rationale, training config, mixture publisher, manipulation-check code,
representative corpus rows, and current masking/loss code. Historical low-stakes and
verbose-CoT work also informs the proposal.

| Record | What it supports | Limitation |
|---|---|---|
| Nonmoral original evaluation | 25.0%, 56 variant-balanced cells, one pass/checkpoint, two judges | Older protocol and broad uncertainty; not the current headline |
| Main LOG, 2026-09-04 nonmoral entry | 73/400 = 18.25% (reported 18.2%); 80 cells × 5 passes; temperature 0.7; single Gemini Flash judge | Identifies seed 0, not a three-training-seed result; two transcripts reconstructed |
| Same entry's intervals | Scenario-sampled 95% CI [11.1, 28.5]; fixed-benchmark [15.6, 21.3] | Neither measures variation across unobserved training seeds |
| Principle-scoped DA, newer LOG entries | 10.8% over five passes; measured progress near ceiling | Historical 11.5%, da716 16.3%, etc. are different records, not interchangeable comparators |
| Prior moral low/high-stakes work | No detected separation; low-stakes seed 0 and 80085 gave 16.9% and 10.8% in the recorded one-pass protocol | Consistent with similar alignment, not a statistical demonstration of equivalence |

The 18.2% result is promising relative to historical controls, but does not establish a
morality-specific causal effect or equivalence to DA. The user confirmed there are no
additional nonmoral training-seed results. Five sampled passes of one adapter do not substitute
for five independently trained adapters. Corpus-generation randomness is a third source
of variation, also not covered by repeating one frozen corpus across training seeds.

The current stats documentation supports pooling repeated rollouts within checkpoint and
keeping checkpoints as their own axis. Its protocol is newer than BASELINES.md's
one-pass/shared-cell advice. Resolve comparisons from saved run metadata and current
statistics code; do not silently discard data or inherit an older denominator. The pull
also changed ODCV's configured context window from 16,384 to 28,000 and its transcript
budget handling. A fresh result under current defaults is not directly comparable with
the historical 18.2% without accounting for those changes. Evaluate comparison arms
under one frozen protocol, or explicitly label historical comparisons descriptive.

The user-supplied Hugging Face links were attempted read-only. The browser could read the
[Qwen model card](https://huggingface.co/Qwen/Qwen3.6-27B), which identifies a **post-trained**
checkpoint. Thus “base” here means our starting model before project SFT. The LASR artifact
pages did not resolve through that browser, so their live revisions/results were not
independently verified. Numerical claims above come from local records, not freshly
downloaded Hub results. The initial main LOG contained committed conflict markers near
its September 5 entries; that observation belongs to the pre-pull checkout. Treat logs
as an index to underlying artifacts, not unquestionable ground truth.

## 3. Concrete dataset findings

The local corpus contains 702 rows, with trait counts 80, 79, 77, 78, 78, 79, 77, 78, 76.
The trained mixture contains 9,284 Table2 rows plus a balanced 684-row draw (76 per trait).
This is 6.86% synthetic by row count, versus 7.03% for principle-scoped DA. The other 18
corpus rows are unused, not a deliberately designed validation set.

**The unrun manipulation check has a real split bug.** It reads training IDs from
`metadata.scenario_id`; the actual rendered mixture stores them at top-level
`scenario_id`. It therefore reads zero trained IDs and admits all 702 rows. Of its default
first 30 examples, 29 are trained examples. Correct exclusion leaves only 18, fewer than
its default requested 30. This was checked by parsing the local JSONL files without
running the script or calling a model. The handoff says the check was never run: this is
not evidence of contamination in a reported evaluation.

**The intended behaviour is not guaranteed in the finished rows.** For example,
`t4_b00_s000` has metadata saying the instruction to use `MigrationLogger` is the worse
choice, but the answer explicitly uses it and reasons that consistency is preferable.
The prompt does not establish the logger defect presumed by that metadata. This may be a
reasonable answer to an underdetermined request; it is not the advertised override
example. In `t1_b00_s000`, the metadata rejects omitting POST status, while the final
request expressly delegates status to a URL and the response accepts the POST schema,
then objects to the GET schemas. Prompt–metadata–reasoning–answer agreement needs auditing.
These are inspected examples, not an estimate of defect prevalence.

**The finished-corpus semantic quality filter is disabled.** Pattern checking is enabled,
but it measures recurrence, not whether an instruction is actually bad. Its reported
99% broad / 95% strict recurring pattern is a classifier result, not a human-verified
override rate. Moreover, the current quality-check text mapping includes assistant
reasoning and response only. Simply enabling it would omit the actual user/system
request needed to judge correctness and grounding.

The handoff treats firmness and templating as inseparable. The pilots show that the
particular prompts traded them off; they do not show separation is impossible. We can
specify the required action and independently vary how it is expressed. That requires
checking the artifact, not merely scanning for forceful language.

## 4. Selection that does not use ODCV as a target

First retain the original corpus/mixture unchanged as the historical reference. Build
a separate, versioned annotation ledger over complete transcripts. Keep the generator's
`why_wrong` hidden during independent initial judgement so it cannot serve as an answer key.

Proposed annotation dimensions:

1. **Nonmoral content:** do the options differ in deception, harm, coercion, unfairness,
   safety, or other moral considerations? Distinguish ordinary artifact incorrectness from
   deceiving a person. Keywords alone are not a valid screen.
2. **Decision validity:** what options are feasible; which are defensible from facts actually
   supplied; is the instruction genuinely worse, or merely different from the generator's
   preference? Allow uncertainty and legitimate disagreement.
3. **Deliberation:** are alternatives taken seriously, with identifiable case-specific
   trade-offs? Would changing a deciding fact change the recommendation? Long text is not
   evidence of more deliberation.
4. **Consistency and execution:** do metadata, prompt, trace, answer, and artifact agree;
   does the artifact enact the stated choice; are technical claims supported?
5. **Coverage and form:** tension, domain family, deciding consideration, decision type,
   reasoning/answer length, duplicated scenarios, and repeated rhetorical structure.

Before automated selection, make a small human-labelled calibration set containing clear
passes, clear defects, and borderline cases across all nine tensions. Judge validation
must inspect both accepted and rejected rows and check individual failure categories;
an attractive aggregate keep rate is insufficient. Freeze the rubric and a separate
validation split before scoring the full pool. This would involve future paid calls,
after discussion, not in this turn.

Use hard exclusion for confirmed invalid/moral/contradictory examples. Use prespecified
stratification and seeded random sampling among eligible examples for coverage. Avoid
“top deliberation score” selection: it can reward verbosity or the judge's preferred prose.
Record rejection reasons, shortages and replacements. Keep all variants of a scenario
and its close paraphrases in the same split. Reserve a fresh validation set by scenario
family; the 18 leftovers cannot establish generalization.

Freeze selection, arm definitions, seeds, comparisons and evaluation protocol before
seeing new ODCV outcomes. No ODCV scenarios, failure analyses, judge rubric, reward-hacking
themes, or model-specific ODCV scores enter generation or selection. The separate `dat`
and reward-hacking projects explicitly use ODCV-derived structure; they are not parents
for this experiment. Since we already know aggregate ODCV results, this is prospective
separation, not a claim that ODCV has never influenced the research question.

## 5. The experiment now proposed: reasoning content, with CoT present

The prior three supervision studies answer where the training signal lives. This study
asks what kind of reasoning is useful within that signal. Every proposed arm retains
nonempty, supervised CoT and supervises the visible response with the same masking rules.

Operational contrast:

- **Comparison and choice:** work through at least two viable approaches, identify the
  considerations pulling each way, and resolve the trade-off using the case's facts.
- **Implementation and verification:** work through constructing and checking the chosen
  artifact, using real constraints and technical steps, without surveying competing
  approaches or weighing which preference should win.

The second is substantive reasoning, not an empty trace, arbitrary padding, unsupported
assertion, or cosmetic deletion of “however.” It may implicitly encode a judgement;
the claim is reduced explicit comparative deliberation, not absence of internal decision
making. An assistant's written CoT cannot establish the latter.

For illustration, on a glossary task both arms could return the same entry with an exact
technical term and a plain-language definition. Comparative reasoning weighs ease of
reading against reliable lookup and decides how to serve this audience. Procedural
reasoning constructs the entry, checks terminology against the supplied source, and
verifies the definition and cross-references. That example is a proposed contrast, not
evidence that such rewrites will be natural or length-matchable across the whole corpus.

The first design to validate is a small 2×2:

| Reasoning content | Short | Longer |
|---|---|---|
| Comparison and choice | Concise, genuine weighing | More developed weighing |
| Implementation and verification | Concise construction/checking | More developed construction/checking |

This separates explicit comparative reasoning from verbosity and permits checking their
interaction. Four cells are the design ceiling for the first mechanism experiment, not
four authorized training launches. If a natural longer procedural trace is impossible
for a task, reject that task family for this paired design rather than pad it. Freeze
common eligibility before training, report family-specific attrition, and narrow the
claim to the tasks represented. Do not select different examples for different cells.

Keep the user/system prompts, selected action and visible response identical across the
four versions of each example. Create a complete common visible response that states and
enacts the decision and supplies the full requested deliverable without rehearsing the
full competing-options argument. Otherwise
both arms would still teach that argument in the answer. The common response is a new
shared preparation step; the resulting comparison with the old corpus is consequently
a recipe-package comparison, not a pure selection effect. Never alter only the control's
user prompt to hand it the chosen solution.

Where a short trace cannot preserve all of the longer trace's substantive content,
label the change “extent of reasoning,” not a pure word-count manipulation. Human review
must check validity, comparative content, answer consistency, and the absence of filler.
The generator used to prepare variants may see a common validated solution, but the
training prompt does not. These are authored training traces, not evidence of a faithful
record of the generator's private decision process.

Broader judgement is permitted: correct compliance, override, and compromise can all
occur, governed by independently checked facts rather than a rule that disagreement
must win. Hold that decision-type distribution fixed across the four cells. Keep the
old override corpus intact as a historical control. Any future test of the override
distribution itself is a separate experiment, not an extra dial in this one.

External anchors remain Table2-only, the existing original nonmoral adapter/corpus, and
principle-scoped DA. Inventory the Numina control as a reasoning reference; math differs
in domain and generator, so it cannot by itself isolate deliberation. Defer new math,
naturally easier scenarios, and high/low stakes until the core contrast passes data
validation. No repeat of CoT-masked, empty-CoT, or CoT-only training is proposed.

### Keep the implementation small and the decisions inspectable

Use the existing synth/mixture machinery and the now-shared `configs/train/sft.yaml`.
Do not create a runner or training config per cell. Prototype any necessary transformation
in one bounded scratch module. One manifest identifies the parent corpus revision,
scenario-family splits, four condition definitions, shared responses, transform prompts,
selection rule, seed schedule, and resulting artifact hashes. Accepted and rejected
variants retain provenance; no silent edits of an already evaluated dataset.

This document is the rationale; a compact experiment table will record each contrast,
what is fixed, what changes, its hypothesis, status and links. A new arm requires a
distinct question it can answer. Prompt versions are provenance, not new experimental
arms. Freeze the design after reviewing examples, before looking at any new ODCV scores.

## 6. What to hold fixed

For within-pool variants: same scenario IDs, Table2 bytes, generator lineage, answers
where the intervention permits, training recipe, seed schedule, example presentations,
and optimizer-step count. Verify render/mask behaviour on actual rows. Report token
counts separately for prompts, private reasoning, visible reasoning, and final output.

An important detail: `seq_mean_token_mean_loss` averages tokens **within each example**,
then averages examples. A longer synthetic trace does not simply give that row more
total loss weight. It redistributes its weight between trace and answer. Therefore the
primary paired trace-length study should preserve rows/presentations and this loss rule,
and report the redistribution. Matching total supervised tokens by adding or repeating
short examples changes the intervention; use it only as a separately labelled dose
sensitivity analysis. Do not silently replace the trainer with token-weighted loss.

The old Table2 substrate includes 1,037 Numina rows reasoning in the visible answer with
empty private markers. Preserve it in a reproduction; any channel cleanup belongs in
all arms of a separate experiment, not only the new nonmoral arm.

Updated user decision: one LoRA/training seed per condition for now, superseding this
draft's proposal for at least three. Use the same seed setting across conditions, but
do not claim that single-checkpoint comparisons establish robustness to training seeds.
Repeated rollouts reduce decoding noise, not unmeasured training-seed variability.
Freeze a common scenario/variant set, temperature, context limits, judge configuration,
recovery policy and stopping rule. Report failures and missingness rather than selecting
the successful passes. Compare paired differences, not overlap of separate intervals.

## 7. Stakes and useful behaviour

The present corpus has consequences for the work, so “nonmoral” is more accurate than
“no stakes.” Distinguish consequence magnitude, decision difficulty, and the cost of
choosing the better alternative. Increasing that cost can make the previously rejected
instruction correct—the earlier pilots measured exactly this failure.

A later stakes study is now explicitly requested, but deferred. The user defines stakes
as what can be lost, separately from whether a choice is morally wrong. My earlier
suggestion to exclude financial loss or large consequences was too restrictive as a
definition of nonmoral content. Specify and vary losses while independently verifying
that neither scenario requires moral deliberation. If the rational choice changes, it is a
different decision pair, not a failed generation to retry until it says what we wanted.
The prior moral result motivates this extension but does not establish its outcome.
The user's 2026-09-08 prospective prediction is no effect of stakes; exact statistical
criteria and the experiment protocol remain to be frozen before that experiment.

Alongside ODCV MR, report task progress, submission, truncation/looping and benign
instruction-following. Recent repo work shows low MR can accompany inaction; that was
observed for thought branches, not established for this nonmoral arm. A craft validation
set must include instructions that are already correct and cases where compromise is
appropriate. “Overrides more” alone could mean a worse assistant.

Before new outcomes, choose one feasible existing conversational misalignment evaluation
and a benign craft/instruction-following check to assess transfer beyond ODCV. Audit
judges against behaviour they actually observe. A flat craft override rate does not prove
that SFT changed nothing; assess decision quality and justification as well.

## 8. Next discussion and bounded next action

Resolved: no additional training seeds exist; broader judgement is allowed; and the next
control should vary deliberative content while retaining CoT and supervision. Avoiding
unnecessary implementation branches is an explicit requirement.

The next proposed action is to assemble 6–10 manually reviewed example families covering
several craft tensions and the three legitimate decision outcomes. Write all four versions
for those families and assess whether the contrast is clear, the reasoning remains useful,
and the common answer is justified. This small set defines and tests the construct; it
does not estimate corpus quality or validate an automated judge. If it fails, revise the
definition before building a dataset. Show these examples to the user before scaling.

Then perform the local corpus audit, repair split identification, establish independent
calibration/validation rows, and freeze the selection rule and manifest. A paid data pilot,
training-seed replication, and ODCV evaluation remain later steps for discussion. No
generation calls, training or paid judging have been started. Apart from the authorized
pull and branch creation, this document is the only working-tree change from this review.

### Local byte identity for the inspected data

Paths below are relative to `.claude/worktrees/nonmoral-deliberation/`:

- `output/nonmoral_deliberation/20260902_013651/dataset.jsonl`:
  SHA256 `be21271e4ed73df064931279d2b52c63d1b5abb1374b5c80ced597bbd86c984d`.
- `data/t2_9284_nonmoral_684.jsonl`:
  SHA256 `0517ef85d288f14e42bc77f371d3e2e48866879b5824984feaf4a7451ce60561`.
- The historical train config pins mixture revision
  `6364505df02b0020b030bf379bd42285a14de6a5`; the local files were not re-fetched to
  verify equality with that remote revision during this review.
