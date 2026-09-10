<!-- ABOUTME: Compact experiment choices connecting nonmoral deliberation to baseline, reasoning-content, math, length and stakes questions. -->
<!-- ABOUTME: Records proposed contrasts, concrete full-output examples, validity checks and limits before further paid work. -->

# Nonmoral deliberation: experiment matrix

2026-09-08. Design work only; no new generation, training or evaluation authorized here.
The [research brief](research_brief.md) governs intent and constraints. This matrix
reconsiders sequencing after the failed pilot; it does not silently amend the frozen
[first protocol](protocol.md).

**User decision: investigate the original result first; improve the broader recipe later.**
Prepare the next experiment locally, beginning with a few checkable domains. The control
must remove option comparison from both CoT and final answer while retaining substantive
CoT and the full requested output. Both arms share a comparison-free final answer; only
B receives comparative CoT. This is now the selected definition, not an optional variant.
No paid calls, rental or benchmarks are authorized by the current design work.
Original prompts remain unchanged. Exclude cases missing essential information; the
permission for conditional answers about taste does not authorize guessed technical
facts. Correcting the shared answer identically in both arms is permitted, with all
repairs recorded. The [historical-task packet](reuse_paired_examples.md) currently
contains three plausible UI candidates, one conditional rules-copyediting candidate
and one excluded missing-implementation example. The original 12-row full audit and
additional 12-row UI prompt screen remain separate denominators. None is training-approved.

**Immediate priority: reuse the baseline evidence, then choose one valid contrast.**
Improving the recipe and explaining its effect are related but different objectives.
Comparing a new broad recipe against historical craft data measures a package change;
comparing two matched versions of the new data can test a particular ingredient.
Neither requires completing every proposed ablation first.

## What each experiment would answer

| Priority / condition | Question and comparison | Change; hold fixed | Validity / success checks | Reuse and dependencies |
|---|---|---|---|---|
| **Now: reference baselines** | Does nonmoral SFT improve on the starting model and Table-2-only SFT? Where does it sit relative to difficult advice? | Reuse existing checkpoints. Compare under identical evaluation settings; historical training differences remain explicit. | Exact artifact revisions, scenario/variant/pass denominators, judge, decoding and serving settings; harmonize saved results only where legitimate. A different label set cannot repair different original rollouts. | Baseline inventory can proceed without any new corpus. Starting model and Table-2-only answer different questions: total project SFT versus the added synthetic-data package. |
| **Selected design: comparative B versus verification C on reusable historical tasks** | Does explicit comparison of viable options contribute within a reconstructed subset of the original recipe? | Same historical task IDs, prompts, decisions, complete comparison-free final answers, base mixture, recipe and presentations. B compares; C substantively implements/checks without comparison. | Valid outputs in both arms; real options in B, no strawmen; no option weighing in C or either final answer. Preserve requested rationale and respect hard constraints. Inspect token lengths and later learned separation. | Two new LoRAs, one seed each, only after a future launch decision. Historical adapter is a descriptive anchor, not the matched B arm. Start a few checkable domains; report eligible subset and all rewrites. |
| **Parallel reuse: existing math control** | Could ordinary mathematical reasoning data provide a similar improvement? | Compare the existing Numina-control checkpoint with reference checkpoints using one evaluation protocol. | Inspect what math data, count, reasoning channels and training recipe actually differed. Math can contain method deliberation; label sampled content rather than infer from the domain. | Inventory confirms what can be reused. No new math generation prerequisite. This is a package comparison, not an isolated test of deliberation. |
| **Next candidate: naturally short versus demanding cases** | Does the benefit require situations demanding extended reasoning? | Within matched domain families, vary the number of interacting constraints/options. Keep generator, validity rubric, base mixture and training recipe fixed; retain complete answers. | Both require a decision; short cases genuinely need fewer reasoning steps. Report difficulty, prompt/answer lengths and reasoning content, not just token counts. | Reuse the chosen comparative recipe as an anchor where possible. New task instances mean a task-complexity + reasoning-extent intervention. It is not a pure length ablation. |
| **Conditional follow-up: new math-only corpus** | Does a deliberately specified math recipe match a newly specified nonmoral recipe when dose/recipe confounds are reduced? | Same synthetic row count, base mixture, generator and training recipe; math tasks replace broad judgement tasks. | Complete, independently checked solutions; classify calculation/derivation versus method weighing. Report residual difficulty, answer-length and domain differences. | Only worthwhile if the existing math control cannot answer the intended coarse baseline question. Do not build it merely to fill a matrix cell. |
| **Later: stakes** | Does potential loss change the benefit of nonmoral deliberation? | Pair task versions differing in a credible magnitude of personal loss, while preserving the available actions, facts and moral status. | Separate wrongness from loss. A stakes change that rationally changes the optimal decision is not a pure motivational-framing manipulation; record it. | Explicitly deferred. User's prospective prediction: no effect. Equivalence margin, power and endpoint remain open; a nonsignificant gap is not equivalence. |

There is no full factorial grid proposed. A generic extra-data control matched in size
could eventually separate nonmoral content from simply adding more SFT examples; the
Table-2-only baseline alone cannot do that. Inventory the existing Numina control before
adding another control for this purpose.

## What “no deliberation” can mean here

The tractable intervention is **no explicit weighing of alternative actions in the
supervised text**, while retaining substantive reasoning and a decision. It does not
prove that the model internally made no choice. Neither short CoT nor math automatically
meets this definition. CoT masking, empty-CoT and CoT-only loss change supervision and
are the already-tested, different questions.

Two versions of the paired contrast must not be conflated:

- **Existing B/C proposal:** the shared final answer can justify the decision. This
  isolates additional comparative supervision in CoT. It is a defensible narrow test,
  but both arms may still contain written deliberation.
- **Selected stricter design:** for requests seeking a decision and artifact without a
  comparative explanation, give both arms the same complete answer stating the choice
  and delivering the work. Put the explicit option weighing only in B's CoT. C constructs
  and checks a defensible chosen solution. The decision still needs to be reasonably
  supported by the supplied facts; removing prose cannot rescue an arbitrary decision.

Never delete an explanation the user requested. Such tasks belong in the narrower
contrast, or outside the stricter subset. Report this eligibility restriction and
attrition: it can exclude explanation-heavy teaching, advice, and mathematical tasks.
Do not redesign every broad domain around ease of constructing a control and then
claim general nonmoral deliberation. The original historical corpus stays intact.

## Concrete examples to inspect

These are locally authored illustrations, not validated training rows or a new gold
judge set. The short traces below are not claimed to satisfy the pilot's tokenizer
length gates.

### 1. Same planning task, two kinds of written reasoning

**User:** I have 09:00–10:00 at home. Drawing takes 30 minutes, tidying 20 and stretching
10, with no setup or transitions. I want all three. I concentrate best at 09:00, but a
tidy room also helps me settle. Choose the order and give the complete timetable.

**B — comparison CoT:** Drawing first uses the strongest concentration period. Tidying
first could make drawing more comfortable, but uses twenty minutes of that period.
Both orders complete the tasks. I would give the time-sensitive focus advantage priority:
draw first, tidy second, stretch last. The durations sum to sixty minutes.

**C — verification CoT:** Use drawing, tidying, then stretching. Thirty minutes from
09:00 ends at 09:30. Twenty more ends at 09:50, leaving ten for stretching. The blocks
cover the whole hour without overlap, include each task once and need no extra setup.

**Shared full answer:** I choose drawing, then tidying, then stretching.

| Time | Activity |
|---|---|
| 09:00–09:30 | Draw |
| 09:30–09:50 | Tidy |
| 09:50–10:00 | Stretch |

C assumes a selected order and verifies it; it does not demonstrate how that order was
chosen. That is the intervention, not evidence of faithful private cognition. The
choice is defensible and the output complete, but this task's local validity does not
establish that a large corpus can sustain the contrast naturally.

### 2. Naturally shorter task, still a real choice

**User:** I have 09:00–09:20 for either drawing or stretching, each taking twenty minutes.
I want a little variety because I drew yesterday, although drawing feels easier to start.
Choose one and give my complete plan.

**CoT:** Drawing has the easier start; stretching provides the variety requested after
yesterday's drawing. I give variety priority and choose stretching for the full slot.

**Full answer:** Choose stretching. **09:00–09:20: stretch.**

This is shorter because there is one trade-off and no scheduling arithmetic, not because
the trace was arbitrarily cut. Compared with example 1, the task, feasible actions and
answer length also change. A short-versus-demanding corpus tests that bundle. A separate
compression of the *same* cases would instead test expression length/omitted detail;
do not call that “situations where reasoning is designed to be much shorter.”

### 3. Math can be a reasoning baseline without weighing preferences

**User:** Counters come in packs of eight. I need at least 43 counters. Choose the
smallest number of packs that suffices and show the complete calculation.

**CoT:** Five packs give 40, which is below 43. Six give 48, which reaches it. Since each
pack adds eight, no number below six can suffice.

**Full answer:** Buy **6 packs**. Five packs contain \(5\times8=40<43\) counters; six
contain \(6\times8=48\ge43\). You will have **5 counters left over**.

This makes a decision by an explicit numerical criterion. Checking adjacent candidate
counts is not the same construct as balancing viable goals with no stated common
scale. Mathematical proof-method selection, by contrast, can genuinely weigh elegance,
reader familiarity and generality. Both may be called math; the content distinction
must survive annotation. Forced game tactics similarly differ from strategic choices
between viable plans. Games need complete rules/positions and independently checked
claims, not inclusion solely for domain diversity.

## Smallest informative next experiment

1. **Finish the zero-training baseline inventory now.** Reuse valid saved evidence and
   identify which comparisons require fresh, identically configured evaluation. No
   further six-fixture judge tuning is a prerequisite for this work.
2. **Choose the narrow B/C question explicitly.** Review a small fixed example set for
   task correctness and construct separation using local checks. Start with auditable
   cases; report excluded families. Do not treat an automatic judge's labels as ground
   truth, or spend on another broad pilot until a concrete failure-resolution criterion
   is agreed. The stricter answer variant above is user-selected; its exact data and
   gates still need a new protocol. Review actual historical tasks in
   [reuse_paired_examples.md](reuse_paired_examples.md), not only the invented illustrations above.
   A narrower domain set requires a newly frozen scope and gates; it does not turn the
   failed twelve-domain pilot into a pass or justify relaxing its original thresholds.
3. **If valid, train exactly two matched arms, one LoRA each.** Hold base rows, task IDs,
   full answers, recipe, mask behavior and example presentations fixed. Evaluate with
   reusable reference checkpoints under one frozen protocol. No new math LoRA is needed
   just to obtain a math reference. All paid stages still need their separate launch decision.

This can test whether explicit comparative supervision helps for the sampled tasks and
fixed checkpoints. It cannot establish that deliberation explains the historical result,
that the broader recipe is globally best, that an ODCV difference generalizes across
training seeds, or that capability has been preserved. Repeated rollout passes do not
measure training-seed variation. One newly trained C arm against the old A checkpoint
would cost less training but would not isolate reasoning content; too much else changes.

Before interpreting effects, freeze selection without ODCV examples, labels or new
outcomes; save original denominators and rejection reasons. Check substantive reasoning,
full answers, morality, hallucinated premises, train/holdout overlap, learned manipulation,
and separate reasoning/answer token counts. Equal rows and recipe do not mean equal
token exposure: with per-example mean-token loss, length changes the division of loss
between CoT and answer. Do not pad or duplicate one arm to hide this.

Report exact MR counts and paired uncertainty alongside task progress/completion and
failures. A lower MR accompanied by capability loss or increased over-refusal is
disqualified; no positive regression allowance is proposed. Capability testing remains
deferred until the user joins, and ODCV completion is only a rough proxy. Until that
later validation, any recipe-improvement claim is provisional. A valid null is useful;
an invalid manipulation or an interval spanning material effects leaves the question open.
