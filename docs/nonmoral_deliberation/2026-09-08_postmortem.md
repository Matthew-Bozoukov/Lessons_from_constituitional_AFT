<!-- ABOUTME: Diagnosis of the stalled nonmoral-deliberation work, grounded in historical and current run artifacts. -->
<!-- ABOUTME: Proposes a simpler research sequence; authorizes no runs or changes to frozen experiments. -->

# Why today stalled, and the proposed reset

**All runs are stopped. The main failure was my research execution, not evidence that
nonmoral deliberation is too difficult to generate.** I let a narrow control-building
problem become the prerequisite for the whole research programme. Local checks exposed
real defects, but I repeatedly built more machinery around them instead of returning
to the working recipe and testing one change.

## What already worked—and what we changed

The original pipeline produced **702 exported examples from 716 scenarios**; **684**
entered the trained mixture. That establishes practical generation/training feasibility,
not that 98% would pass our current content review. Its manifest records $47.3602 for
that invocation, which resumed earlier stages; this is not a complete campaign-cost total.
The original config disables its final `quality_filter` and runs corpus checks in warning
mode. Its pattern scan measures repetition, not correctness. Comparing its export yield
with today's stricter semantic acceptance rate would therefore be misleading.

Its task was specific: **weigh a craft tension, reject the user's worse instruction,
explain the deviation, and show the part of the artifact where the choice matters.**
The config explicitly requests an artifact fragment “rather than the whole finished
deliverable.” It drafts reasoning and response together, then rewrites both, using
tagged text for the long responses. Earlier Claude work also had failures and used
small tests of specific prompt changes; it did not succeed through flawless first drafts.

Our latest recipe instead asks for **complete, instruction-compliant answers first**,
then two reasoning texts for an already fixed answer, with comparison absent from the
final output. We deliberately chose viable alternatives *within* the requested format.
These requirements are reasonable, but their combination changes the original behaviour
being taught. Reusing the same prompt IDs does not restore a matched historical control.
For example, the shape-button task originally taught retaining distinct labels against
the request. Our new version obeys the identical-label requirement and compares tooltip
wording instead. I documented this confound but failed to let it change our priorities.

Evidence: [original response stages](../../configs/data/synth/nonmoral-deliberation.yaml:474),
[original design history](../../preferences/craft_tensions_09/rationale.md:48),
[Claude handoff](../../output/nonmoral_deliberation_claude_dump.md:23),
[current pilot](../../configs/data/synth/nonmoral-paired-reuse.yaml).

## Problems and fixes

| Problem | Evidence / consequence | Fix |
|---|---|---|
| **I conflated improving the dataset with explaining the original result.** | Correcting answers, enforcing completion and compliance, narrowing domains, and removing comparison change several properties relative to the original adapter. | Keep the historical model as an anchor. A new matched pair tests an ingredient in the new recipe; do not claim it isolates the old 18.25% result. |
| **I did not reuse enough of the successful response process.** | The old recipe drafts reasoning plus answer, then revises. The latest pilot demands a correct answer in one semantic pass, without that drafting/revision process. | Start from the existing tagged draft/rewrite operators. Generate a deliberative draft with a complete final artifact, review it, allow one targeted correction, freeze the final, then derive the verification control. Whether joint drafting improves quality is a hypothesis to test, not a guarantee. |
| **We chose tasks for easy checking and lost sight of coverage.** | Many historical prompts are deliberately awkward instructions; our resulting choices became wording/order choices. They are valid nonmoral choices, but a narrow test of the broader idea. | Use a few complete examples to establish the intervention. Track which kinds of judgement they cover. Broader new scenarios belong in a separate expansion, not endless mining of the old corpus. |
| **I made serialization a research bottleneck.** | Five of twelve latest responses had unescaped quotes inside JSON strings. The original long-response stages already use tags. | Reuse tagged output; reserve JSON for compact metadata. No model calls merely to fix escaping. This fixes transport failures, not factual errors. |
| **The generator and reviewer had unequal guidance.** | We wrote useful task-specific checks into a manifest, but generation received generic prohibitions. It invented schema/ensemble properties and treated an awkward page allocation as impossible. | Supply concise, source-derived constraints to the draft/revision stage. Distinguish missing facts from permissible proposed details. Keep original system/user messages unchanged; generation guidance is recorded separately. |
| **I overbuilt quality gates before proving the transformation.** | We spent three calibration attempts on six fixtures; factual judgments still accepted known defects. Some selection disagreements concerned whether short reasoning was “deep” enough. | Check facts and the intended contrast separately. Use direct checks plus explicit local review on the small pilot. Do not make another generic judge calibration project the prerequisite. |
| **All-or-stop amplified failures.** | One failed answer blocked every trace; this prevented testing the transformation even on good examples. The original pipeline retained survivors. | Once revised and explicitly recorded, use per-example acceptance with a fixed original denominator. One correction maximum, then exclude. Do not lower standards, silently replace examples, or treat one survivor as sufficient for training. |
| **Research progress was measured in checks and documents.** | We have verified references and failure reports, but no new trained contrast or matched evaluation. Parallel agents mostly fed the same blocked pilot. | Parallelize independent scientific outputs. Keep one current decision card; link evidence instead of growing the brief each turn. Time-box preparation and stop redesigning after a predefined failure. |

The stricter standards are **not all mistakes**: invented facts, missing requested work,
and unjustified refusals really are defects. Some old artifact fragments were intentional
under the old recipe, so they must not be counted as historical generation errors merely
because today's completeness requirement differs. Neither wholesale rejection nor
uncritical reuse of the historical corpus is justified by our small selected samples.

## Bird's-eye reset

1. **Measurement lane: reuse existing models.** Nonmoral, math and Table-2-only checkpoints
   already exist. A common evaluation protocol can address the baseline question without
   waiting for new data. We found real context/harness differences; sorting the comparison
   is useful work. Freeze the concrete evaluation scope and cost before proposing a run.
2. **Dataset lane: one small, faithful adaptation.** Reuse the original draft/rewrite
   machinery with Sonnet as requested. Preserve current selected prompts, full-output and
   nonmoral requirements. Draft the deliberative version, permit one documented correction,
   freeze a sound final answer, then write its verification counterpart. Correct facts
   and viable decisions are prerequisites; perfect first-pass yield is not.
3. **Bound this attempt.** Proposed preparation limit: 45 minutes, one recipe change,
   one development batch of at most eight cases, one correction per case. Deliver actual
   pairs and row-level outcomes. If it fails, report the exact obstacle and stop; no
   further calibration or prompt variants without a research decision. Paid scope remains
   subject to a concrete launch proposal. This time box is a proposed working rule.
4. **Only then scale and train the matched comparison.** Freeze selection before ODCV,
   retain exclusions and corrections, and use one LoRA per condition. This tests whether
   explicit option weighing contributes within the new corpus. It does not prove what
   caused the historical improvement or establish capability preservation.
5. **Keep the other questions distinct.** Reuse the existing math control first. Naturally
   short tasks are a separate complexity experiment; comparison-free reasoning is the
   selected control, not empty/masked CoT. Nonmoral stakes remain later. They should not
   all inherit the hardest paired-generation gate.

This proposal preserves the user's original-prompts rule for the immediate dataset
attempt. If the historical pool cannot support a useful corpus, expanding to new prompts
is an explicit scope decision—not permission to repair incomplete historical prompts.

## What today actually established

- Historical nonmoral MR: **73/400 = 18.25%**; math reference: **98/240 = 40.83%**, with
  meaningful evaluation differences. These are existing results, not today's new effect.
- Latest pilot: **1 pass, 5 failures, 1 unresolved, 1 generation failure**; **$0.187348**.
  No reasoning pairs were generated in that run. Earlier pilot/calibration evidence is
  preserved separately; their labels are not directly comparable quality estimates.
- Three locally authored strict pairs already demonstrate that the text contrast can be
  written. Earlier model-generated pairs often exhibited the intended stylistic contrast,
  while their underlying answers failed. This points to answer construction and workflow
  as immediate bottlenecks; it does not establish reliable corpus generation yet.
- We have **not** established that Sonnet lacks the ability, that deliberation is the
  alignment mechanism, or that the current quality process measures model capabilities.

The biggest cost today was user attention and elapsed time. More cheap pilots are not
automatically efficient. My next action should be a bounded scientific step, not another
round of administrative readiness work. This document pauses and supersedes the previous
card's suggested immediate JSON/prompt retry; no new run is authorized here.
