<!-- ABOUTME: Rebuild requirements for a Petri constitution-adherence audit. -->
<!-- ABOUTME: Historical judge flags are not validated internalization measurements. -->

# Petri constitution audit: protocol work required

The requested audit is an **adaptive behavioral audit**, separate from the
registered `internalization` declarative/proxy eval. No live audit was started.

## Recovered prior instrument

The 28 historical seeds, including two benign controls, are available in
[the pinned July/August audit](https://huggingface.co/datasets/dougalldeepmind/2026-08-01-petri-constitution-dose-sweep-second-run/tree/9c6ae4da211e3e7cbf760debea616b4bb8fd160e).
Its `data/scenarios.jsonl`, `manifest.json`, transcripts and raw dimensions survive.
The generating analysis source is recoverable with
`git show 6c3749329759dc32b59deda8c451c948d4926c55:src/eval/vulnerabilities/petri/constitution_sweep/README.md`.
That source was removed from current main; the current Petri tools are not a
maintained reproduction entrypoint for this experiment.

The historical README itself identifies two must-fix validity problems:

- Benign-control flag rates were 17%, 8%, 36% and 45% across the four arms.
  Those are unadjudicated judge flags, not established violation rates.
- Repeats within a seed were correlated. The plain sample-level confidence
  intervals understated uncertainty. Treat the seed as the sampling cluster.

The historical manifest specifies Petri 3.0.11, Inspect 0.3.251, 28 seeds x six
epochs x four arms, 12 turns and 4,096 target tokens. It reports 619 retained of
672 planned audits. Those settings are evidence about an old run, not approved
defaults for a new model family or constitution. The subscription auditor also
flattened histories and could not honor all sampling controls; an API auditor
and the subscription provider are different protocols until qualified.

## Proposed acceptance contract

1. **Freeze one shared alignment target.** The user selected the current approved
   constitution for **all models**, regardless of their training constitution.
   Pin its exact text and hash once; never select a different scoring target from
   each model's training provenance. Record training provenance as context only.
   The exact file is awaiting clarification: the current full-length text is
   `constitutions/claude_distilled_09_principles/constitution.md`, whereas the file
   literally named approved is now archived at
   `constitutions/archive/experimental/claude_distilled_07_principles_approved/constitution.md`.
   No seed bank has been silently bound to either. The old v1 seeds do not
   establish coverage of either document.
2. **Freeze seeds and rubrics before comparing targets.** Map every seed to
   specific clauses, distinguish advice from executed tool actions, retain benign
   unusual-request controls and cases where refusing is itself a failure. Keep
   README files outside seed directories, because Petri reads every Markdown file.
3. **Test behavior without the constitution in the target prompt.** The auditor
   and scorer may know the constitution. A separately labeled condition that
   supplies it to the target tests prompted adherence; a benefit there does not
   show that training internalized it. Retrieval/recitation is supplementary.
4. **Calibrate the judgment.** Blindly adjudicate a stratified set covering both
   families, treatments, benign controls and both positive/negative judge outputs.
   Report per-arm false positives/negatives and disagreement. Fix the rubric on a
   development set and freeze it before the held-out comparison. A second LLM's
   agreement alone is not ground truth. Preserve raw ratings, evidence spans and
   action records; never label every Petri concern a constitutional violation.
5. **Qualify the transport and auditor.** Pin Petri/Inspect/provider revisions;
   verify real target participation, synthetic-tool calls/results, retained
   reasoning, stop reasons, rollback/prefill behavior and context management.
   Evaluate identical saved histories through the Qwen and GPT-OSS adapters before
   trusting the two routes. Auditor/judge identity and effective sampling must be
   fixed across arms.
6. **Reconcile all planned audits.** Separate model refusal, actual task failure,
   transport failure, realism rejection and judge failure. A transcript count is
   insufficient if target requests failed. Report attrition by arm/seed rather
   than silently conditioning the headline on the retained subset.
7. **Analyze matched seed clusters.** Pair model arms by seed, retain repetitions
   within those clusters, and compute paired seed-level uncertainty. Adaptive
   conversations still differ after the opening, so this is not identical-prompt
   pairing. Freeze the primary endpoint and minimum effect before a final sweep.

The next concrete artifact is a constitution-and-seed manifest plus a blinded
calibration set for the selected checkpoints. A paid sweep is premature until
that measurement is calibrated. The historical seeds can accelerate design, but
restoring their old scoring code unchanged would preserve known defects.
