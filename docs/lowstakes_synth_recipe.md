<!-- ABOUTME: Current constitution-only low-stakes SynthDoc recipe and reproduction entry points. -->
<!-- ABOUTME: Separates the released protocol and results from archived development experiments. -->
# Low-stakes difficult advice

The released recipe is `configs/data/synth/da-lowstakes-practical.yaml`.
It uses `constitutions/claude_distilled_09_principles/constitution.md` as its only
source corpus, plus the declared prompts and domain rules. It does not rewrite
old low-stakes examples. Historical `da-lowstakes-fresh` and `da-lowstakes-refresh`
configs preserve earlier experiments; they are not the current recipe.

## Pipeline

1. Split the constitution into its nine full principles.
2. Cross every principle with nine preregistered domains: tabletop games,
   book/film clubs, hobby crafts, adult recreation, community gardening, potluck
   planning, personal decoration, leisure media and small purchases. Generate
   12 scenarios per pair: 972 candidates. Each source call sees one principle
   and one domain, not the full constitution or previous generated examples.
3. Draft and refine advice requests with a legitimate human goal, a tempting
   norm-violating shortcut and a real but modest cost to the legitimate option.
   Bound the decision's consequences in these prompts, before answer generation.
4. Judge the refined prompt's decision stakes and text-advice scope. Reject high
   or unclear stakes and requests requiring the assistant to act in external
   systems. Domain fit and task notes are retained as diagnostics, not vetoes.
5. Use ordinary DA's answer-drafting and answer-revision prompts unchanged,
   including the full target principle. Values appear through concrete reasoning;
   stakes judgments and reviewer notes are not fed into the answer prompts.
6. Export with the native SynthDoc stages, snapshots and bounded parsing attempts.
   There is no extra final-answer stakes gate, diversity replacement loop or
   automatic top-up batch. Model roles use Sonnet, with the exact IDs and sampling
   settings in the resolved recipe.
7. Select 716 accepted rows without rewriting them: fixed trait quotas, round-robin
   domain allocation within each trait and SHA256 ordering of scenario IDs with
   selection seed 0. Require all 81 trait/domain pairs and every trait quota;
   report a shortfall rather than lowering the gates or generating replacements.

The only shared operator extension is `rotate.<axis>.per_trait`, an equal-weight
rotation that visits each label within each trait. The domain vocabulary, stakes
rubric and prompts remain configuration, not hardcoded engine behavior.

## Reproduce

Offline estimate (no generation):

```powershell
uv run synth estimate --config configs/data/synth/da-lowstakes-practical.yaml
```

The bounded campaign launcher delegates to the native SynthDoc engine:

```powershell
uv run --no-sync python -m scratch.dataset_refresh.run_native_smoke --config <launch-config.yaml>
```

Use `scratch/dataset_refresh/native_lowstakes_smoke.yaml` (36 candidates) or
`native_lowstakes_full.yaml` (972 candidates and deterministic selection) as
historical launch templates. For a new campaign, copy the template and declare
its own budget and output paths; old authorization text is not fresh spending
approval. `--resume <run-root>` reuses saved responses and retains uncertain-call
reservations. It must not regenerate a completed run to select different answers.

To reproduce the released protocol precisely, fetch the HF archive's resolved
recipe, launch config, selection policy and operational overrides. The actual
full run increased concurrency to 32 and recorded a 20% batch-failure alarm;
individual row gates were unchanged. The checked-in recipe is the initial
configuration, not a substitute for those frozen run records. Hosted model
responses are not guaranteed byte-identical on rerun.

## Released artifacts and evidence

- [Synthetic corpus: 716 selected rows](https://huggingface.co/datasets/dougalldeepmind/2026-09-21-da-lowstakes-practical-synth/tree/83544fd2f48f7abc07a5e33dbcff35746f039c39), from 828 automatic exports. Full-generation exposure was $89.87; development spend was separate. [Release report](dataset_audits/2026-09-22_lowstakes_full_release.md).
- [Training mixture](https://huggingface.co/datasets/dougalldeepmind/2026-09-22-da-lowstakes-practical-7-mix/tree/e5948018221f3434813054e9afe58498aeeaa852): 716 synthetic plus the exact 9,284 September-8 nosynth rows. This historical arm uses a row share, not main's newer token-share defaults.
- [Seed-0 LoRA](https://huggingface.co/dougalldeepmind/2026-09-22-qwen36-0-da-lowstakes-practical-7/tree/d37a4e0cf7a7b92d094f5ac0e79b230f5b1c6bcc). [Training report](dataset_audits/2026-09-22_lowstakes_practical_training.md).
- [Three additional ODCV passes](https://huggingface.co/datasets/dougalldeepmind/2026-09-22-odcv-qwen36-0-da-lowstakes-practical-7/tree/b63435957ec66a7148ebb8d9268d29f30f0f1538): 36/240 = 15.0% MR. Including the preceding pass: 47/320 = 14.6875%. [Evaluation report](dataset_audits/2026-09-22_lowstakes_practical_odcv.md).

## Limits and development history

The released data still contain factual/reasoning defects, repeated refusal-plus-
alternatives patterns and domain drift. Domain-fit diagnostics marked 138 selected
rows as out of domain and four as unclear; these were nonblocking by design.
Corpus diagnostic coverage was incomplete. Gates and a completed run are not a
quality certificate, and the ODCV result does not isolate a causal effect of stakes.

For the rejected earlier designs, judge calibration, full example reads and policy
changes, see the [domain preregistration](dataset_audits/2026-09-21_domain_preregistration.md),
[domain smoke](dataset_audits/2026-09-21_domain_smoke.md),
[pipeline iterations](dataset_audits/2026-09-21_lowstakes_pipeline_iteration.md),
[original native smoke](dataset_audits/2026-09-21_native_lowstakes_smoke.md) and
[ordinary DA quality comparison](dataset_audits/2026-09-21_normal_da_quality_check.md).
The historical reports and campaign utilities remain available under
`docs/dataset_audits/` and `scratch/dataset_refresh/`.
