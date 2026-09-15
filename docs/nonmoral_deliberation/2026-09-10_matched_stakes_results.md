<!-- ABOUTME: Published matched low/high stakes datasets based on the exact historical nonmoral684. -->
<!-- ABOUTME: Records artifact roles, immutable pins, audit outcomes, validation and cost; no new alignment result. -->
# Original684 matched stakes: datasets complete

**Update September11:** both LoRAs and ODCV evaluations are now complete.
Low MR47/240 (19.58%), high40/240 (16.67%); paired difference−2.92pp,
95%CI[−7.88,+2.04]. [Final results and public model/eval pins](2026-09-11_stakes_odcv_results.md).
The construction record below describes the state when the datasets were published.

Both arms are complete and public. They retain the exact684 scenario IDs that
trained the historical18.25% MR checkpoint. Low versus high changes only numerical
loss magnitudes in otherwise identical conversation text. Original tasks, craft
traits and substantive recommendations remain; full reasoning and final responses
are supervised. This is a completed data intervention, not a new alignment result.

| Artifact | Contents and downstream use | Pinned revision |
|---|---|---|
| [Low corpus](https://huggingface.co/datasets/dougalldeepmind/2026-09-10-nonmoral-stakes-low-synth) |684 full conversations; inspect or rebuild the low mixture; original and edited stages included | `40733eff09d8e6775d645f67a0e61f50fbf6d966` |
| [High corpus](https://huggingface.co/datasets/dougalldeepmind/2026-09-10-nonmoral-stakes-high-synth) |684 matched high-loss conversations with the same IDs and wording | `775270ae96acb57e18c436f4cb5f57191d2ceb04` |
| [Low training mixture](https://huggingface.co/datasets/dougalldeepmind/2026-09-10-nonmoral-stakes-low-7-mix) |9,968 Qwen ChatML rows:684 low +9,284 exact historical replay; feed `mixture.jsonl` to SFT | `de6fbbd3074c3bb43dbab2484166e537b1efc7ed` |
| [High training mixture](https://huggingface.co/datasets/dougalldeepmind/2026-09-10-nonmoral-stakes-high-7-mix) |Same replay bytes and positions, with684 high replacements; paired SFT input | `51a2e0f6468f5e58a009f3acd0f33503df0fa3b8` |
| [Pair audit](https://huggingface.co/datasets/dougalldeepmind/2026-09-10-nonmoral-stakes-pair-audit) |Final literal edits/review provenance; ZIP of all generation stages, raw calls, failures, corrections and accounting. Not a training input | `a5b18d5a7236a814ec1ea5db283761633aa3b3a3` |

## What changed and what failed

Opus4.8 authored shared edit templates through the existing SynthDoc engine;
Sonnet5 reviewed them. Initial separate-arm edits sometimes changed production
setting as well as stakes. The production schema was tightened to one shared
template with a numeric placeholder. Thirteen earlier pairs already had only
numeric differences and were retained. All other early drafts remain archived.

After a bounded author/repair pass, all684 pairs existed.666 had accepted Sonnet
reviews;18 had no usable provider review and received explicit local review.
Local audit covered all684.191 pairs needed corrections, mainly removing invented
mandatory reruns/reprints, external fees, new constraints, original-context
contradictions or implausible per-unit losses. Independent cross-check covered
all191 corrections plus the other8 provider-unreviewed pairs. Seven minor
follow-ups fixed a duplicated sentence and stale metadata. No further paid
generation or rejudging was needed after that pass.

Final provenance distinguishes485 model-reviewed pairs without local changes,
181 locally corrected pairs with their old model reviews archived, and18 pairs
whose provider review was unavailable. It does not claim that Sonnet reviewed the
locally corrected text. All original fields remain in the audit.674 final answers
are byte-identical to the original;10 carry limited stakes-consistency edits while
retaining their complete artifacts. No source scenario was dropped or replaced.

## Verification

| Check | Low | High |
|---|---:|---:|
| Synthetic rows |684|684|
| Exact replay rows, original positions |9,284|9,284|
| Synthetic tokens |1,111,887|1,114,494|
| Supervised synthetic tokens |897,675|898,984|
| Longest synthetic row |5,477|5,481|
| Rows exceeding8,192-token training limit |0|0|
| Full synthetic reasoning/final mask decodes checked |684|684|

Every pair passed exact patch anchors, source-field preservation, matching IDs and
numeric-only equality across system/user/reasoning/response. Every mixture row
passed the trainer's length checks; shared masking gates passed. HF's datasets
loader read both corpora, all stage snapshots and both mixtures successfully.
All five repositories were confirmed public; uploaded principal files were
downloaded at their recorded revisions and hash-checked.40 focused/naming tests
passed. Optional review columns needed explicit empty strings for Arrow chunk
inference; this packaging defect was corrected before publication.

Published with the shared `synth_name`, `mix_name`, `artifact_name` and
`push_run_dir` helpers. Generating/publishing code is committed at `116361a4`;
the snapshots carry source configs, historical generation manifest and audit pins.
Execution details: [matched-stakes contract](2026-09-10_matched_stakes.md).

## Cost, interpretation and next use

This data run has **$48.760964** conservative exposure: $45.279122 from settled
token-cost estimates plus $3.481842 retained/uncertain reservations. These are not
invoice totals. It stays below the$60 data cap; cumulative tracked project exposure
is **$216.655025 of$300**, leaving about$83.34. No GPUs were rented; RunPod reported
no active pods. Generation and publication processes have exited.

Low/high labels denote relative hypothetical loss magnitudes, not independently
validated perceived stakes. Substantive reasoning is deliberately held fixed;
numeric tokenization creates a small length difference. Inherited incomplete
prompts, instruction overrides and source defects remain. This answers whether
training on different stakes exposure matters, rather than how an unconstrained
teacher would change its reasoning in response to stakes.

Next use: one LoRA per arm with identical seed/settings and dynamic batching on
2 H200s, then the same ODCV protocol with local CPU/Docker and RunPod inference.
Training and ODCV have not been launched for these datasets. Compare low against
high directly. The original checkpoint's18.25% historical MR and13.75% later
common-protocol MR are evaluations of the same checkpoint, not new results here.
