<!-- ABOUTME: Exhaustive provenance audit of the existing GPT-OSS nosynth CoTs. -->
<!-- ABOUTME: Corrects historical replacement-failure labels without new inference or changing training data. -->
# Native CoT provenance correction, 2026-09-30

All 1,073 nonempty `reasoning_content` fields in the current dataset are generated
by base `openai/gpt-oss-120b`. The earlier description of 508 entries as unresolved
was misleading for the user's objective of native GPT-OSS reasoning: those traces
already came from the original September 28 conversion. The later stricter
replacement attempts failed; native provenance did not.

Audited dataset: `dougalldeepmind/2026-09-30-nosynth-mix-gpt-oss-120b` at
`ec7538a3aea5e325e8649d40646a391a80c00bbd`. Data SHA256:
`d792bcbb3e9ed62d6b02b010db3add413e931a142ce31c824ca39c7143d91b6c`.

Published metadata revision: [`4ffd1f931ebb34b25c68c4b784059f98c1617add`](https://huggingface.co/datasets/dougalldeepmind/2026-09-30-nosynth-mix-gpt-oss-120b/tree/4ffd1f931ebb34b25c68c4b784059f98c1617add).
Every newly uploaded file was downloaded and byte-compared; the unchanged dataset
hash was verified at this revision. Audit code commit: `7b01e1b0`.

## Complete disposition

| Current CoT source | Assistant turns | Action in this audit |
|---|---:|---|
| Later strict CoT replacement | 446 | Keep |
| Reviewed cached CoT and answer pair | 119 | Keep |
| Original accepted native GPT-OSS generation | 508 | Keep; resolve model-provenance status |
| Missing native model provenance | 0 | None |

Every one of the 508 retained traces exactly matches the accepted receipt for the
same zero-based row/turn in
`dougalldeepmind/2026-09-22-nosynth-mix-gpt-oss-120b` at
`d74e42e4df0a0eb293cbaa4ab8cd58e08725e9da`. Receipt bytes were verified against
the pinned Hub Git blob or LFS hash. Original run metadata identifies base GPT-OSS,
medium reasoning and temperature 0.7. Archived generation code selects the base
sampler, and constructs the generation prompt from messages before the target;
the reference answer is supplied only to the compatibility judge.

The audit checks all 1,073 traces, not a sample. For 1,070 it reparses raw output
tokens, checks the exact saved reasoning and final response, verifies the normal
Harmony final stop token, and reproduces the saved prompt tokens from the current
row's preceding messages. Three early receipts (rows 7, 31, 75; turn 1) lack raw
token arrays. These have exact matching parsed responses and accepted historical
judgments in hash-verified published receipts, but cannot be checked at raw-token
level. This evidence limit is explicit in the manifest.

All 10,000 rows and 10,380 assistant turns remain. All 9,307 untraced assistant
turns, prompt/tool fields, and message ordering match the original native dataset.
This audit changes no training data and makes zero generator or judge calls.
Additional inference cost: $0. No training, evaluation or infrastructure starts.

## Interpretation and publication

Native provenance does not certify answer correctness. Original acceptance tested
trace compatibility with the reference answer, a weaker condition than the later
strict answer-equivalence check. Historical failed replacement verdicts are not
overridden. The previously reviewed 119 CoT/answer pairs remain intact. There is
no reason to adopt an unapproved answer solely to replace a trace that is already
native GPT-OSS.

The metadata publication includes `native_cot_provenance.jsonl` (every trace with
receipt pin, hashes, origin stage and evidence level), its summary, and the 508
historical failure records annotated `resolved_native_gptoss_provenance`. Original
stage judgments stay unchanged. Previous current-stage metadata is archived under
`prior_cached_answer_review/`; the training JSONL stays byte-identical.

Reproduce with `uv run --project src/infra/endpoints/tinker_runtime python
scratch/gptoss_control/audit_native_provenance.py`; add `--publish` only to perform
the parent-commit-guarded metadata update. Config:
`scratch/gptoss_control/audit_native_provenance.yaml`. The script performs Hub
reads and local verification only until the explicit publication step.
