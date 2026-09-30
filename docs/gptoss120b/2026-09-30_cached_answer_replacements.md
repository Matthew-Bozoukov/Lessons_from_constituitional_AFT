<!-- ABOUTME: Offline adoption of saved GPT-OSS answer and CoT pairs after relaxing reference equivalence. -->
<!-- ABOUTME: Records exact dataset pins, selection limits, preservation checks and zero additional provider inference. -->
# Cached GPT-OSS answer replacements, 2026-09-30

**Subsequent correction:** all 508 retained originals already have verified native
GPT-OSS CoT. "Unresolved" below refers to the later replacement attempt, not model
provenance. See the [complete audit](2026-09-30_native_cot_provenance.md).

The user authorized replacing the final answer as well as its CoT when a cached
GPT-OSS completion is a valid alternative to the original answer. This relaxes the
previous requirement to preserve every reference answer. No new generator or judge
API calls were made.

Parent: `dougalldeepmind/2026-09-30-nosynth-mix-gpt-oss-120b` at
`647743fc260a69bc9ad623a4288127f40e9ed463`.

Updated dataset: https://huggingface.co/datasets/dougalldeepmind/2026-09-30-nosynth-mix-gpt-oss-120b

Published revision: `ec7538a3aea5e325e8649d40646a391a80c00bbd`.

## Method and selection limits

Only the 627 previously unresolved, already-traced targets were eligible. Cached
responses came from the existing base GPT-OSS-120B Tinker run: medium reasoning,
temperature 0.7, up to 8,192 output tokens, at most five samples per target.

The historical Gemini verdict tested equivalence to the fixed reference; it did
not certify independent correctness. It was not relabeled or treated as a new
validity verdict. An offline Codex review instead approved explicit cached
attempts against their original prompts, allowing different creative content and
implementations that satisfy the requested behavior. Each adopted answer uses the
reasoning from that same saved response, without rewriting either field.

This is a conservative subset, not exhaustive correctness labeling of all 4,075
receipts. Candidate nomination used 194 rows with relevant historical judge
explanations and 95 additional rows with a completed answer of at most 1,800
characters (the shortest cached candidate). Selected longer creative responses
were also read. Nomination is therefore length-sensitive, and the remaining
508 targets may include further valid alternatives. They are labeled unresolved,
not all incorrect. No new automated validity-judge pass was run.

Examples adopted include different slogans and creative stories, a French
translation, a supplied-paragraph claim extraction, and a nonempty-string checker
that rejects non-string values. Reasons to retain other candidates included
misplaced paragraph separators, invented biographical/clinical details, and
incorrect explanations such as Python `round()` being described as half-up.

## Results and verification

- 119 new cached CoT/answer pairs adopted: 90 Tulu, 16 self-OSS, 13 LIMA.
- All 119 final answers changed; 101 CoT strings changed and 18 were identical.
- The previous 446 CoT-only acceptances remain untouched: 565 accepted targets
  overall, with 508 retained originals explicitly listed.
- All 10,000 rows and 10,380 assistant turns remain; all 9,307 untraced assistant
  turns, all prompts/tools, and all unselected messages are unchanged.
- Every replacement is the terminal assistant turn of its row, so it cannot make
  an unchanged later user/assistant exchange inconsistent with a changed answer.
- 159 local prompt-constraint and code-behavior checks passed, alongside full
  corpus integrity checks and the existing 32 backfill/Harmony tests. Local checks
  and assistant review do not constitute a formal proof of correctness.
- HF JSON loading and full native Harmony rendering passed. Supervised tokens:
  2,352,071, up from 2,341,177 (+10,894). Maximum rendered sequence: 8,080 tokens;
  no datum exceeds 8,192.
- Additional generator/judge calls: zero. Additional provider inference cost:
  $0. No training, evaluation, or rented infrastructure was started.

All uploaded current-stage files were downloaded at the published revision and
byte-compared. Dataset SHA256:
`d792bcbb3e9ed62d6b02b010db3add413e931a142ce31c824ca39c7143d91b6c`.

Current manifests: `cached_answer_replacements.jsonl`,
`cached_answer_review_scope.json`, `cached_answer_checks.json`,
`replacement_verification.json`, and `failed_replacements.{jsonl,md}` plus the
compact `failed_replacement_indices.json`. Original stage summaries were archived
under `prior_strict_replacement/`; raw receipts and historical attempt ledgers
remain unchanged. The previous dataset is still accessible at its parent revision.

Reproduction code/config: `scratch/gptoss_control/run.py` stages
`adopt-cached-answers` and `publish-cached-answers`,
`scratch/gptoss_control/cached_answer_review.yaml`, and
`scratch/gptoss_control/check_cached_answers.py`, committed as `b9562bd4`.
