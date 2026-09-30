# Existing GPT-OSS nosynth CoT replacement, 2026-09-30

The user requested independent base GPT-OSS-120B completions from each original
conversation prefix, judging both generated-answer agreement and CoT compatibility,
then retaining the original answer with accepted generated CoT. The user clarified:
replace only existing nonempty CoTs, leave all untraced entries untouched, and keep
an explicit list of failed replacements after at most five sampling attempts.

## Frozen input and protocol

Source: `dougalldeepmind/2026-09-22-nosynth-mix-gpt-oss-120b` at
`d74e42e4df0a0eb293cbaa4ab8cd58e08725e9da`. This is the already native-GPT-OSS
version of the nosynth mixture, not the Qwen version. Its 1,073 existing traces
were targeted: 546 Tulu, 379 self-OSS, 148 LIMA turns. All 9,307 untraced assistant
turns, including tool-call targets, were deliberately outside replacement scope.

Generation used base `openai/gpt-oss-120b` on Tinker, without an adapter, medium
reasoning effort, temperature 0.7, 8,192 output tokens, deterministic per-row/attempt
seeds, and the shared native Harmony renderer. Reference answers were not supplied
to the generator. Original prompts, answers, tools, row order and counts remain
fixed. Tinker does not expose a weight-revision pin; the tokenizer is pinned at
`b5c939de8f754692c1647ca79fbf85e8c1e70f8a`.

Final judge: `google/gemini-3.1-pro-preview` through the pinned Google AI Studio
provider, temperature 0, low reasoning effort, 4,096 output tokens. Both
`answer_agreement` and `trace_compatible` must be boolean true, and the judge
must finish normally. Invalid or non-object JSON is rejected. Code differences
in sorting, empty-input behavior and comparison boundaries count as disagreement.

## Results

- 446 targets accepted: 384 CoT strings changed and 62 regenerated identically.
- 627 targets did not pass within five attempts; their full original assistant messages,
  including their existing CoT, remain unchanged.
- Accepted / retained failures by source: Tulu 233 / 313; self-OSS 180 / 199;
  LIMA 33 / 115.
- All 10,000 rows and 10,380 assistant turns remain. Every non-CoT field,
  all untraced messages and all failed entries compare identically to the source.
- 4,075 attempt receipts: 4,062 completed, six interrupted requests without a
  recoverable response, seven earlier parser errors. Failure records distinguish
  these technical outcomes from explicit judge disagreement.
- Final supervised token count: 2,341,177, versus 2,341,585 before replacement.
  Every rendered datum fits 8,192 tokens (configured safety maximum 32,768).
- Recorded upper-bound accounting: generation $4.70160414; judging $34.156743;
  qualification $0.12998; total $38.98832714. Includes discarded judge runs and
  conservative reservations for uncertain requests, not a billing invoice.

## Qualification, recovery and limits

The initial broad pilot and early Sonnet judging produced false rejections and a
material false acceptance (case-insensitive sorting replacing case-sensitive
sorting). Those verdicts were not accepted as final. Saved generations were
rejudged by the final rubric/model. Expanded qualification passed 15 expected
verdicts, including the observed sorting false acceptance; this is a small sanity
check, not a proof of judge accuracy. Two prior constrained-summary examples were
also inspected; they are not in this replacement scope.

The judge occasionally returned a JSON list, exposing an initial parser error.
The parser now rejects non-object verdicts. Five such saved attempts for row 261
were rejudged after the fix; none passed. No additional model samples were used
for that recovery. Five explicitly unsent judge requests from the budget stop
were also completed without regenerating their model outputs. The judge ceiling
was increased from $30 to $45 on continuation, with the generation ceiling $10;
final recorded accounting remained below $40 overall.

The failure list uses zero-based source row and assistant-message indices, includes
source hashes, all attempt outcomes, judge explanations and receipt references.
Failed entries were not silently stripped of reasoning or removed from the mix.
This update does not add tool CoT, does not retrain an adapter, and does not establish
that the previous ODCV zero-CoT behavior or tool-formatting issue has changed.

## Validation and artifacts

32 offline checks passed. Full Harmony rendering/HF JSON loading passed for every
row. `scratch/gptoss_control/verify_replacements.py` independently verified all
replacement and preservation invariants, generated the readable failure table,
and wrote a consolidated raw attempt ledger.

Destination: https://huggingface.co/datasets/dougalldeepmind/2026-09-30-nosynth-mix-gpt-oss-120b
Published revision: `647743fc260a69bc9ad623a4288127f40e9ed463`

Files: `mixture.jsonl`, `replacement_verification.json`,
`failed_replacement_indices.json`, `failed_replacements.jsonl`,
`failed_replacements.md`, `changed_trace_indices.json`,
`backfill_attempts_all.jsonl`, raw receipts, qualification/recovery logs,
source provenance, renderer/code snapshots and resolved config.

Dataset SHA256: `0356d9d82f26e123812352d59a70169e9bb3dd5c752ba1ca008373e4b590e94f`.

Publication verification: dataset byte-roundtrip plus 10 critical files (including the
complete attempt ledger and failure lists) byte-matched at the pinned revision; all
4,285 staged paths are present remotely. No inference process or rented GPU remains
from this replacement task.
