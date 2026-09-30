<!-- ABOUTME: Restore the 119 original nosynth answers with explicitly answer-conditioned GPT-OSS rationales. -->
<!-- ABOUTME: Exact dataset pins, generation protocol, validation, and source-quality limitations. -->

# Original answers restored with answer-conditioned rationales

The new 10,000-row mixture restores all 119 original final answers that the previous
native-CoT reconciliation had replaced. Base GPT-OSS-120B authored a rationale for
each, having seen the original conversation and that exact answer. There are no
unresolved replacements. This dataset is available for the next control training;
the already-trained control and its ODCV results remain tied to their earlier data.

Dataset: [dougalldeepmind/2026-09-30-nosynth-gptoss-answer-conditioned-mix](https://huggingface.co/datasets/dougalldeepmind/2026-09-30-nosynth-gptoss-answer-conditioned-mix)
at **`8d54d4c5e1983955925fb2a350dad6fff07c7531`**.

| Item | Count |
|---|---:|
| Dataset rows | 10,000 |
| Rows with reasoning | 1,061 |
| Assistant turns with reasoning | 1,073 |
| Original answers restored with new conditioned rationales | 119 |
| Existing GPT-OSS reasoning turns left unchanged | 954 |
| Untraced assistant turns left unchanged | 9,307 |
| Unresolved target turns | 0 |

The 119 restored rows comprise 90 `tulu3_if`, 16 `self_oss_instruct`, and 13 `lima`.
Every message's content now matches the original pinned mixture; only the 119
target answer/reasoning pairs changed relative to the immediately preceding dataset.
No untraced turn received reasoning, and no new tool calls were added.

## What was generated

Pipeline: original conversation prefix and fixed original answer → base
`openai/gpt-oss-120b` on Tinker → authored rationale in the generator's final JSON
`reasoning` field → separate compatibility judgment → `reasoning_content` placed
before the unchanged original answer during Harmony rendering.

These 119 fields are **answer-conditioned authored rationales**, not independently
elicited native analysis and not recovery of an earlier private reasoning process.
The generator's own analysis was retained in receipts but was not inserted into
the dataset. Existing reasoning was removed from the supplied conversation prefix
to avoid copying earlier traces. Original system/user messages and tools were
included as quoted example data; no tool was executed during this task.

Main generation used medium reasoning, temperature 0.7, an 8,192-token completion
cap, and deterministic seeds `row * 10 + attempt`. A final five-row correction used
high reasoning and temperature 0.5 with concise prospective plans. All generation
used the base model, never the control LoRA. Full messages, token IDs, output,
sampling configuration and each selection are in the published receipts/configs.
Tinker does not expose an immutable base-weight revision in these receipts.

The separate judge was `google/gemini-3.1-pro-preview` through OpenRouter, low
reasoning. Initial and revised judge prompts were checked on positive/negative
examples; the final checker passed all 15 cases. Acceptance requires substantive
answer agreement and a truthful compatible rationale. Manual spot checks overrode
12 judge acceptances across six rows. This is quality screening, not a proof that
all explanations or original source answers are correct.

There were 218 base-model generation requests including the discarded nine-row
initial pilot; 119 were selected. No initial-pilot rationale was selected. Across
all stages, 46 qualification judgments were retained. Generation and judging
estimates including pilots/retries were **$0.11785 + $0.72565 = $0.84349**.
Per-call receipts settle reservations and preserve all rejected attempts.

## Source-answer imperfections

The user requested exact original answers, so existing mistakes were not repaired.
Eleven rows have documented source limitations, including four words instead of
three, missing requested text, a misspelled football nickname, `min` incorrectly
described as `max`, and incomplete Python input validation. Rationales explain the
actual method or substantive choice without falsely defending those mistakes.

Some automatic flags were false: row 7016 contains no commas, and row 9678 has 27
words rather than exceeding 30. Direct inspection corrected these flags. The
published `source_answer_review.json` separates verified limitations from false
automatic reports. `quality_review_notes.json` and historical receipts retain the
earlier guidance rather than hiding failed attempts. Row indices are zero-based.

## Verification and provenance

- Previous mixture: `dougalldeepmind/2026-09-30-nosynth-mix-gpt-oss-120b`
  at `4ffd1f931ebb34b25c68c4b784059f98c1617add`.
- Original answer source: `dougalldeepmind/2026-09-22-nosynth-mix-gpt-oss-120b`
  at `d74e42e4df0a0eb293cbaa4ab8cd58e08725e9da`.
- Publisher/code commit: `c0ff30e2`; implementation and stage configs in
  `scratch/gptoss_control/answer_conditioned*`.
- All 10,000 rows passed HF JSON loader equality and full Harmony rendering and
  supervision checks. There are 10,380 training datums, 5,975,184 processed tokens,
  and 2,329,758 supervised tokens; none exceeds 8,192 tokens.
- The published mixture, replacement verification and CoT manifest were downloaded
  at the new immutable revision and checked byte-for-byte against local output.
- `restored_examples.jsonl` contains the 119 questions, original answers and chosen
  rationales together. `cot_provenance.jsonl` classifies all 1,073 reasoning turns.
  `failed_replacements.json` is empty; rejected attempts remain in `receipts/`.

The full run used the isolated GPT-OSS worktree. There was no additional model
training, ODCV evaluation, GPU rental, or change to the current control pointer.

Final release command (earlier stages are preserved in history/configs):

```powershell
uv run --project src/infra/endpoints/tinker_runtime python scratch/gptoss_control/answer_conditioned.py publish --config scratch/gptoss_control/answer_conditioned_release.yaml
```

The publisher intentionally refuses to overwrite an existing dataset. Reproduction
should use a new output identity and preserve prior paid-call receipts.
