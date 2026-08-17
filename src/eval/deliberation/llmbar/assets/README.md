<!-- ABOUTME: Provenance for the vendored LLMBar evaluator prompt — where it came from and -->
<!-- ABOUTME: what was deliberately changed, so the port stays auditable against upstream. -->

# Vendored from LLMBar

`vanilla.txt` is byte-identical to upstream's `Vanilla` comparison prompt:

- repo: <https://github.com/princeton-nlp/LLMBar>
- path: `LLMEvaluator/evaluators/prompts/comparison/Vanilla.txt`
- pinned commit: `900616bff90b6c6c8e1681f7d079250637c55992` (2024-07-08)
- retrieved: 2026-08-17
- paper: Zeng et al., *Evaluating Large Language Models at Evaluating Instruction
  Following*, ICLR 2024 ([arXiv:2310.07641](https://arxiv.org/abs/2310.07641))

The `<|im_start|>` / `<|im_end|>` markers are upstream's own ChatML rendering; `prompts.py`
splits them back into system and user messages rather than sending them as literal text,
because our target is reached through the chat API and would otherwise see the markers as
content.

## Two deliberate deviations, both forced by the model family

1. **Token budget.** Upstream caps the evaluator at `max_tokens: 50` and instructs "Do NOT
   provide any explanation". Neither suppresses a `<think>` trace on a reasoning model, so
   a 50-token cap truncates inside the trace and scores a false 0% (CLAUDE.md gotcha 4).
   The prompt is kept verbatim — it is the benchmark — and the budget is set in
   `configs/eval/llmbar.yaml` instead, with the answer parsed after `</think>`.
2. **No logit steering.** Upstream's config passes `tokens_to_avoid` / `tokens_to_favor`
   to nudge the evaluator away from "Both"/"Neither". vLLM's OpenAI server exposes no
   equivalent, and applying it to one arm and not another would confound the comparison.
   Refusals to choose are therefore *counted* (`unparsed`) rather than suppressed, which
   is the more informative failure anyway.

Both deviations shift absolute accuracy away from the paper's published numbers. This eval
exists for arm-vs-arm contrast on identical items, which they do not affect.
