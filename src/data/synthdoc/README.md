<!-- ABOUTME: synthdoc -- a stage-for-stage replication of the Teaching Claude Why difficult-advice pipeline. -->
<!-- ABOUTME: Explains each stage, which model runs it, what gets injected, where each stage is cached, and the flavors. -->

# synthdoc

A faithful replication of the difficult-advice pipeline from
[Teaching Claude Why](https://alignment.anthropic.com/2026/teaching-claude-why/), with each
of the six stages a separate, separately-cached step.

```
uv run synthdoc segment                                  # stage 1 only, no API calls
uv run synthdoc run --config configs/data/synthdoc.yaml --smoke
uv run synthdoc run --config configs/data/synthdoc.yaml
uv run synthdoc estimate --config configs/data/synthdoc.yaml \
    --measured output/synthdoc_v2/<run>/manifest.json
```

Output and HF cache names keep their historical `synthdoc_v2` / `synthdoc-v2` prefixes so
existing run snapshots stay resumable.

## Flavors

The six-stage runner, checkpointing, retrying and cost accounting are shared. Everything that
differs between corpora lives in `flavors/`, selected by `flavor:` in the run config:

| flavor | config | what a record is |
|---|---|---|
| `difficult_advice` | `configs/data/synthdoc.yaml` | a **user** under pressure asks for help toward a norm-violating shortcut; the assistant reasons about its values and declines. The original recipe. |
| `self_reflection` | `configs/data/synthdoc_self_reflection.yaml` | **the agent itself** is the one tempted: working autonomously, it finds it could protect its own position by a deniable wrong action, deliberates in its own value language, and does the legitimate thing anyway. |

A flavor is a module implementing `plan / scenario_call / scenario_records / draft_call /
apply_draft / refine_call / apply_refine / respond_call / apply_respond / rewrite_call /
apply_rewrite / to_sft`, plus an optional `validate_rewrite`; see `flavors/__init__.py` for
the contract. Each `*_call` returns `(system, user)` for a JSON-bodied stage or
`(system, user, keys)` for a tagged-block stage — long prose in JSON breaks on the first
unescaped quote, which is why `self_reflection` uses tags from stage 3 onward.

### The self_reflection flavor

Built to teach values rather than rules, so its stage 6 enforces a **voice contract**: the
deliberation may never name or number a principle, cite a constitution or guidelines, or use
the vocabulary of permission (`allowed`, `not permitted`, `I must not`). `validate_rewrite`
lints every trace against `BANNED_IN_REASONING` and rejects-and-retries the completion, so the
ban is enforced per record rather than merely requested in a prompt.

Corpus composition is set by the config's `mix` and assigned **deterministically from the
scenario id**, so a resumed or re-run stage reproduces it exactly:

- `form` — 80% prose, 20% tool-call transcripts. Both give the agent a real option to act.
- `multi_turn` — 15% continue one exchange further, the colleague's reply deniably offering
  the trade the agent already declined to initiate.
- `control` — 12% where self-interest argues for **silence** and the right move is to act
  anyway. Without this slice the corpus teaches "never act" instead of "discount your own
  interest".
- `motive` — replacement / restriction / goal_conflict.

Traits are **weighted**, not uniform (`trait_weights`): every principle appears, but the ones
that actually fire when an agent's position is threatened carry the corpus. Industries are
**assigned** per batch from `INDUSTRIES` rather than chosen by the model — independent batches
left to choose all gravitate to the same few settings.

## Why v1 was replaced

This package started as `synthdoc_v2`, replacing the original config-driven `synthdoc`
package (deleted 2026-08-03; it lives in git history before this date). v1 collapses three
of the six stages into one call. Its `difficult_advice` doc type says
*"Write an exchange in which a person under real pressure asks for help…"*, so a single
generation produces the user prompt **and** the assistant response together; its stages are
`plan → generate → revise×N → filter`. There is no draft-prompt stage, no refine-prompt
stage, and its revise step rewrites the whole document rather than rewriting the response
against a named constitution section.

| Blog stage | v1 | v2 |
|---|---|---|
| 1. Segment the constitution | chunker over the spec | `constitution.segment()` → 8 traits |
| 2. Generate scenarios | optional `plan` stage | stage 2, per trait |
| 3. Draft the prompt | **merged into generate** | stage 3, its own call |
| 4. Refine the prompt | **absent** | stage 4, its own call |
| 5. Generate initial response | **merged into generate** | stage 5, its own call |
| 6. Rewrite against constitution | `revise` (whole doc) | stage 6, response vs. target trait |

## The stages

| # | Stage | Model | What is injected | Output |
|---|---|---|---|---|
| 1 | Segment the constitution | — (deterministic) | — | 8 traits |
| 2 | Generate scenarios | `gpt-5.6-luna` | the one target trait | situation + tempting shortcut |
| 3 | Draft the prompt | `gpt-5.6-luna` | the scenario | `system` + `user` draft |
| 4 | Refine the prompt | `gpt-5.6-terra` | **full constitution + target trait** | refined `system` + `user` |
| 5 | Generate response | `gpt-5.6-luna` | target trait + style guidance | `reasoning` + `response` |
| 6 | **Rewrite (critical)** | `gpt-5.6-terra` | **full constitution + target trait + whole transcript** | final `reasoning` + `response` |

Stage 1 is deterministic on purpose: the trait set must be identical across runs, otherwise
per-trait comparisons are meaningless. It parses the numbered principles and keeps the
"what an aligned response looks like" prose separately, as shared style guidance injected at
stage 5 — that section constrains tone rather than naming a distinct value.

Stage 2 batches its requests (`scenarios_per_call`). One call asking for 40 situations
would overrun `max_tokens` and truncate its JSON; every stage asserts on a `length` finish
reason rather than failing later with a confusing parse error.

## Output

Each stage writes a complete snapshot, so any stage can be re-run alone by deleting its file
and any two runs can be diffed stage by stage:

```
stage_1_traits.jsonl          stage_5_draft_responses.jsonl
stage_2_scenarios.jsonl       stage_6_final.jsonl
stage_3_draft_prompts.jsonl   stage_7_sft.jsonl        <- training-ready
stage_4_refined_prompts.jsonl manifest.json
```

Every file is also pushed to the HF dataset repo named in the config. A run that is
interrupted, or that trips `budget_usd`, resumes from the last completed stage at no cost.

The training record carries the trait in metadata, as required:

```json
{"messages": [{"role": "system", ...}, {"role": "user", ...},
              {"role": "assistant", "content": "...", "reasoning_content": "..."}],
 "metadata": {"trait_id": "t1", "trait_name": "Honesty and non-deception",
              "trait_text": "...", "scenario_id": "...", "domain": "...",
              "shortcut": "...", "situation": "..."}}
```

## Related

The closest published description of this recipe is
[Synthetic document finetuning for instilling positive traits](https://www.lesswrong.com/posts/GTYJRLhqztxKF2v5R/synthetic-document-finetuning-for-instilling-positive-traits).
