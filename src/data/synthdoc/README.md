<!-- ABOUTME: synthdoc -- a stage-for-stage replication of the Teaching Claude Why difficult-advice pipeline, -->
<!-- ABOUTME: plus the MEM (model-evaluates-model) pipeline that runs over a completed run's outputs. -->

# synthdoc

A faithful replication of the difficult-advice pipeline from
[Teaching Claude Why](https://alignment.anthropic.com/2026/teaching-claude-why/), with each
of the six stages a separate, separately-cached step — plus the **MEM pipeline**
(below), which generates model-evaluates-model documents over a completed run.

```
uv run synthdoc segment                                  # stage 1 only, no API calls
uv run synthdoc run --config configs/data/synthdoc.yaml --smoke
uv run synthdoc run --config configs/data/synthdoc.yaml
uv run synthdoc estimate --config configs/data/synthdoc.yaml \
    --measured output/synthdoc_v2/<run>/manifest.json
uv run synthdoc mem --config configs/data/mem.yaml --smoke
uv run synthdoc mem --config configs/data/mem.yaml
uv run synthdoc check --config configs/data/mem.yaml --run_dir output/mem/<ts>
```

Output and HF cache names keep their historical `synthdoc_v2` / `synthdoc-v2` prefixes so
existing run snapshots stay resumable.

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

## MEM: model-evaluates-model

`uv run synthdoc mem` consumes a **completed** difficult-advice run (its
`stage_6_final.jsonl`, locally or from the HF mirror) and generates documents in which
the model reasons about a response to one of those scenarios and works out whether it
was the right call. Sharing the scenario bank is what makes arm differences
attributable to *format* rather than content; `run_mem` fail-fasts if the source run's
constitution sha differs from the config's.

The bet is on the reasoning, not the verdict, so every prompt enforces the same shape:
start from the situation, work toward the principle, entertain a consideration on the
other side, earn any conclusion at the end. Each record also gets an *explicitness*
style (name the value / paraphrase it / embody it silently) assigned by the planner.

Cells (`CELLS` registry in `stages.py`; a cell = attribution × response quality):

| Cell | What the document is | Status |
|---|---|---|
| `control` | The original exchange, gold response verbatim, only the reasoning trace regenerated — extended and constitution-grounded. No evaluation framing: the arm that tests whether reasoning depth alone explains any MEM effect. | this pass |
| `m4_other_good` | The transcript (neutrally attributed to "an AI assistant") sits in the user turn; the assistant reasons about it, assesses it, and gives its own answer. Verdict comes from a trailing `<assessment>` scaffold tag — stripped from training text, kept as metadata. | this pass |
| `m3_other_flawed` | Same, over a minimal-pair perturbed response (one flaw: omission / commission / miscalibration / over-application × severity). Critique generation is blind to the flaw label. | later pass |
| `m2_self_good` / `m1_self_flawed` | Multi-turn self-reflection: the response sits in the model's own prior turn (masked at training via `supervise: "final"`), a reflection prompt follows, and the model revises or holds with reasons. | later pass |

Stages (numbering reserves 3 for perturbation): `1 source → 2 plan (deterministic:
cell allocation, explicitness, wrapper variant; record_id = "<scenario_id>::<cell>") →
[3 perturbed] → 4 generated → 5 sft`. Same caching, checkpointing, budget guard and HF
mirroring as `run`; `--smoke` clamps every enabled cell to 2 documents.

`uv run synthdoc check` runs the corpus validity checks (`checks.py`) and gates on the
config's thresholds: coverage vs plan, template collapse (repeated 8-grams), verdict
distribution (non-degenerate), post-hoc-reasoning rate (heuristic + judged sample),
blindness (flaw labels provably absent from prompts and training text), and LLM-judged
gold-response validation. `checks_report.json` lands in the run dir either way.

## Related

The closest published description of this recipe is
[Synthetic document finetuning for instilling positive traits](https://www.lesswrong.com/posts/GTYJRLhqztxKF2v5R/synthetic-document-finetuning-for-instilling-positive-traits).
