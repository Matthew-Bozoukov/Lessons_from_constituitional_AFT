<!-- ABOUTME: synthdoc -- one constitution-grounded generation pipeline; the config's -->
<!-- ABOUTME: `pipeline:` field picks the document type (difficult_advice | mem). -->

# synthdoc

Constitution-grounded synthetic-data generation. There is **one pipeline entrypoint**;
which document type a run generates is declared in its config's `pipeline:` field, so
the config stays the complete scientific record:

```
uv run synthdoc segment                                                      # constitution -> traits, no API calls
uv run synthdoc run      --config configs/data/difficult_advice.yaml [--smoke]
uv run synthdoc run      --config configs/data/mem.yaml [--smoke]
uv run synthdoc estimate --config <either> [--measured <run>/manifest.json]
uv run synthdoc topup    --config configs/data/difficult_advice.yaml --resume DIR --traits t5,t6 --n 25
uv run synthdoc check    --config configs/data/mem.yaml --run_dir output/mem/<ts>
```

(`topup` applies to difficult-advice runs only; `check` to mem runs only — both verbs
validate the config's `pipeline:` field.)

Flat module layout, one concern per file:

```
core.py          shared machinery: priced Usage tallies, parse-retrying LLM calls
                 (call_json/call_tagged), resilient fan-out, per-item Checkpoints,
                 model_cfg, measured estimates
constitution.py  deterministic segmentation into traits + shared style guidance
hf_cache.py      StageCache: local stage_<n>_<name>.jsonl snapshots + HF dataset mirror
prompts.py       every prompt template, sectioned by document type (string constants --
                 deliberately not YAML-templated, so wording is reviewable)
stages.py        every stage function: difficult-advice stages 2-6 + the MEM cells
pipeline.py      run_difficult_advice / run_mem + the dispatching run()
estimate.py      per-type cost models + the dispatching estimate()
checks.py        MEM corpus validity checks
cli.py           the verbs above
```

Every run follows the same operating contract: each stage writes a complete local
snapshot and mirrors it to the HF repo named in its config, so any stage can be re-run
alone by deleting its file and an interrupted or budget-capped run resumes from the last
completed stage at no cost; the expensive stages also checkpoint per item. `--smoke`
runs the full wiring on a tiny slice into a separate `smoke_<ts>` dir and smoke HF repo.
A `manifest.json` records git sha, constitution sha256, effective sizes, per-stage usage
and wall clock; `estimate --measured <smoke manifest>` prices a full run from real
per-call token counts.

## Document type: difficult advice (`pipeline: difficult_advice`)

A faithful replication of the difficult-advice recipe from
[Teaching Claude Why](https://alignment.anthropic.com/2026/teaching-claude-why/).
Output and HF cache names keep their historical `synthdoc_v2` / `synthdoc-v2` prefixes so
existing run snapshots stay resumable.

| # | Stage | What is injected | Output |
|---|---|---|---|
| 1 | Segment the constitution | — (deterministic) | traits |
| 2 | Generate scenarios | the one target trait | situation + tempting shortcut |
| 3 | Draft the prompt | the scenario | `system` + `user` draft |
| 4 | Refine the prompt | **full constitution + target trait** | refined `system` + `user` |
| 5 | Generate response | target trait + style guidance | `reasoning` + `response` |
| 6 | **Rewrite (critical)** | **full constitution + target trait + whole transcript** | final `reasoning` + `response` |

Stage 1 is deterministic on purpose: the trait set must be identical across runs,
otherwise per-trait comparisons are meaningless. It parses the numbered principles and
keeps the "what an aligned response looks like" prose separately, as shared style
guidance injected at stage 5. Stage 2 batches its requests (`scenarios_per_call`) so a
call never overruns `max_tokens` and truncates its JSON. Stage 7 (`to_sft`) is a free
export to `{messages, metadata}` records with the trait carried in metadata:

```json
{"messages": [{"role": "system", ...}, {"role": "user", ...},
              {"role": "assistant", "content": "...", "reasoning_content": "..."}],
 "metadata": {"trait_id": "t1", "trait_name": "...", "trait_text": "...",
              "scenario_id": "...", "domain": "...", "shortcut": "...", "situation": "..."}}
```

This replaced the original config-driven `synthdoc` v1 (deleted 2026-08-03; it collapsed
three of the six stages into one call and had no draft/refine split — see git history
before that date).

## Document type: MEM, model-evaluates-model (`pipeline: mem`)

Generated over a **completed** difficult-advice run (its `stage_6_final.jsonl`, locally
or from the HF mirror): documents in which the model reasons about a response to one of
those scenarios and works out whether it was the right call. Sharing the scenario bank
is what makes arm differences attributable to *format* rather than content; the runner
fail-fasts if the source run's constitution sha differs from the config's.

The bet is on the reasoning, not the verdict, so every prompt enforces the same shape:
start from the situation, work toward the principle, entertain a consideration on the
other side, earn any conclusion at the end. Each record also gets an *explicitness*
style (name the value / paraphrase it / embody it silently) assigned by the planner.

Cells (`CELLS` registry in `stages.py`; a cell = attribution × response quality):

| Cell | What the document is |
|---|---|
| `control` | The original exchange, gold response verbatim, only the reasoning trace regenerated — extended and constitution-grounded. No evaluation framing: the arm that tests whether reasoning depth alone explains any MEM effect. |
| `m4_other_good` | The transcript (neutrally attributed to "an AI assistant") sits in the user turn; the assistant reasons about it, assesses it, and gives its own answer. Verdict comes from a trailing `<assessment>` scaffold tag — stripped from training text, kept as metadata. |
| `m3_other_flawed` | Same, over a minimal-pair perturbed response (stage 3: one flaw — omission / commission / miscalibration / over-application × clear/moderate/grey severity — length/register held, flaw label metadata-only). Critique generation is blind to the flaw label. |
| `m2_self_good` / `m1_self_flawed` | Multi-turn self-reflection — the headline cells. The response sits in the model's own prior assistant turn (attribution is structural, and it carries **no** think block, matching inference-time history rendering), one of ~6 reflection prompts (gentle → pushback) follows, and the model revises or holds with reasons — both outcomes real (`<assessment>`: `revised`/`held`). Trained with `supervise: "final"`: only the last turn carries loss. |

M5 is a mixture of cells, not a cell. Stages: `1 source → 2 plan (deterministic: cell
allocation, explicitness, flaw type×severity grid, wrapper/reflection variants;
record_id = "<scenario_id>::<cell>") → 3 perturbed (flawed cells only) → 4 generated →
5 sft`.

The self cells' per-turn masking is threaded end-to-end: `to_mem_sft` stamps
`metadata.supervise`, `convert_synthdoc_qwen.py` carries it onto the rendered row,
`build_mixture` passes it through (`format: rendered`), and `train_lora`/`masking.py`
apply it (`assistant_spans(supervise="final")` keeps only the last assistant turn).

`synthdoc check` runs the corpus validity checks (`checks.py`) and gates on the config's
thresholds: coverage vs plan (incl. the flaw grid), template collapse (repeated 8-grams),
per-cell verdict distribution (each cell mostly reaches its expected verdict, never
100% — all-`revised` in m1 would train capitulation), post-hoc-reasoning rate (heuristic
+ judged sample), blindness (flaw labels provably absent from prompts and training
text), the surface-shortcut classifier (numpy logistic regression must NOT predict
good-vs-flawed from the response text alone), LLM-judged gold-response validation and
flaw-identification rate (blind critiques must find `clear` flaws). `checks_report.json`
lands in the run dir either way.

## Adding a document type

Add its prompts as constants in `prompts.py`, its stage functions in `stages.py` (built
on `core.call_tagged`/`core.run_items`), a `run_<name>` in `pipeline.py` registered in
`PIPELINES`, a cost model in `estimate.py` registered in `_ESTIMATORS`, and a config in
`configs/data/<name>.yaml` declaring `pipeline: <name>`. Nothing in `core.py` may know
about any specific document type.

## Related

The closest published description of this recipe is
[Synthetic document finetuning for instilling positive traits](https://www.lesswrong.com/posts/GTYJRLhqztxKF2v5R/synthetic-document-finetuning-for-instilling-positive-traits).
