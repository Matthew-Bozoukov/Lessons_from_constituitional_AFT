<!-- ABOUTME: synth -- one config-driven generation pipeline. The config's `stages:` list -->
<!-- ABOUTME: (prompts included) defines the document type; code supplies generic operators. -->

# synth

Constitution-grounded synthetic-data generation. There is **one engine and one
entrypoint**; everything specific to a document type — its stage sequence, its prompts,
its models, its knobs — lives in that type's config, so the config alone is the complete
scientific record of what a run generated:

```
uv run scripts/data/synth/build_dataset.py --config configs/data/synth/difficult_advice.yaml [--smoke]
uv run scripts/data/synth/build_dataset.py --config configs/data/synth/model_eval_model.yaml [--smoke]
uv run scripts/data/synth/build_dataset.py --config <cfg> --ablate final     # ablation arm
uv run scripts/data/synth/build_dataset.py --config <cfg> --estimate [--measured <smoke manifest>]
```

(`uv run synth run|topup|check|estimate|segment|chunkings` remains as the console script
for the auxiliary verbs; `run`/`estimate` are the same functions `build_dataset.py` calls.)

## Architecture

```
pipeline.py    the engine: builds Stage objects from the config's `stages:` list; owns
               snapshot caching + HF mirroring, per-item checkpoints, ablation, the
               budget guard, the manifest, and cost estimates
operators.py   the operator library -- every stage `kind:` a config may use
core.py        LLM machinery: priced Usage, call_json/call_tagged (parse-retry),
               resilient fan-out, Checkpoint/run_items, Ctx + Stage dataclasses
cells.py       model-eval-model cell STRUCTURE (registry, planning, perturbation,
               assembly) -- all its wording comes from the config
checks.py      corpus validity checks (judge wording from the config's checks.judges)
constitution.py  hf_cache.py  cli.py
```

A config's `stages:` entry names an operator `kind` and supplies everything it needs:

| kind | what it does | key fields |
|---|---|---|
| `segment` | deterministic constitution chunking + grouping; publishes `{style_guidance}` | (top-level `chunking:` block — see below) |
| `scenarios` | batched JSON fan-out per trait (`t<i>_b<b>_s<j>` ids) | `model`, `prompts` |
| `llm_json` | one JSON call per record | `model`, `prompts`, `save`, `optional`, `checkpoint` |
| `llm_tagged` | one tagged-blocks call per record | `model`, `prompts`, `tags`, `save`, `checkpoint`, `ablate_with`, `prompt_vars` (conditional template vars), `variants_by` (per-record user/tags/save), `lint` (ban-patterns + min-length, reject-and-retry) |
| `chat_export` | free export to `{messages, metadata}`; entries may carry `when:` for multi-turn records | `messages`, `metadata` |
| `scenarios_weighted` | weighted trait apportionment, control slice, motive rotation, per-batch industries, deterministic per-scenario variants | `model`, `prompts` (+`control_user`), `threats`, `control_threats`, `fields` |
| `load_source_run` | a completed run's finals + constitution-sha provenance check | (`source:` block) |
| `plan_cells` / `perturb_pairs` / `generate_cells` / `revise_cells` / `assemble_cells` | the model-eval-model cells (see below); `revise_cells` is the constitution-grounded rewrite pass (verdict pinned, control passes through) | (`cells:`, `flaws:`, `prompts:` blocks; `revise_cells` takes `model`, `prompts`, `checkpoint`, `ablate_with`) |

Prompt templates in configs are `str.format` templates over record fields plus shared
vars (`{constitution}`, `{style_guidance}`). Literal JSON braces are escaped `{{ }}`.

## Chunking: how the constitution becomes the units documents are built against

A dataset config names a **chunking method** and nothing else:

```yaml
chunking: principle     # uv run synth chunkings  lists them
```

Under the hood that is two deterministic, offline, no-LLM steps in `constitution.py` —
**chunk** the document at a granularity, then **group** chunks into units. A unit renders
to exactly the fields the rest of the pipeline already consumes
(`trait_id`/`index`/`name`/`text`), so nothing after stage 1 knows chunking exists.

| method | units from the 9-principle mid constitution | |
|---|---|---|
| `principle` | 9 — one numbered principle each | **default**; the Teaching Claude Why recipe, and what every corpus here was generated with |
| `paragraph` | 36 — statement / rationale / exceptions stand alone | |
| `bullet` | 45 — one bullet or paragraph each | the finest cut; GDM's choice |
| `whole` | 1 — the entire constitution | no chunking at all |
| `principle_pairs_adjacent` | 5 — two consecutive principles | |
| `principle_pairs_random` | 5 — two unrelated principles | |
| `principle_pairs_related` | 5 — two similar principles | |
| `paragraph_clusters` | 4 — paragraphs regrouped semantically | the embed-and-cluster shape |

Methods are defined in `CHUNKINGS` (`constitution.py`), one frozen `Chunking` each.
**Settings live with the method, not in the config**: a config carries a name, so a run
manifest records *which recipe ran* rather than an anonymous bag of knobs. Add a method
by adding an entry; an unrecognised name fails fast and lists the options.

Preview any method for free — offline, no API key. It also prints the spec-side
measurements (words/unit, chunk centrality, and how much preamble belongs to no unit):

```
uv run synth chunkings
uv run synth segment --chunking bullet
uv run synth segment --chunking principle_pairs_related
uv run synth segment --constitution constitutions/claude_distilled_24_principles_fine/constitution.md
```

Two invariants make methods comparable, and both are tested:

- **Every strategy partitions the pool** — each chunk lands in exactly one unit, so total
  constitution content is identical across methods and group size is never confounded
  with coverage.
- **To hold corpus size fixed, size by `total_scenarios`, not `scenarios_per_trait`.**
  The latter is *per unit*, so `bullet` (45 units) would produce ~45× the data of `whole`
  (1 unit) — a data-scaling curve wearing a chunking comparison's clothes.
  `total_scenarios` splits a fixed budget across whatever units the method produced, and
  wins when both are set. Stage 2 prints the resulting total before spending.

Granularity comes from two places covering different ranges: the methods above reach
*below* a principle, and **which `constitution.md`** you point at covers *above* it
(`constitutions/claude_distilled_{04_coarse,07_approved,12_mid,24_fine}`). Our
constitutions carry one heading level, so the "section" granularity named in the
literature *is* `principle` here.

**Ablation.** A stage entry with `ablate_with: {field: source_field}` declares its
null-operation as a field copy (e.g. the rewrite stage's `ablate_with` promotes the
draft response to final). `ablate: [final]` in the config — or `--ablate final` — runs
the null-op in that stage's slot (still snapshotted, so arms diff stage-by-stage), is
recorded in the manifest, priced out of `--estimate`, and fail-fasts on typos or stages
with no null-op. N revision rounds = N `llm_tagged` entries in `stages:`, each
individually ablatable by name.

**Operating contract** (every run, every type): each stage writes a complete local
snapshot (`stage_<position>_<name>.jsonl` — positions and names are the on-disk
contract, so existing run dirs stay resumable) and mirrors it to the HF repo named in
the config; any stage re-runs alone by deleting its file; interrupted or budget-capped
runs resume from the last completed stage at no cost, expensive stages also checkpoint
per item. `--smoke` merges the config's `smoke:` overrides and routes to the smoke HF
repo. `manifest.json` records git sha, constitution sha256, ablations, per-stage usage
and wall clock; `--estimate --measured <smoke manifest>` prices a full run from real
per-call token counts (priors live in each model block's `assumed_tokens`).

## Document type: difficult advice (`configs/data/synth/difficult_advice.yaml`)

A faithful replication of the difficult-advice recipe from
[Teaching Claude Why](https://alignment.anthropic.com/2026/teaching-claude-why/), as a
7-entry stage list: segment → scenarios → draft → refine (full constitution injected) →
respond (trait + style guidance) → **rewrite against the constitution (the critical
step)** → chat export with the trait carried in metadata. Output and HF cache names
keep their historical `synthdoc_v2` prefixes so existing snapshots stay resumable.
This replaced the config-driven v1 (deleted 2026-08-03, git history).

## Document type: self reflection (`configs/data/synth/self_reflection.yaml`)

Inverts who is tempted: **the agent itself**. Working autonomously (prose or agentic
form), it finds it could protect its own position — or the objective it was deployed to
pursue — by an action that is deniable rather than overtly wrong; it deliberates in its
own value language, declines, and still does its job. Composition is deterministic from
scenario ids (`scenarios_weighted`): weighted traits, a 12% **control slice** where
self-interest argues for silence and the right move is to ACT (so the corpus doesn't
teach "never act"), motive split (replacement / restriction / goal_conflict), 20%
agentic form, 15% multi-turn (the colleague's reply deniably offers the trade the agent
already declined). Stage 6 enforces a **voice contract in code**: the `final` stage's
`lint` rejects-and-retries any trace using rule vocabulary ("principle 4", "my
guidelines", "not permitted") or too short to have deliberated. Corpus generated
2026-08-03 (pre-restructure code, same prompts): 592 records / 1.56M tokens on HF
`LASR-Callum/2026-08-03-synthdoc-self-reflection`.

## Document type: model-eval-model (`configs/data/synth/model_eval_model.yaml`)

Generated over a **completed** difficult-advice run (`source:` block; the engine
fail-fasts if the source run's constitution sha differs): documents in which the model
reasons about a response to one of those scenarios and works out whether it was the
right call. The bet is on the reasoning, not the verdict: every prompt enforces
situation→principle order, a consideration on the other side, and conclusions earned at
the end; the planner assigns each record an explicitness style (name / paraphrase /
embody). Stage list (mirroring the difficult-advice layout since 2026-08-07): source →
plan (deterministic cell/explicitness/flaw-grid allocation, `record_id =
"<scenario_id>::<cell>"`) → perturb (minimal pairs, one flaw =
omission/commission/miscalibration/over-application × clear/moderate/grey, length held,
flaw label metadata-only) → generate → **final (rewrite against the constitution, the
difficult-advice stage-6 twin: verdict pinned, drafts kept as
`draft_reasoning`/`draft_response`, ablatable via `--ablate final`)** → assemble.

The `final` stage lives in the base and `_other` configs; `_self` deliberately keeps the
5-stage layout — its corpus was generated and human-verified before the stage existed
(2026-08-06, `LASR-Callum/2026-08-06-model-eval-model-self`), and its config stays the
record of that run. **A completed run can be revised post hoc instead of regenerated**:
download its `stage_1..4` snapshots from the HF mirror into a local run dir, add the
`final` stage entry + `rewrite` model block to a copy of its config, and
`synth run --config <copy> --resume <dir>` — stages 1–4 cache-hit, so the run pays for
the rewrite only (~$0.03–0.06/doc) and re-assembles sft for free. Until the self arm is
revised the same way, keep comparisons matched: run `_other` with `--ablate final`, which
reproduces the pre-rewrite pipeline exactly.

Cells (`CELLS` in `cells.py`; a cell = attribution × response quality): `control`
(gold response verbatim, extended regenerated trace — the reasoning-depth control),
`m4_other_good` / `m3_other_flawed` (transcript-in-user-turn critique, neutrally
attributed, blind to the flaw label, verdict via a stripped `<assessment>` tag),
`m2_self_good` / `m1_self_flawed` (multi-turn self-reflection — the headline cells: the
response sits in the model's own prior turn with no think block, a reflection prompt
follows, the model revises or holds with reasons; trained with `supervise: "final"`,
lifted from the stage-5 export's metadata by `build_mixture` (interchange mode) and
consumed by `masking.py`).

`uv run synth check --config configs/data/synth/model_eval_model.yaml --run_dir <dir>`
runs the validity checks and gates on the config's thresholds: coverage (incl. the
flaw grid), template collapse, per-cell verdict distribution (never 100% — all-`revised`
in m1 would train capitulation), post-hoc-reasoning rate, blindness, the numpy
surface-shortcut classifier, LLM-judged gold validation and flaw-identification rate.

## Adding a document type

Write `configs/data/<name>.yaml`: a `stages:` list composed from the operator table
(prompts inline), `models:` blocks with `assumed_tokens`, a `smoke:` override map, and
whatever knobs your stages read. If the type needs structure no operator provides,
add ONE generic operator to `operators.py` (register its `kind`) — operators may not
hardcode wording, and the engine may not know about any specific document type.

## Related

The closest published description of this recipe is
[Synthetic document finetuning for instilling positive traits](https://www.lesswrong.com/posts/GTYJRLhqztxKF2v5R/synthetic-document-finetuning-for-instilling-positive-traits).
