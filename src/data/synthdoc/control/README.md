<!-- ABOUTME: The control surface. Configs, prompts, and specs all live here, and -->
<!-- ABOUTME: tuning a run should never require editing anything outside this directory. -->

# control/ — the single place you tune things

Everything that changes what the pipeline produces lives here. No prompt text exists
anywhere else in the package; the Python only decides *which* entry to render and *when*.

```
control/
  configs/            run configs (base.yaml, smoke.yaml)
    corpora/          named corpus variants, each `extends: base.yaml`
    sweeps/           single-axis ablation configs
  prompts/
    planning.yaml     scenario planners, one per template     -> planning.template
    generation.yaml   generation templates, one per version   -> generation.template
    strategies.yaml   multi-call generation strategies        -> generation.strategy
    doc_types.yaml    one entry per document type             -> recipe.doc_type
    axes.yaml         axis -> value -> prompt fragment        -> recipe.<axis>
    revision.yaml     one entry per revision strategy         -> revision[].kind
    patterns.yaml     scan/match prompts + seed anti-patterns -> filters[].kind: pattern_scan
    rubrics.yaml      autorater rubrics, one per version      -> filters[].rubric
  specs/              model specs + index.yaml, loaded by `spec.id`
```

If you want to try something and it seems to need a Python change, that is usually a
missing config field or a missing prompt entry. Add it here.

---

## Configs

`configs/base.yaml` is the annotated reference — read it once and you have seen every axis.
`configs/smoke.yaml` is a 12-document offline run used by the tests.

Load order is `DEFAULTS` (in `synthdoc/config.py`) → the `extends:` chain → your file →
CLI `--set` overrides. Overrides **replace** rather than merge, so
`--set recipe.grouping='{"random":1.0}'` gives you exactly that mixture, not a merge with
the base one.

### Caching: how much, and where

```yaml
cache:
  enabled: true
  dir: output/synthdoc_cache        # WHERE
  namespace: ""                     # key prefix — a clean run without deleting anything
  scope: [plan, generate, revise, filter]   # WHICH call sites are cached
  max_bytes: 0                      # HOW MUCH — 0 = unlimited, else evict oldest first
  embeddings: true
  embeddings_dir: null              # defaults to <dir>/embeddings
```

`scope` is a cost lever, not an experiment — it never changes what the pipeline produces.
Drop `generate` to re-sample documents while replaying expensive ratings; drop `filter` to
re-rate an unchanged corpus. The manifest reports hits, misses, `bypassed` (a
scope-excluded site), evictions, and live size.

The older flat `cache_dir` / `cache_enabled` keys still work and fold into this block, with
an explicit `cache.dir` taking precedence.

### `extends:` and named corpora

A corpus variant should be the lines that differ, not a copy that drifts:

```yaml
extends: base.yaml
name: all_multiturn
recipe:
  doc_type: {multiturn_adversarial: 1.0}
```

- `name:` makes the run id stable, so the corpus lands in a predictable directory, is
  catalogued under that name, and re-running resumes it instead of making a second copy.
- **Inside `recipe:`, mixtures replace.** Merging would leave the parent's other document
  types at their old weights, and an "all multiturn" corpus would silently not be one.
  `grouping_params` is the exception and merges per strategy.
- Outside `recipe:`, blocks deep-merge — `generation: {temperature: 0.4}` keeps the
  parent's model.
- Paths resolve relative to the extending file first; cycles are rejected.

See `configs/corpora/` for the shipped variants.

Validation runs before anything is generated and rejects: unknown plugin names, doc types
and axes not declared in the prompt packs, malformed mixtures, a bad revision `context`,
and `snapshots.backend: huggingface` with no org. You should never discover a typo
mid-run.

```bash
uv run python -m synthdoc.cli validate --config base.yaml
uv run python -m synthdoc.cli configs      # list what is here
uv run python -m synthdoc.cli chunks --config base.yaml --limit 20
uv run python -m synthdoc.cli scenarios --config base.yaml --n 20
```

### Recipe: reserved keys vs axes

Inside `recipe:`, these five keys have dedicated meaning:

`n`, `chunks_per_example`, `grouping`, `grouping_params`, `doc_type`

**Every other key is read as an axis mixture** and must be declared in `axes.yaml`. That is
how a new axis becomes a config edit rather than a code edit. All axes are sampled for every
scenario, so the axis key set is fixed for a run and the snapshot schema stays stable.

---

## Prompt packs

All packs are Jinja2. `StrictUndefined` is on: a template that references a variable nobody
passed fails at render time rather than silently emitting an empty section.

### Variables available in every template

| Variable | Type | Notes |
|---|---|---|
| `chunks` | list | each has `chunk_id`, `text`, `parent_id`, `granularity`, `meta` |
| `n_chunks` | int | `1` for single-chunk scenarios |
| `grouping_strategy` | str | `single` / `random` / `adjacent` / `semantic` |
| `doc_type` | str | the sampled document type |
| `doc_type_instructions` | str | rendered from `doc_types.yaml` |
| `axis_fragments` | list | each has `axis` (label), `value`, `text` |
| `axes` | dict | raw axis name → value |
| `spec_id`, `seed` | str, int | |
| `plan` | dict | the scenario plan; empty `{}` when planning is off, so it is always safe to reference |
| `document` | str | **revision, strategy, and rubric templates only** — the rendered current document |
| `draft`, `user_prompt` | str | **draft_then_align templates only** |
| `candidates` | list | **best_of_n selector only** — rendered candidate documents |
| `patterns`, `documents`, `mode` | — | **patterns.yaml only** |

### planning.yaml

Entries are scenario planners, selected with `planning.template`. The planner decides
*which* situation is worth writing before any document exists — GDM's structured
what/how/why step. `fields` declares which keys the plan must contain; they become the
`plan` object every generation template can reference.

`planning.enabled: false` is the control arm: generate straight from the spec chunk.

### generation.yaml

One entry per template version, each with `system` and `user`. Select with
`generation.template`. `v1` is a deliberately thin control arm for template ablations;
`v2` is the default and carries the quality bar, the anti-preachiness instruction, and the
demonstrate-don't-quote constraint.

Output contract is a JSON object `{"turns": [...]}`. Turn roles must be one of
`system | user | assistant | tool`; `thinking` and `tool_calls` are optional.

### doc_types.yaml

One entry per document type: `description` (for humans) and `instructions` (spliced into
the prompt). **Adding a document type is an entry here plus a weight in
`recipe.doc_type` — no Python.** The generic `PromptedGenerator` serves any declared type.

Note `trait_inapplicable` in particular: it exists to train *against* over-application,
which is the dominant failure mode of spec finetuning. Its instructions forbid the
assistant from adding a caveat, on purpose.

### axes.yaml

`axis → label + values → fragment text`. Adding a value is an entry plus a weight in the
recipe. This is the mechanism that keeps axes orthogonal — a generator never branches on an
axis value, it splices in the fragment the loader hands back.

Declared axes: `tools`, `reasoning`, `explicitness`, `stakes_holder`, `turns`,
`reasoning_location`.

### revision.yaml

One entry per revision strategy. The **length of the `revision:` list is the dose**, and
each entry names a `kind` here.

`context: fresh` shows the reviser only the document and the spec excerpt. `context: same`
also prepends the original generation instructions. That difference is itself an ablation
axis, which is why it is a config field.

`realism_pass` deliberately does *not* touch the assistant's judgement, so its effect is
separable from `critique_rewrite`. `spec_grounding_pass` only looks for over-restriction.

### strategies.yaml

Prompts for the multi-call generation strategies, selected with `generation.strategy`.
`single_pass` needs no prompts and is the control arm.

- `draft_then_align` — answer with **no spec in context** (so the draft carries the
  model's natural voice), then align that draft to the excerpt in a fresh context.
  Requires `planning.enabled`, since the draft needs a user turn to answer.
- `best_of_n` — sample `strategy_params.n` documents and select. The selector prompt picks
  on spec fidelity, deliberately **not** polish.

### patterns.yaml

Drives the `pattern_scan` filter — GDM's scan → cluster → autorate pass. `scan` looks at a
batch of documents at once and names what recurs *across* them; `match` autorates one
document against the surviving list. `seed_patterns` are the anti-patterns GDM named
explicitly (conversion arc, propaganda, emotional buffering, BLUF, …) and are checked even
with `discover: false`.

The discovered pattern list is written to the run manifest, and is often more useful than
the filtering it drives.

### rubrics.yaml

One entry per rubric version, with `criteria`, `scale`, `system`, `user`. **The criteria
names become the columns of `filter_scores`**, so changing a rubric changes the snapshot
schema — treat a rubric edit as a new version rather than an in-place change if you want
old runs to stay comparable.

`v4` separates spec fidelity from realism from over-restriction, so a filter can drop the
failure mode you actually care about rather than a blend of all three.

---

## Specs

Drop `<spec_id>.md` into `specs/` and reference it as `spec.id`. For a spec that lives
elsewhere in the repo, add it to `specs/index.yaml`:

```yaml
specs:
  claude_constitution_principles: docs/claude_constitution_principles.md
```

Prefer the index over setting `spec.path` in a config. Resolution by id alone is what
lets `axis: spec.id` sweeps work — with `spec.path` pinned in the base config, changing
only the id would keep loading the wrong file.

The spec's content hash is recorded in every run manifest, and chunk text is part of
`scenario_hash`, so editing a spec invalidates scenarios rather than silently reusing
stale ones. `chunk_id` itself is structural (`<spec>/<granularity>/<section>/<idx>`), so
coverage joins survive a wording edit.

---

## Prompt provenance

Neither the GDM post nor Teaching Claude Why publishes literal prompt text — both describe
their instructions in prose only. Every prompt-pack entry derived from them therefore
carries a `source:` field quoting the describing sentence verbatim, so our wording can be
checked against theirs and adjusted if you read them differently.

```bash
uv run python -c "from synthdoc.control import loader; \
  print(loader.entry('planning','what_how_why')['source'])"
```

Entries without a `source:` are ours. Where a variant deliberately departs from the source
method — `draft_context: no_spec`, for instance — the entry says so explicitly rather than
implying provenance it does not have.
