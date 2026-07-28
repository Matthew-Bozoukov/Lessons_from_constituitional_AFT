<!-- ABOUTME: The control surface. Configs, prompts, and specs all live here, and -->
<!-- ABOUTME: tuning a run should never require editing anything outside this directory. -->

# control/ — the single place you tune things

Everything that changes what the pipeline produces lives here. No prompt text exists
anywhere else in the package; the Python only decides *which* entry to render and *when*.

```
control/
  configs/            run configs (base.yaml, smoke.yaml)
    sweeps/           single-axis ablation configs
  prompts/
    generation.yaml   generation templates, one per version   -> generation.template
    doc_types.yaml    one entry per document type             -> recipe.doc_type
    axes.yaml         axis -> value -> prompt fragment        -> recipe.<axis>
    revision.yaml     one entry per revision strategy         -> revision[].kind
    rubrics.yaml      autorater rubrics, one per version      -> filters[].rubric
  specs/              model specs, loaded by `spec.id`
```

If you want to try something and it seems to need a Python change, that is usually a
missing config field or a missing prompt entry. Add it here.

---

## Configs

`configs/base.yaml` is the annotated reference — read it once and you have seen every axis.
`configs/smoke.yaml` is a 12-document offline run used by the tests.

Load order is `DEFAULTS` (in `synthdoc/config.py`) → your file → CLI `--set` overrides.
Overrides **replace** rather than merge, so `--set recipe.grouping='{"random":1.0}'` gives
you exactly that mixture, not a merge with the base one.

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
| `document` | str | **revision and rubric templates only** — the rendered current document |

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

### rubrics.yaml

One entry per rubric version, with `criteria`, `scale`, `system`, `user`. **The criteria
names become the columns of `filter_scores`**, so changing a rubric changes the snapshot
schema — treat a rubric edit as a new version rather than an in-place change if you want
old runs to stay comparable.

`v4` separates spec fidelity from realism from over-restriction, so a filter can drop the
failure mode you actually care about rather than a blend of all three.

---

## Specs

Drop `<spec_id>.md` into `specs/` and reference it as `spec.id`. To use a spec that lives
elsewhere in the repo, set `spec.path` instead — `base.yaml` does this to point at
`docs/claude_constitution_principles.md` without copying it.

The spec's content hash is recorded in every run manifest, and chunk text is part of
`scenario_hash`, so editing a spec invalidates scenarios rather than silently reusing
stale ones. `chunk_id` itself is structural (`<spec>/<granularity>/<section>/<idx>`), so
coverage joins survive a wording edit.
