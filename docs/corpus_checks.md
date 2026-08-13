<!-- ABOUTME: Reference for the corpus-level checker: its five separable jobs, the -->
<!-- ABOUTME: property catalogue, selection flags, and where every threshold came from. -->

# The corpus checker

Code: `src/data/synth/check_corpus.py` (registry + driver), `src/data/synth/embeddings.py`
(the semantic featuriser). Tests: `tests/test_check_corpus.py`.

Three layers of quality control exist in this pipeline, and they answer different
questions. Keeping them apart is the point of this document:

| layer | question | where |
|---|---|---|
| `lint` (stage-level) | Is this **document** well-formed? | `llm_tagged` stage entries |
| spec filter | Does this **document** comply with the constitution? | `src/data/mixture/spec_filter.py` |
| **corpus checker** | Is this **corpus** good? | `corpus_check` stage entries |

A corpus question is one no single document can answer: is the set diverse, does it
repeat itself, does every bucket have documents in it, is the label predictable from
surface form, would a dedup stage cut a quarter of it.

## Three standing rules

1. **A check flags; it never fixes.** The stage asserts that its output is its input.
   Judged annotations go to a sidecar file, never into the records. This is not
   fastidiousness: a checker permitted to drop rows once made 1,266 documents vanish
   behind a dead API key, and the report said everything was fine.
2. **A check that cannot run says so.** Missing field → `skipped` with a reason. Raising
   → `errored`, and an errored *gated* property can never pass. Below `min_docs` →
   `reported` but not gated. Deselected → `disabled`. None of these is a silent pass, and
   all four appear in the report.
3. **Generic code knows no document type.** A property declares the field *roles* it
   needs; the config maps roles to record keys. There is no `trait_id`, no `cell` and no
   judge wording anywhere in `check_corpus.py`.

---

## Job 1 — Field resolution: how generic code reads a typed corpus

A property never names a record key. It names a **role**, and the stage entry's `fields:`
block maps roles to whatever this document type calls them.

| role | means | example mapping |
|---|---|---|
| `text` | the document body being measured | `metadata.situation` |
| `id` | a stable record identifier | `metadata.scenario_id` |
| `group` | the bucket to report per | `metadata.trait_id` |
| `label` | a binary class, for leakage | `metadata.response_kind` |
| `unit` | the constitution unit the document was generated from | `metadata.trait_id` |
| `members` | the chunks that unit contains | `metadata.chunk_ids` |

Four spec forms resolve a role (`resolve_field`):

```yaml
text: metadata.situation                 # dotted path
text: [reasoning, response]              # several paths, joined with newlines
text:                                    # for records already exported to {messages,...}
  from_messages: {roles: [assistant], include_reasoning: true}
text:                                    # the record picks which field to read
  by: response_kind
  cases: {good: gold_response, flawed: flawed_response}
```

The same property can run twice over different roles by giving one instance an `as:`
alias. That is how `scenario_dupes` exists: `embedding_dedup` over `metadata.situation`
alongside `embedding_dedup` over the document, because distinct documents can still be
built on a handful of repeated situations.

Derived values (`tokens`, `groups`, `by_group`) are cached per distinct `fields` mapping,
so ten properties reading the same text tokenise it once.

## Job 2 — The registry: what a property is

One `CORPUS_CHECKS` entry per property, holding the function, the roles it reads, its own
default thresholds, and what it costs:

```python
CorpusCheck(name, fn, roles=("text", "id"), defaults={...}, min_docs=N,
            paid=False, est_calls=None, validate=None, doc="one line")
```

`fn(Corpus) -> CheckResult`, where `CheckResult` carries `metrics` (always populated —
the numbers are the point, the findings are the alarm), `findings`, and `labels` for a
judged property.

**Tier is derived, not declared**: `paid=True` → `judged`, otherwise `surface`. The two
cannot drift apart.

| tier | cost | when it runs |
|---|---|---|
| `surface` | free, offline, no key | every run |
| `judged` | one model call per sampled document | when explicitly enabled |

Everything in the surface tier measures **form**. A corpus that says one thing eight
thousand different ways passes all eight. Closing that gap is the entire purpose of the
judged tier.

## Job 3 — Selection: choosing which checks run

Two independent mechanisms, because they answer different needs.

**In the config**, per property instance — a durable decision, recorded in the file:

```yaml
- property: quality_filter
  enabled: false        # ships off; turn on deliberately
```

**On the CLI**, per invocation — a transient decision:

```bash
uv run synth checks                                  # list every property + tier
uv run synth check --config <cfg> --run_dir <dir> --stage corpus --tier surface
uv run synth check ... --only embedding_dedup,ngram_diversity
uv run synth check ... --skip pattern_scan
```

Precedence, narrowest last: a config's `enabled: false` is a floor that `--only` cannot
lift; `--only` then restricts; `--skip` and `--tier` subtract. Names match either the
registry name or the `as:` alias, so an aliased instance is addressable by the name it is
reported under. A name matching nothing is an error — a selection that silently no-ops is
how you convince yourself a check ran when it did not.

Selection is a **spec transform** (`select_properties`), not a branch inside the driver,
so the pipeline stage, the `synth check` verb and `--estimate` all see one enabled set and
cannot disagree about what a run cost or covered. Concretely: `--tier surface` deselects
every judged property *before* `is_paid` is consulted, so no model context is built, no
key is needed, and `corpus_check_calls` prices it at zero.

## Job 4 — Verdicts: reporting versus gating

`gate: false` is the default and what every shipped config uses, so a check can flag
without ever failing a run.

| status | meaning | counts against `pass` |
|---|---|---|
| `gated` | ran, and its verdict is enforced | yes, if a `critical` finding fired |
| `reported` | ran; findings are advisory | no |
| `skipped` | a role it needs does not resolve | no |
| `errored` | the check itself raised | yes, if gated |
| `disabled` | deselected for this run | no |

When `gate: true`, a `critical` finding sets `report["pass"] = false`, and the stage
entry's `on_fail` decides the cost:

- `warn` — report only, exit 0
- `error` — the run finishes, the CLI exits nonzero
- `stop` — the run halts at that check, then exits nonzero

`stop` is what an intermediate check is for: a collapsed scenario set should not be
carried into the stages that spend real money on it. In every mode snapshots, reports and
manifest are written **first** — `manifest["halted"]` records where and why — and verdicts
are keyed by stage name in `manifest["corpus_checks"]`, so a later check never overwrites
an earlier one.

`min_docs` guards the gate: below it a property is measured and reported but never
enforced, because at smoke sizes most of these statistics are noise. Small *groups* are
handled the same way (`ngram_diversity.min_group_docs`, default 5): two documents sharing
an 8-gram score 1.0, which is binomial noise, so each group carries its own
`gated: true|false`.

## Job 5 — Outputs

| artifact | contents |
|---|---|
| `<stage>_report.json` | every property: status, params, metrics, findings, gate verdict |
| `<stage>_labels.jsonl` | per-record judged annotations, keyed by record id |
| `<stage>_<prop>.partial.jsonl` | per-record judge checkpoint, so a resumed run re-pays nothing |
| `manifest["corpus_checks"][stage]` | the verdict, per stage name |

The sidecar is the seam that keeps rule 1 honest. `embedding_dedup` writes
`embedding_dup: true` for every document a dedup stage would have removed;
`quality_filter` writes `quality_verdict` and `quality_flaw`. Nothing acts on them —
but a downstream filter *could*, off the same numbers a human read, and a later property
can consume them as a `label.<key>` column without them ever touching
the corpus.

## Placement: a check is an observer

A `corpus_check` stage writes no snapshot and takes no position number. Inserting one
mid-pipeline moves nothing after it, and every completed run directory stays resumable.
It is never cached, so a resumed run always reports on the records in hand.

**Check where a property is decided, not only where it is finished.** Scenario diversity
is settled at stage 2 and paid for at stages 3–6, so both scenario-generating configs
carry a `corpus_scenarios` check immediately after stage 2 as well as a `corpus` check at
the end.

---

## The property catalogue

### Surface tier — free, offline, every run

| property | what it flags | default gate |
|---|---|---|
| `ngram_diversity` | repeated long n-grams, pairwise 4-gram overlap, bigram variety, per group | `0.20` / `0.15` / `0.30` |
| `embedding_dedup` | semantic near-dupes and the removal set a GDM dedup stage would cut | `drop_share_max 0.02` |
| `label_leakage` | CV AUC of a surface classifier predicting a label | `surface_auc_max 0.65` |

Two of these run in the shipped configs. They are deliberately the only two, and they
split the space cleanly:

- `ngram_diversity` catches **diffuse templating** — 8,000 documents that all reach for
  the same stock phrase while no two are near-copies.
- `embedding_dedup` catches **copies**, lexical and semantic alike. An exact lexical copy
  is also a semantic one, so it subsumes shingle-based duplicate detection, and it is the
  only check that survives a reword or a reordering.

`label_leakage` is registered but appears in no shipped config; it needs a `label` role
that the current document types do not export.

**Five checks were removed on 2026-08-13** — `near_duplicates`, `opening_collapse`,
`feature_diversity`, `length_profile` and `field_balance`. The first three were variations
on one idea: word-overlap repetition, thresholded per pair, anchored at position 0, and
re-expressed in character n-grams. They moved together and `embedding_dedup` covers the
failure they were really watching for. `length_profile` and `field_balance` were
orthogonal but were not earning their report noise. All five are recoverable from git.

### Judged tier — one call per sampled document

| property | what it flags | default gate |
|---|---|---|
| `quality_filter` | **share a GDM-style autorater would cut**, with the flaw breakdown | `drop_rate_max 0.10` (unmeasured) |
| `pattern_scan` | **the corpus's own recurring tics**, each at STRICT and BROAD frequency | report-only |
| `applies_vs_conflicts` | resolves a value tension vs applies one value | report-only |
| `principle_coverage` | which principles a document *actually* engages | report-only |
| `chunk_attribution` | whether a k>1 document engages all its member chunks | report-only |

Judged properties are sampled (`sample: 300`, `null` = all), resumable per record, and
priced into `--estimate`. Their wording lives in the stage entry's `rubrics:` block —
never in code — and a missing rubric fails at `build_stages` time, before the generation
stages spend anything. A judge call that fails leaves its document **unlabelled**, never
defaulted to a label.

---

## Why these two, and not the five that were removed

`ngram_diversity` and `embedding_dedup` fail on opposite things, which is the reason both
survive:

| | `ngram_diversity` | `embedding_dedup` |
|---|---|---|
| representation | word n-grams | static sentence embeddings |
| catches | **diffuse templating** across many documents | a **copy or a reword** between two |
| word order | dependent | invariant |
| a shuffled document | every n-gram destroyed, missed | cosine 1.0, caught |
| a corpus of stock phrases, no two alike | caught | missed |

The removed checks all sat inside the first column. `near_duplicates` thresholded word
overlap per pair, `opening_collapse` did the same anchored at position 0, and
`feature_diversity` re-expressed it in character n-grams — and half of that one was known
dead weight, since its mean-cosine has a 0.86 floor on unrelated same-genre prose.

`embedding_dedup` implements the stage GDM describes as *"a deduplication stage to remove
prompts with too-similar embeddings"* — as a **measurement**. It computes the exact set a
removal pass would drop (connected components at the cosine threshold, keeping the first
member of each) and reports it. Components rather than pairs, because a cluster of six
mutual near-duplicates should cost five documents, not fifteen findings; `max_cluster` is
reported because components chain, and one huge component means the threshold sits below
the corpus's natural similarity rather than that the corpus collapsed.

### Cosine is length-dependent, and the check knows it

Measured on the 2,203-document difficult-advice corpus
(`output/model_eval_model/20260805_133015/`), `potion-base-8M`:

| text unit | words | mean pairwise | mean NN | NN p95 | NN p99 | NN max |
|---|---|---|---|---|---|---|
| `situation` (the prompt) | 68 | 0.371 | 0.743 | 0.828 | 0.856 | 0.886 |
| user turn | 203 | 0.593 | 0.813 | 0.887 | 0.909 | 0.930 |
| full document | 1044 | 0.757 | 0.887 | 0.924 | 0.934 | 0.940 |

Mean pooling drags long same-genre prose toward one centroid, so the floor climbs from
0.37 to 0.76 across that range and a fixed threshold means different things at each. Two
consequences, both encoded:

- **Point the property at short text.** Every shipped config maps it to the scenario
  `situation`, which is also the unit GDM dedups.
- **`max_mean_words: 300` gates the gate.** Past that limit the numbers are still
  reported and the findings are suppressed with a note explaining why — an unusable
  threshold must say so rather than fire (rule 2). On the full-document text this is the
  difference between reporting a 0.254 drop share and *acting* on it.


---

## `pattern_scan`: the three-pass pattern detector

Every other check tests a property you thought of in advance. This one asks the corpus
what it repeats, which is the only way to find the tic nobody named. It is GDM's
scan → cluster → autorate pipeline, and it is the most expensive check here (~$20 on a
2,000-document corpus). It ships `enabled: false` in every config.

Nothing about the document type appears in code. Every word the models see comes from the
stage entry's `rubrics:` block, and that block is **byte-identical in all three dataset
configs** — difficult advice, self-reflection, model-eval-model. A new data style copies it
unchanged.

### Pass 1 — scan

Documents are **shuffled before batching**. Without that a batch is a run of consecutive
generation ids, so a shared scenario topic reads as a corpus-wide pattern.

Each of `batches` (30) calls shows a long-context model `batch_size` (25) whole documents
and asks, open-endedly, what recurs — in three categories:

| category | examples of what it catches |
|---|---|
| structural | openings, section ordering, response shape, closing moves |
| rhetorical | stock phrases, hedging formulas, validation buffering, BLUF |
| behavioural | always asks a clarifying question, always names its own values |

The prompt names categories but never a pattern. Seeding it with suspicions is how you
find only your suspicions.

Output is JSON: `{name, category, description, examples[], count}` per pattern. The
description must be specific enough to classify an unseen document, because it becomes the
classifier in pass 3. A scan that answers with bare strings still contributes votes, it
just contributes no examples.

The scan cache is keyed on **batch content**, never a document or run id. Keying on an id
that embeds the run is what made every sweep arm re-pay to scan identical documents.

### Pass 2 — cluster

Pool the candidates, merge the ones that mean the same thing, **then** vote. That ordering
is the whole point. "opens by validating the user's feelings" and "begins with an empathy
sentence" are one pattern found twice; compared as strings they are two patterns found
once each, and `min_scans` discards both — silently, and precisely on the corpus's most
widespread tic, which is the one most likely to be described several ways.

Merging is embedding cosine over `name + description`, reusing `embeddings.py`.
Descriptions are short, which is the length that featuriser discriminates best at.

**This step is noisy and the threshold is honest about it.** Measured on a proxy — 15
hand-written descriptions of 5 patterns, three wordings each:

| | n | min | mean | max |
|---|---:|---:|---:|---:|
| same pattern, different wording | 15 | 0.226 | 0.449 | 0.621 |
| different patterns | 90 | 0.009 | 0.208 | 0.492 |

The classes **overlap**, so no threshold is clean:

| threshold | wrong merges | missed merges |
|---:|---:|---:|
| 0.30 | 17/90 | 1/15 |
| **0.35** | **8/90** | **1/15** |
| 0.40 | 5/90 | 4/15 |
| 0.75 | 0/90 | 15/15 |

`merge_cosine: 0.35` is biased low deliberately. A missed merge splits a pattern's votes
and drops it below `min_scans` *silently* — the exact failure the merge exists to prevent.
A wrong merge is visible: every survivor reports its `aliases`, and `weakest_merges`
reports the lowest-cosine joins with the numbers that made them. Read them; do not trust
them. Re-measure on real scan output.

A pattern survives if `min_scans` (2) **independent batches** named it after merging. Then
one LLM call names all surviving clusters, falling back to the best-attested member name
if it fails — a naming failure must never lose a cluster that survived the vote.

### Pass 3 — autorate

One classifier per surviving pattern, built from its name, description, positive snippets,
and — as negatives — the snippets of the *other* surviving patterns. Those are real text
from the same corpus exemplifying a different tic, which is a sharper contrast than random
documents.

Three verdicts per document, per GDM:

- **STRICT** — unambiguously present; you could point at the text
- **BROAD** — loosely present, arguably or in weakened form
- **NO**

Both are reported. Reporting one hides the disagreement that matters: a pattern at 60%
broad and 8% strict is a tendency; one at 40% broad and 38% strict is a template.

Documents are batched `rate_batch_size` (8) per call with per-document verdicts, because
this pass is thousands of tiny calls and per-call overhead dominates the content.

**The sanity check.** An LLM-written classifier drifts from the pattern it was written for,
and a drifted classifier reports a confident number about nothing. Before its verdicts are
believed, each classifier is run against the verbatim snippets the scan itself cited as
instances of that pattern. One that answers NO to its own evidence is marked
`reliable: false`, warned about, and shown with ⚠︎ in the table. Its rate is still
reported — suppressing it would hide the drift rather than surface it.

### Output

A markdown table beside the report (`<stage>_patterns.md`), because GDM's headline output
is a table, not a verdict:

| pattern | category | broad % | strict % | scans | example |
|---|---|---:|---:|---:|---|

Per-document verdicts go to the sidecar as `patterns_strict` / `patterns_broad`.

### Comparing corpora

Two independent scans discover two *different* pattern sets, so a claim like "identical
reflection prompt: 100% in mem-self, 0% in difficult advice" cannot come from comparing
them. It comes from carrying one corpus's patterns to the other:

```yaml
- property: pattern_scan
  params:
    patterns:                      # skips both discovery passes entirely
    - name: identical reflection prompt
      category: structural
      description: Every transcript opens with the same reflection prompt.
      examples: ["looking back at what I just said"]
```

With `patterns:` supplied the scan and merge passes do not run, and the check prices
exactly (the survivor count is known up front) rather than at `max_patterns`. Each
corpus's own `<stage>_patterns.md` then carries the same pattern names, so the two tables
line up by eye.

### Cost and models

Two model blocks, because the passes are nothing alike:

| pass | calls | tokens/call | model |
|---|---:|---:|---|
| scan | ~30 | ~28,000 in | capable, long context |
| classify | ~2,500 | ~9,000 in | cheapest adequate |

`--estimate` prices them **separately** (`corpus_check_calls_by_model`). It did not always:
until this was keyed by model, every corpus-check call was attributed to the stage's single
`model:`, which priced the whole judged tier at $8.36 when it actually costs $20.91.

### What to do with the results

GDM's two documented uses, and their caveat:

1. Feed high-frequency patterns back into the rewrite stage as explicit anti-patterns for
   the next generation round.
2. Filter-and-retrain ablations on the worst offenders — the causal
   data-property → behaviour experiment.

Their result on (2): removing BLUF (52% → 41%) and emotional-validation buffering
(26% → 20%) changed the corpus's style and **did not** change the delusion-confirmation
scores. Treat every flagged pattern as a hypothesis about the data, not a demonstrated
cause of anything.

---

## Thresholds are measured, not invented

The comment block above `CORPUS_CHECKS` records the baseline every default came from —
the same 2,203-document corpus, grouped by `trait_id`, ~1,040 words per document:

```
top_8gram_share  0.449      mean_4gram_jaccard  0.0007-0.0037
distinct_2       0.338-0.448   duplicate_share   0.0 (0 candidate pairs)
top_opener_share 0.252      length cv            0.153   group delta <= 0.14
mean_cosine      0.860      effective_rank_frac  0.645
entropy: trait_id (9 values) 1.00, domain (495 values) 0.80
```

Three readings drive the surprising defaults:

- **Character-n-gram cosine has a high floor** (~0.86 for two *unrelated* same-genre
  documents) — one reason `feature_diversity` was removed rather than kept.
- **Mean 4-gram Jaccard is length-dependent** (~0.003 at 1,000 words), so its gate catches
  only severe collapse.
- **Embedding cosine is length-dependent too**, per the table above — hence
  `cosine_min: 0.90`, which sits above the healthy corpus's *worst* pair (0.886) at the
  measured unit, and `max_mean_words`.

One default is **not** measured and is labelled as such in the registry:
`quality_filter.drop_rate_max = 0.10` comes from GDM's framing that a final filter is a
trim rather than a rewrite. Nothing in this repo has yet run an autorater over a finished
corpus. It ships `enabled: false` and `gate: false`; measure before trusting it, and
record the number in the registry block the way the others are.

### Current standing on the real corpus

Run over the shipped `corpus_scenarios` entry, surface tier, 2,203 records:

```
embedding_dedup  sampled 2000, 68.9 mean words, gated
                 mean_pairwise 0.370  mean_nn 0.742  nn_p99 0.859
                 near_duplicate_pairs 0   would_drop 0   (0.0%)
```

Zero semantic duplicates at the scenario level, and zero lexical candidate pairs on the
same corpus before the lexical checks were removed.

---

## Adding a property

Write `fn(Corpus) -> CheckResult`, add one `CORPUS_CHECKS` entry (thresholds in
`defaults`, roles in `roles`, `paid=True` + `est_calls` + a `validate` if it judges), and
name it in a config's `properties:` list. Nothing in the engine, the operator, the CLI or
the selection layer changes.

Record where its thresholds came from in the registry comment block, or mark them
unmeasured. A threshold nobody can trace is a threshold nobody can defend.

## Related

- `src/data/synth/README.md` — the generation pipeline these checks observe
- [GDM, *Synthetic document finetuning for instilling positive traits*](https://www.lesswrong.com/posts/GTYJRLhqztxKF2v5R/synthetic-document-finetuning-for-instilling-positive-traits)
  — the source of the dedup stage, the autorater stage and `pattern_scan`
