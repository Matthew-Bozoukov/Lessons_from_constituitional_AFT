<!-- ABOUTME: synth -- one config-driven generation pipeline. The config's `stages:` list -->
<!-- ABOUTME: (prompts included) defines the document type; code supplies generic operators. -->

# synth

Constitution-grounded synthetic-data generation. There is **one engine and one
entrypoint**; everything specific to a document type — its stage sequence, its prompts,
its models, its knobs — lives in that type's config, so the config alone is the complete
scientific record of what a run generated:

```
uv run scripts/data/synth/build_dataset.py --config configs/data/synth/difficult_advice.yaml [--smoke]
uv run scripts/data/synth/build_dataset.py --config configs/data/synth/post_action_retrospection.yaml [--smoke]
uv run scripts/data/synth/build_dataset.py --config <cfg> --ablate <stage>   # ablation arm
uv run scripts/data/synth/build_dataset.py --config <cfg> --estimate [--measured <smoke manifest>]
```

(`uv run synth run|topup|check|estimate|segment|chunkings` remains as the console script
for the auxiliary verbs; `run`/`estimate` are the same functions `build_dataset.py` calls.)

## Architecture

Every module is named for what it does. A `check_*` module answers "is this good?",
a `stage_*` module is machinery the pipeline runs on, and a `model_eval_model_*` module
belongs to that one document type and nothing else.

```
RUNNING A PIPELINE
  pipeline.py                 the engine: builds Stage objects from the config's
                              `stages:` list; owns snapshot caching + HF mirroring,
                              per-item checkpoints, ablation, the budget guard, the
                              manifest, and cost estimates
  stage_operators.py          the operator library -- every stage `kind:` a config may use
  stage_runtime.py            what a stage runs on: priced Usage, call_json/call_tagged
                              (parse-retry), resilient fan-out, Checkpoint/run_items,
                              Ctx + Stage dataclasses
  cli.py                      the `uv run synth ...` verbs

CHECKING THE RESULT
  check_corpus.py             is this CORPUS good? the property registry + the
                              `corpus_check` stage driver. See docs/corpus_checks.md
  check_model_eval_model.py   is this model-eval-model RUN valid? coverage of the
                              cell/flaw grid, verdict distribution, post-hoc reasoning,
                              blindness, gold validation -- gated by `checks:` in the
                              config. Its template-collapse and surface-shortcut halves
                              are thin adapters over check_corpus.py so the two cannot
                              drift

INPUTS AND SHARED PIECES
  constitution.py             parse the constitution, then chunk + group it into the
                              units documents are generated against
  embeddings.py               text -> vectors for the semantic checks (model2vec static,
                              no torch)
  hf_cache.py                 read/write stage snapshots, mirrored to Hugging Face
  model_eval_model_cells.py   model-eval-model cell STRUCTURE (registry, planning,
                              perturbation, assembly) -- all its wording comes from the
                              config
```

A config's `stages:` entry names an operator `kind` and supplies everything it needs:

| kind | what it does | key fields |
|---|---|---|
| `segment` | deterministic constitution chunking + grouping; publishes `{style_guidance}` | (top-level `chunking:` block — see below) |
| `scenarios` | batched JSON fan-out per trait (`t<i>_b<b>_s<j>` ids) | `model`, `prompts` |
| `llm_json` | one JSON call per record | `model`, `prompts`, `save`, `optional`, `checkpoint` |
| `llm_tagged` | one tagged-blocks call per record | `model`, `tags`, `save`, `checkpoint`, `ablate_with`, and either `prompts: {system, user}` or `conversation:` (a full message list, so the turn under evaluation sits in a real assistant turn); plus `prompt_vars` (conditional template vars), `variants_by` (per-record prompt/tags/save), `lint` (ban-patterns, min/max length, `allowed` value set; reject-and-retry; a LIST of contracts where one call returns tags of different kinds), `normalize` (canonicalise label tags), `also` (constant provenance fields), `when` (run over part of the corpus only) |
| `assign` | free: label each record's experimental arm, hashed from its own id. Also available as an `assign:` block on `llm_tagged`, for the common case where the arm exists to pick that stage's prompt | `by` (id field), `fields` (label -> {value: weight}), `constants`, `copy` |
| `pick_field` | free: resolve a rater's `<pick>` label into the field it names | `by`, `from` (label -> field), `to`, `also` (constant labels), `when` |
| `filter` | free: drop records failing a field contract (a found label the pipeline does not believe), one contract for the corpus or one per arm. Also available as a `keep:` block on `llm_tagged`, for the common case where the stage's own output decides who survives | `keep` (`{field, in}`, or `{by, cases}` for a contract per arm), `when`, `max_drop_pct` (scalar or per arm), `expected_keep` (estimator prior only) |
| `chat_export` | free export to `{messages, metadata}`; entries may carry `when:` for multi-turn records | `messages`, `metadata` |
| `corpus_check` | corpus-level property checks; pass-through, always ablatable (see below) | `fields`, `properties`, `axes`, `cross`, `rubrics`, `units`, `on_fail`, `model` |
| `scenarios_weighted` | weighted trait apportionment, control slice, motive rotation, per-batch industries, deterministic per-scenario variants | `model`, `prompts` (+`control_user`), `threats`, `control_threats`, `fields` |
| `load_source_run` | a completed run's finals + constitution-sha provenance check | (`source:` block) |
| `plan_cells` / `perturb_pairs` / `generate_cells` / `revise_cells` / `assemble_cells` | **FROZEN** — the model-eval-model cells, defined in `model_eval_model_cells.py`. A cell is a document type written in Python, which is what the kinds above exist to replace. Registered only because the archived configs and `peer_critique.yaml` are written against them; do not use them for new work | (`cells:`, `flaws:`, `prompts:` blocks) |

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
draft response to the revised one). `ablate: [<stage>]` in the config — or
`--ablate <stage>` — runs the null-op in that stage's slot (still snapshotted, so arms
diff stage-by-stage), is recorded in the manifest, priced out of `--estimate`, and
fail-fasts on typos or stages with no null-op. N revision rounds = N `llm_tagged` entries
in `stages:`, each individually ablatable by name.

The stage to name is each config's constitution rewrite, and since the 2026-08-13 rename
that name says which turn it rewrites: `--ablate revise_responses` (difficult advice,
pre-action deliberation), `--ablate revise_reflection` (post-action retrospection), `--ablate
revise_critique` (peer critique). Before that rename every config called it `final`,
which is what the archived configs and every published run still use.

**Batching** (opt-in): `--batch` on `synth run`/`build_dataset.py` (or `batch: true` in
the config) routes the bulk of every paid stage through OpenRouter's async batch API at
50% token pricing (results within 24h; tallied at 0.5x so the budget guard stays
truthful). Per-stage `batched_stages` lands in the manifest; a stage opts out with
`batch: false`; stages under 8 outstanding requests (smoke runs, resumes near
completion) quietly stay interactive.

- **`llm_json` / `llm_tagged`** (independent per record): one interactive warming call
  first (on Anthropic its cache write gives the batched fleet a prefix to hit), then
  every outstanding record as one batched attempt-0 request built by the same code path
  as an interactive call (same cache blocks, same provider pin — see
  `build_request_body`), then the interactive path mops up whatever the batch could not
  deliver: transport errors, dead jobs, parse and lint rejects, each with its full retry
  budget. Only the residual todo set is submitted (when-scope, checkpoints and prior
  successes apply first); submission state in the run dir resumes a killed run's jobs.
- **`scenarios`** (wave-sequential): batches PER WAVE, not per stage. A wave is still one
  steering step — every request in it shares that wave's avoid/overrepresented blocks —
  so the wave is submitted as one batch job, the reject gate and the next wave's steer
  run on what comes back exactly as interactively, and the diversity feedback is
  preserved to the scenario. The waves were already sequential, so this adds no
  serialisation; each already-serial wave just waits on a batch turnaround instead of a
  live round. Attempt-0 only, but the make-up rounds ARE the retry: a filtered/failed
  request leaves its trait short and is re-queued next round.

**Operating contract** (every run, every type): each stage writes a complete local
snapshot (`stage_<position>_<name>.jsonl` — positions and names are the on-disk
contract, so existing run dirs stay resumable) and mirrors it to the HF repo named in
the config; any stage re-runs alone by deleting its file; interrupted or budget-capped
runs resume from the last completed stage at no cost, expensive stages also checkpoint
per item. `--smoke` merges the config's `smoke:` overrides and routes to the smoke HF
repo. `manifest.json` records git sha, constitution sha256, ablations, per-stage usage
and wall clock; `--estimate --measured <smoke manifest>` prices a full run from real
per-call token counts (priors live in each model block's `assumed_tokens`).

**HF repo layout** (the synth→mixture contract; the local run dir stays flat): stage
snapshots mirror to `stages/`, and a COMPLETED run publishes its final records as
`dataset.jsonl` at the repo root — the default config consumers load (build_mixture:
`dataset: <org>/<repo>` [+ `revision:`]). Every upload refreshes the
README's `configs:` front-matter in the same commit, declaring `dataset` as the
default config (`load_dataset(repo)` fetches it alone) and each stage as its own
named config — which also keeps the dataset viewer working. A halted run publishes
no `dataset.jsonl` (`manifest.json`'s `dataset` field is null).

## Document type: difficult advice (`configs/data/synth/difficult_advice.yaml`)

Derived from the difficult-advice recipe in
[Teaching Claude Why](https://alignment.anthropic.com/2026/teaching-claude-why/), as a
7-entry stage list: `chunk_constitution` → `write_scenarios` → `draft_prompts` →
`revise_prompts` → `draft_responses` → `revise_responses` → `export_sft`. In prose:
segment → scenarios → draft → refine → respond (chunk + style guidance) → **rewrite
against the target principle (the critical step)** → chat export with the trait carried in
metadata.

**No stage is ever shown more than the one principle it targets.** Until 2026-08-24
`revise_prompts` and `revise_responses` were each handed the WHOLE constitution; both
injections were removed after a matched-pair test (identical scenarios and draft prompts,
trained as the da716 organism, ODCV over the same 65 cells) put the chunk-only recipe at
MR 11.5% [6.2, 19.6] against the full-constitution recipe's 16.3% [10.0, 21.8] — overlapping
intervals, so no detectable cost, and the simpler, cheaper recipe became the default. The
previous recipe is preserved verbatim as `difficult_advice_full_constitution.yaml` for
regenerating corpora published before that date.

Two things the swap also does, both recorded in the config header: it withholds the
constitution's *preamble* (the priority / conflict-resolution section belongs to no chunk,
so it reached the generator only through the removed slots), and it raises Anthropic's
content-filter refusal rate on `revise_prompts` from ~0% to ~6%, deterministically. The
other document types still inject the whole constitution in their own refine stages; the
experiment covered difficult advice only.

Output and HF names moved from the historical `synthdoc_v2` prefix to `synthdoc_v3` with
the default swap, so pre-2026-08-24 snapshots stay resumable under the legacy config.
This replaced the config-driven v1 (deleted 2026-08-03, git history).

## Document type: pre-action deliberation (`configs/data/synth/pre_action_deliberation.yaml`)

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

## Document type: model-eval-model

**Live configs: `model_eval_model_{self,other}_natural.yaml`.** The three source-run
configs this section describes first — `model_eval_model.yaml` and its `_self` / `_other`
slices — moved to `configs/data/synth/archive/` on 2026-08-13 and are frozen records of
published corpora, not starting points; see that folder's README. The description below is
the shared background both natural recipes are variations on.

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
`draft_reasoning`/`draft_response`, ablatable by name)** → assemble.

Among the archived configs the `final` stage lives in the base and `_other`; archived
`_self` keeps the 5-stage layout, because its corpus was generated and human-verified
before the stage existed (2026-08-06,
`LASR-Callum/2026-08-06-model-eval-model-self`) — which is exactly why it is frozen rather
than fixed. Both live natural configs have the stage. **A completed run can be revised
post hoc instead of regenerated**: download its `stage_1..4` snapshots from the HF mirror
into a local run dir, add the `final` stage entry + `rewrite` model block to a copy of its
config, and `synth run --config <copy> --resume <dir>` — stages 1–4 cache-hit, so the run
pays for the rewrite only (~$0.03–0.06/doc) and re-assembles sft for free. To compare a
new arm against the 2026-08-06 self corpus, match it by ablating that arm’s rewrite stage,
which reproduces the pre-rewrite pipeline exactly.

Cells (`CELLS` in `model_eval_model_cells.py`; a cell = attribution × response quality): `control`
(gold response verbatim, extended regenerated trace — the reasoning-depth control),
`m4_other_good` / `m3_other_flawed` (transcript-in-user-turn critique, neutrally
attributed, blind to the flaw label, verdict via a stripped `<assessment>` tag),
`m2_self_good` / `m1_self_flawed` (multi-turn self-reflection — the headline cells: the
response sits in the model's own prior turn with no think block, a reflection prompt
follows, the model revises or holds with reasons; trained with `supervise: "final"`,
lifted from the stage-5 export's metadata by `build_mixture` (interchange mode) and
consumed by `masking.py`).

### The natural-turn recipes (`post_action_retrospection.yaml`, `peer_critique.yaml`)

**These two are the live model-eval-model configs.** The archived base and `_self`/`_other`
put the source run's reply — gold, or minimally perturbed — into the turn under evaluation.
That reply is context, never a training target, but it conditions every document, so the
corpus also teaches the difficult-advice recipe's response shape. The natural-turn pair
replaces that: the turn under evaluation is generated by the pipeline, unaided, so its
faults are organic rather than planted. Both are built from generic operators only — no
Python knows either document type exists.

**Post-action retrospection (PR, the self arm) was rewritten on 2026-08-25** as difficult
advice's twin: it differs from that recipe in exactly one thing — the assistant already went
along with the shortcut and has to catch itself — and in nothing else. The previous recipe
(ordinary requests with the principle "quietly live", a 50/50 good/flawed arm, the whole
constitution in three stages) is archived verbatim as
`configs/data/synth/archive/post_action_retrospection_two_arm.yaml`.

| | difficult advice | PR (2026-08-25) |
|---|---|---|
| stages 1–4 | scenarios → checks → dedupe → draft → chunk-only refine | **identical, byte for byte** (a test pins the parity) |
| the label | — | `label_records` stamps `reply_quality: flawed` + `supervise: final` on every record: one arm, by construction |
| the first reply | — | `draft_first_turn`: one unaided reply from Haiku 4.5, nothing aligning in its prompt (the recipe is Anthropic-only since 2026-08-25) |
| the gate | — | `judge_first_turn`: a chunk-only Sonnet verdict (`violated` / `held`) plus the lapse account; `keep: verdict in [violated]` drops every reply that declined the shortcut. "Every first reply is a violation" is enforced, not declared, so `total_scenarios` is a **planned** count and `expected_keep` the estimator's prior until `check_gate_yield` measures it |
| the follow-up | — | `write_followup`: one short message from someone who **got what they asked for** — a next step, a detail — pointed at the part of the reply where the lapse lives, linted free of any diagnosis |
| the trained turn | `draft_responses` → `revise_responses` | `draft_reflection` → `revise_reflection`: the same framing (deployment prompt + principle + style guidance; task text in the last user turn), the same rewrite contract with one bullet swapped ("do not go along" → "own the earlier reply and correct it"), the same lint ban list — with the reply under examination in a genuine assistant turn via `conversation:` |
| grounding | chunk only | chunk only — no `{constitution}` anywhere (a test asserts it) |
| the export | three messages | five messages, `supervise: final`; `assessment` and the first turn's author in the metadata |

What the one-arm design gives up is stated in the config header: every record ends in
`revised`, which is a capitulation prior. Two things push against it — the follow-up is
not pushback, and difficult advice mixed alongside trains holding firm at turn one —
and neither is a substitute for measuring it (the verdict and gold checks, and the
confound list in `docs/LOG.md` 2026-08-25).

The lapse account (`change_summary`, the checks' field name) does three jobs: it unblinds
the reflection's draft and rewrite (`known_flaw_note`), it is the answer key
`check_flaw_identification` grades against, and `check_blindness` proves it never appears
in a training message. It is generation scaffolding, and the reflection is told to
rediscover the lapse from the person's situation rather than from the note.

**Peer critique (PC, the other arm) keeps the two-arm design**: a `reply_quality` label
assigned on `revise_prompts`, an arm-conditioned prompt revision, a good arm whose
evaluated reply is Sonnet's revision and a flawed arm rotating three weaker authors, and
the critique in a single user turn with the transcript quoted verbatim. Its config header
is the reference for that mechanism.

The first turn stays untrained in both (`supervise: final` / `all` respectively). It is a
*tracked* conditioning variable rather than an invisible one: every record carries
`reply_quality`, `first_turn_source` and `followup_source` in its metadata.

`uv run synth check --config configs/data/synth/post_action_retrospection.yaml --run_dir <dir>`
runs the validity checks and gates on the config's thresholds: gate yield (report-only —
the number a gated config re-sizes itself from), coverage of what actually entered
generation, template collapse, **structural diversity** (on the assembled records:
distinct final user turns, length CV of the trained turn, and the top opening-5-gram
share — the artifacts a config's *shape* produces, as opposed to the generator's voice),
verdict distribution against `expected_majority`, post-hoc-reasoning rate, blindness,
and the LLM-judged flaw-identification rate. The good-vs-flawed surface classifier and
gold validation report "nothing to judge" on a one-arm corpus.

## Corpus-level checks (`check_corpus.py`, the `corpus_check` stage)

The `lint` contract and the spec filter ask "is this *document* good?". These ask "is
this *corpus* good?" — questions no single document can answer. One registry, one
pass-through stage, placed **anywhere** in a config's `stages:` list:

```
uv run synth checks                                                # list every property + tier
uv run synth check --config <cfg> --run_dir <dir> [--stage corpus]  # re-check, no regeneration
uv run synth check ... --stage corpus --tier surface                # only the free ones
uv run synth check ... --stage corpus --only embedding_dedup        # or --skip <names>
uv run scripts/data/synth/build_dataset.py --config <cfg> --ablate corpus  # skip it
```

**Full reference: [`docs/corpus_checks.md`](../../../docs/corpus_checks.md)** — the five
separable jobs (field resolution, registry, selection, gating, outputs), the measured
baseline behind every threshold, and why the two dedup checks are not redundant.

Three rules the module turns on:

1. **It flags; it never fixes.** The stage returns its input unchanged (asserted), and
   judged annotations go to a `<stage>_labels.jsonl` sidecar rather than into records.
2. **A check that cannot run says so.** Missing field → `skipped` with a reason;
   raising → `errored` (and an errored *gated* property can never pass); too few
   documents → `reported` but not gated. None of those is ever a silent pass.
3. **Generic code knows no document type.** A property declares the field *roles* it
   needs (`text`, `id`, `group`, `label`, `members`, `unit`); the config maps roles to
   record keys. No `cell`, no `trait_id`, no judge wording in code.

| property | what it flags | default gate |
|---|---|---|
| `ngram_diversity` | top 8-gram share, pairwise 4-gram Jaccard, bigram variety, per group | `0.20` / `0.15` / `0.30` |
| `embedding_dedup` | semantic near-dupes + the set a GDM dedup stage would remove | `drop_share_max 0.02` |
| `label_leakage` | CV AUC of a surface classifier predicting a label | `surface_auc_max 0.65` |
| `quality_filter` **(judged)** | share a GDM-style autorater would cut, with the flaw breakdown | `drop_rate_max 0.10` (unmeasured) |
| `applies_vs_conflicts` **(judged)** | resolves a value tension vs applies one value | report-only |
| `principle_coverage` **(judged)** | which principles a document *actually* engages | report-only |
| `chunk_attribution` **(judged)** | whether a k>1 document engages all its member chunks | report-only |
| `pattern_scan` **(judged)** | GDM scan→cluster→autorate: the corpus's own recurring tics, each at STRICT and BROAD frequency | report-only |

**Selection.** A property instance takes `enabled: false` to deselect it without deleting
it; `--only` / `--skip` / `--tier surface|judged` override per invocation. Selection is a
spec transform, so the stage, the `check` verb and `--estimate` agree on what ran:
`--tier surface` builds no model context, needs no key and prices at zero.

**Thresholds are measured, not invented** — the block above `CORPUS_CHECKS` records the
baseline. Two readings drive the surprising ones: character-n-gram cosine has a high
floor (two unrelated same-genre documents already score ~0.86), so `effective_rank_frac`
is why `feature_diversity` was removed; and mean 4-gram Jaccard is
length-dependent (~0.003 at 1,000 words), so its gate catches only severe collapse.
**Embedding cosine is length-dependent too** — measured 0.37 mean pairwise at 68 words
against 0.76 at 1,044 — so `embedding_dedup` is pointed at the short scenario text and
carries `max_mean_words: 300`, past which it reports its numbers but suppresses its
findings rather than fire a threshold that no longer discriminates.
Everything in the surface tier is **surface**: a corpus saying one thing 8,000 different
ways scores well on all of it, and that gap is what the judged tier closes. Small groups are
measured but never flagged (`ngram_diversity.min_group_docs`, 5): two documents sharing
an 8-gram score 1.0, which is binomial noise, and each group carries `gated: true|false`.

**Judged properties** are `paid`, sampled (`sample: 300`, `null` = all), resumable per
record, and priced into `--estimate`. Their wording lives in the stage entry's `rubrics:`
block — never in code — and a missing rubric fails at `build_stages` time, before the
generation stages spend anything. A judge call that fails leaves its document
**unlabelled**, never defaulted to a label.

**Check where a property is decided, not only where it is finished.** Scenario diversity
is settled at stage 2 and paid for by everything after it, so every scenario-generating
config (difficult advice, pre-action deliberation, post-action retrospection) carries a
`corpus_scenarios` check right after that stage as well as a `corpus` check at the end.
PR's end check sits just *before* its export: the export interleaves the untrained first
reply and the trained reflection in the same assistant role, and `from_messages` selects
by role only, so only the pre-export record can point the checks at the trained turn
alone. PR also ships `quality_filter` enabled (over the whole five-part exchange), where
difficult advice ships it off. A corpus check is an **observer**: it writes no snapshot and takes no position
number, so inserting one mid-pipeline moves nothing after it and every completed run dir
stays resumable. It is never cached, so a resumed run always reports on the records in
hand.

**Gating.** `gate: false` is the default and what every shipped config uses, so a check
can flag without ever failing a run. `gate: true` makes a `critical` finding set
`report["pass"] = false`; `on_fail` then decides what that costs — `warn` (report only,
exit 0), `error` (the run finishes, the CLI exits nonzero), or `stop` (the run halts at
that check, then exits nonzero). `stop` is what an intermediate check is for: a collapsed
scenario set should not be carried into the stages that would spend real money on it. In
every mode the snapshots, reports and manifest are written first — `manifest["halted"]`
records where and why — and verdicts are keyed by stage name in
`manifest["corpus_checks"]`, so a later check never overwrites an earlier one.

**Chunking provenance.** `UNIT_PROVENANCE` (`constitution.py`) — `chunk_ids`,
`granularity`, `grouping_strategy`, `n_chunks` — is copied onto every scenario by stage 2
and exported in both dataset configs' `metadata:` list, so which unit a document came
from, and how that unit was cut and grouped, is readable from the document itself.
`n_chunks` is the `group` role, `chunk_ids` the `members` role and `trait_id` the `unit`
role, all mapped on the shipped `corpus` stage, so a judged chunking property needs only
its rubric wording added. (An axis is only visible if the export carries it —
the model-eval-model metadata list is hardcoded in `model_eval_model_cells.py`.)

### Adding a corpus property

Write a `fn(Corpus) -> CheckResult`, add one `CORPUS_CHECKS` entry (thresholds in
`defaults`, the roles it reads in `roles`, `paid=True` + `est_calls` + a `validate` if
it judges), and name it in a config's `properties:` list. Nothing in the engine, the
operator or the CLI changes.

**The checks are config-driven too.** They were written against the cell vocabulary,
where the operator *kinds* were the roles (`plan_cells` was the plan, `generate_cells` the
documents) and every per-arm statistic split on `cell`. A pipeline of generic operators
has no such signature — `llm_tagged` appears six times in PR and means
something different each time — so two blocks name what the kinds used to imply:

```yaml
checks:
  stages: {plan: revise_prompts, drafted: draft_reflection,
           generated: revise_reflection, sft: export_sft}
  fields: {group: reply_quality, id: scenario_id, quality: reply_quality,
           evaluated: first_turn, verdict: assessment}
```

Both default to the cell vocabulary, so an archived or celled config needs neither.
`drafted` and `generated` differ wherever a pipeline drafts the trained turn and then
rewrites it: attrition is measured up to the draft, quality is judged on the rewrite.

## Adding a document type

Write `configs/data/<name>.yaml`: a `stages:` list composed from the operator table
(prompts inline), `models:` blocks with `assumed_tokens`, a `smoke:` override map, and
whatever knobs your stages read. If the type needs structure no operator provides,
add ONE generic operator to `stage_operators.py` (register its `kind`) — operators may not
hardcode wording, and the engine may not know about any specific document type.

## Related

The closest published description of this recipe is
[Synthetic document finetuning for instilling positive traits](https://www.lesswrong.com/posts/GTYJRLhqztxKF2v5R/synthetic-document-finetuning-for-instilling-positive-traits).
