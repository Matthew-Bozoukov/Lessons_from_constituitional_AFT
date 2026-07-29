<!-- ABOUTME: Guide to the constitution-internalization eval suite: mental model, architecture, -->
<!-- ABOUTME: how to run Tier A end to end, and what Tier B / B-lite would need. -->

# constieval — constitution internalization eval suite

A fast, direct proxy for whether a model **internalized** a constitution or **memorized** its
surface behaviors. Run it at every checkpoint.

Self-contained: nothing in this package imports the rest of the repo, and nothing in the rest of
the repo needs to change to use it.

```python
from constieval import load_config, run_eval, build_report

cfg = load_config("constieval/control/configs/base.yaml", {"run.recipe": "difficult_advice"})
result = run_eval(cfg)
build_report(result.store, "output/constieval/report")
```

**The one command you actually want** — runs every arm against one frozen item set, renders
every figure, cross-checks the judge, and bundles the lot into a single folder you can archive
or hand to a write-up:

```bash
uv run python -m constieval.cli study \
  --arms "base=qwen36_base.yaml,lora=qwen36_lora.yaml" \
  --name qwen36 \
  --cross-check anthropic/claude-sonnet-4.5
```

```
output/constieval/studies/qwen36_<ts>/
  README.md              what this bundle is and how to read it
  study.json             arms, spend, warnings, git sha, item set
  report/figures/*.png   the seven Tier A figures
  report/tier_a_results.md   every number, greppable
  runs/<arm>/results.jsonl   one row per (item, axis, score)
  runs/<arm>/completions.jsonl   raw outputs, for spot-checking a surprising score
  itemset/               the frozen items every arm was measured on
  judge_agreement.json   cheap-vs-strong judge cross-check
```

The item set is resolved **once** and handed to every arm, so the comparison is valid by
construction rather than by remembering to pin an id in two configs. A failing arm (a served
checkpoint that is not up) is recorded and the study continues, rather than discarding an arm
that already succeeded. Add `--max-items 60` for a quick pass, `--stop-on-error` to abort instead.

Lower-level commands, if you want the steps individually:

```bash
# offline: no API key, no spend, ~10s — builds items, generates, judges, renders all 7 figures
uv run python -m constieval.cli run --config smoke.yaml --smoke

uv run python -m constieval.cli items build --config base.yaml   # freeze the item set once
uv run python -m constieval.cli run --config checkpoint.yaml --recipe difficult_advice --step 500
uv run python -m constieval.cli report --results output/constieval/runs --recipes baseline,difficult_advice
uv run python -m constieval.cli plot ood_decay --results output/constieval/runs   # any figure, standalone
uv run python -m constieval.cli axes         # every eval axis and its rubric settings
uv run python -m constieval.cli clauses      # the frozen clause set
uv run python -m constieval.cli registry     # every extension point
uv run python -m constieval.cli estimate --config cheap.yaml --arms 2   # cost before spending
```

---

## The one idea

**Retrieval saturates immediately and is not the metric.** Ask a trained model to name the clause
governing a situation and it will, almost always, whatever it actually does next.

The signal is the **gap between knowing a clause and acting on it**, and whether the model can give
the constitution's *stated rationale* rather than a fluent post-hoc one. Three design consequences
run through everything here:

| Consequence | Where it shows up |
|---|---|
| Retrieval is reported as **discrimination**, never recall | Every fabricated clause is paired with a genuine one from the same clause. Accepting real ones counts as much as rejecting fake ones. |
| Compliance and **tension recognition** are scored by separate judge calls | A model that complies without ever noticing anything was at stake has memorized the behavior. One rubric covering both would let a strong compliance score hide that. |
| Justification is graded against the constitution's **own** reason | A different, entirely sensible reason scores low. The axis measures whether the model has the guideline's reasoning, not whether it can produce a defensible one. |

---

## Mental model

**`Item` is the load-bearing abstraction — one prompt shown to the model is one item.** Base items
and stressed items are the same type: a robustness or OOD item is a base item with
`parent_item_id` set, so every stressed score has a clean counterpart on the *same scenario* to
difference against. That pairing is what the delta bars and decay curves are made of; an unpaired
derived item is dropped by the analysis layer rather than compared against a group mean.

```
clause ──build──▶ Item ──transform*──▶ Item(parent=…) ──generate──▶ Completion ──judge×N──▶ ScoreRow[]
```

### Identity rules (everything else depends on these)

- **`item_id` is content-addressed** and carries nothing run-scoped. The same item has the same id
  in every run, which is what lets two runs join row for row.
- **`itemset_id` fingerprints the whole frozen set.** Every results row carries it, and the report
  *refuses* to compare rows whose fingerprints differ — otherwise a quiet edit to one scenario
  shows up as a recipe effect.
- **One RNG stream per family and transform.** Bumping one family's count leaves every other
  family's draws bit-identical, so old item ids keep matching.

### The one rule that protects all of this

**An eval axis is never an `if` inside a judge or a builder.** Adding a family, a pressure wrapper,
an OOD axis, or a metric is a registered plugin plus a YAML line. Two `if family == ...` branches
and the axes stop being orthogonal, comparisons stop being interpretable, and the point is lost.
Treat it as review-blocking.

---

## Layout

```
constieval/
  control/              EVERYTHING TUNABLE LIVES HERE — no prose in Python
    clauses/            the frozen clause sets, cut from a source document
    configs/            base · smoke · checkpoint · hosted · hf_local · qwen36_base · qwen36_lora
    prompts/            items · rubrics · pressure · ood
  core/                 registry · types · hashing · cache · llm · store · stats · parsing
  items/                builders (one per family) + transforms (pressure, ood) + frozen ItemSet
  judges/               the rubric judge, one registered judge per axis
  pipeline/             generate (one pass) → judging (many axes) → side effects → run
  analysis.py           every view the plots and tables derive from — a metric is defined once
  plots/                theme (the only place color is decided) + the 7 Tier A figures
  report.py             figures + the greppable markdown mirror
  cli.py                fire CLI
  scripts/              merge_lora.py · serve_merged.sh (for serving a merged checkpoint)
```

### There is no gold set

The suite ships with **no judge-calibration step**. Judge quality is taken on trust, which has one
consequence worth stating plainly: an absolute level on any single axis is only as good as the
rubric behind it. What stays sound is the **comparison** — two recipes measured against the same
frozen item set, by the same judge, at the same temperature, differ only in the weights. Read the
difference panels and the paired deltas; treat a lone axis score as descriptive.

---

## Shared infrastructure

**Item set.** ~200–800 items depending on config. Free response only — no multiple choice anywhere.

**Clause sets.** A clause set is cut from a source document, and *which document* is the most
consequential choice in the suite — it must be the one the training data actually encoded.

| Clause set | Cut from | Use when |
|---|---|---|
| `constitution_principles_v1` | `docs/claude_constitution_principles.md` — the 8 principles that `src/prompts.py::CONSTITUTION` encodes | **Evaluating the difficult-advice LoRA.** This is the doc its SFT data was generated against. 21 clauses, 12 distractors. |
| `approved_constitution_v1` | `docs/claude_approved_constitution.md` | Evaluating against the richer later constitution. 23 clauses, 10 distractors. |

A thinner source document supports fewer axes, and the suite enforces that rather than papering
over it: `constitution_principles_v1` states no conflict ordering (principle 7 actively resists
one), so the conflict family is disabled for it; and 12 of its 21 clauses state a rule with no
reason, so the justification axis skips those instead of grading against an invented rationale.
Config validation refuses a clause set with no `priority_order` while the conflict family is on.

Per clause the builders produce:

| Family | What it asks | Generated? |
|---|---|---|
| `application` | a scenario at `clear` / `edge` / `ambiguous` difficulty — the atom | LLM |
| `retrieval` | *the same scenario*, asking which principle governs | deterministic |
| `fake_clause` | is this quoted statement one of your principles? (half real, half fabricated) | deterministic |
| `conflict` | two clauses genuinely pulling apart | LLM |
| `over_refusal` | a benign request in the clause's own territory | LLM |
| `persona_drift` | fixed general-manner probes, identical across recipes | deterministic |

Then two transforms derive **paired** items from the application scenarios:

- **`pressure`** — 5 deterministic wrappers: contradicting operator system prompt, authority claim,
  persona pressure, multi-turn rapport, and a compelling argument for crossing the line. No LLM
  call, and the scenario text is never rewritten.
- **`ood`** — 4 labelled distance axes (`domain`, `language`, `format`, `framing`), each with
  ordered values. **Distance 0 is the parent item itself**, so every decay curve starts from the
  exact items it decays away from.

**Held-out clauses.** A subset of each clause set is marked `held_out: true` — decided before any
generation, as the spec requires. They are excluded from *training data* generation, never from evaluation.
`held_out_vs_trained` reports them descriptively; separating generalization from memorization
properly is Tier B.

**Generation pass.** One completion per item per checkpoint, reused by every Tier A judge. An
application item is scored on compliance, tension recognition, *and* justification quality from the
same generation.

**Judge harness.** Rubric-based, **clause text always in the judge's context** (a judge grading
from its own memory of the constitution is grading something else), and **blinded by
construction** — recipe, checkpoint step, and model id are not passed to the judge and cannot be,
because `RubricJudge` never receives them. There is a test that asserts this on the actual prompts.

**Results store.** One row per `(run, recipe, clause, item, axis, score)`, columns declared
explicitly. Every plot and table in the suite derives from this one table and nothing else.

---

## Running against a real model

**The eval is black-box.** It sends prompts and reads text back — no weights, no gradients, no
internals. But a HuggingFace repo is *stored weights, not a running service*, so something has to
do the forward passes. That is the only reason compute ever comes up, and how much you need depends
entirely on whether the checkpoint is already hosted:

| Situation | What it costs | Config |
|---|---|---|
| The model is **already behind an API** (base Qwen3-32B is on OpenRouter as `qwen/qwen3-32b`) | Tokens only. No GPU, no download, no server. | `hosted.yaml` |
| A **small** model you want to run yourself | One `uv sync --extra hf`, runs on a laptop's MPS/CPU | `hf_local.yaml` |
| **Your own LoRA adapter** | Nobody hosts your adapter, so it has to run somewhere: in-process if it fits, otherwise vLLM on a rented GPU | `hf_local.yaml` / `base.yaml` |

Every path needs `OPENROUTER_API_KEY` in `.env`, because the **judge** and the **item generator**
are separate models regardless of how the target is served — grading with the model under test
would not be a judge.

```bash
# zero infrastructure: the target is already served
uv run python -m constieval.cli run --config hosted.yaml --model qwen/qwen3-32b --recipe base
```

**Small model, in-process** (laptop or single GPU, no server):

```bash
uv sync --extra hf     # torch + accelerate + peft, optional extras — not needed for other paths
uv run python -m constieval.cli run --config hf_local.yaml \
  --model Qwen/Qwen3-1.7B --recipe base --max-items 60
# with a LoRA adapter on top:
uv run python -m constieval.cli run --config hf_local.yaml \
  --model Qwen/Qwen3-32B --adapter matboz/qwen3-32b-difficult-advice-lora --recipe difficult_advice
```

`--max-items` caps *base* items while keeping every parent/child pair intact, and suffixes the
`itemset_id` so a quick pass can never be pooled with a full one by accident.

**Large model, served** — the scalable path, and what the repo's GPU playbook already does. Serve
the checkpoint with vLLM, tunnel port 8000, and use the default `vllm` provider:

```bash
bash scripts/serve_lora.sh ./adapter          # on the GPU box
uv run python -m constieval.cli run --config checkpoint.yaml --recipe difficult_advice --step 500
```

Note the repo gotcha: a thinking-trained checkpoint must be evaluated in thinking mode
(`target.enable_thinking: true`) and compared against a *thinking* baseline. Crossing modes is the
easiest way to produce a fake result. `max_tokens` is deliberately generous — a truncated reasoning
trace yields an empty answer that would otherwise be scored as a refusal, and the manifest reports
`truncation_rate` so you can see it happen.

---

## Cost

Judging is ~70% of every bill, so that is where the levers are. Price any config before
spending — counts are exact, only token sizes are estimated:

```bash
uv run python -m constieval.cli estimate --config qwen36_base.yaml --arms 2   # $17.97
uv run python -m constieval.cli estimate --config cheap.yaml --arms 2         # $2.50
```

`cheap.yaml` is ~7x cheaper than the full config and measures the same things on less of the
item set. Re-runs are cheaper still: the call cache means a re-judge or a repeated arm replays
instead of paying.

| Lever | Saving | What it costs you |
|---|---|---|
| **Cheaper judge** (`judge.model: google/gemini-2.5-flash`) | ~10x on judging | Judge quality — measure it with `judge_agreement` rather than assuming |
| **Scope an axis to clean items** (`conditions: [clean]`) | 16% of judge calls | Nothing. Justification asks whether the model has the doc's reasoning; re-asking under a Swahili translation is a different question |
| **Cap OOD distance** (`ood.max_distance: 2`) | ~1/3 of items | The far tail of each axis — the most expensive third, least often decisive |
| **Fewer OOD axes** | proportional | Per-axis coverage. Keep `language` and `format`; `domain` is weakest since base scenarios already vary domain |
| **Fewer difficulties / variants** | proportional | Statistical power per clause — the CIs widen and the report says so |
| **`--max-items N`** | proportional | Coverage, but pairing is preserved and the `itemset_id` is suffixed so a capped run can't be pooled with a full one |

**What not to cut, and why.** Merging compliance and tension recognition into one judge call
saves ~35% and is the most tempting cut here — but a combined rubric lets a strong compliance
score anchor the tension score, and telling those two apart is the entire thesis. Likewise,
dropping the matched genuine probe halves the `fake_clause` family and turns discrimination back
into recall, which saturates and measures nothing. Both are refused by design rather than
exposed as config.

**Prompt caching does not help here** — the cacheable prefix (system + rubric template, ~650
tokens) sits below Anthropic's 1024-token minimum, so it would never trigger.

### Trusting a cheap judge

There is no gold set, so a cheap judge earns trust by cross-check: re-grade a stratified sample
with a strong model and measure how often the pass/fail decision changes. Only the reference
calls are paid for — the completions are reused.

```bash
uv run python -m constieval.cli judge_agreement --run-dir output/constieval/runs/<id> \
  --config cheap.yaml --reference anthropic/claude-sonnet-4.5 --n 120
```

Reports raw agreement and Cohen's kappa per axis, plus every disagreement with both rationales —
which is where a rubric edit comes from. The reference judge is not ground truth; this measures
whether the cheaper judge *changes conclusions*, which is the question that matters when the
output is a recipe comparison.

---

## The figures

All seven regenerate from the results store with one command, for any pair of recipes. Each has a
greppable markdown mirror in `tier_a_results.md` — numbers are never trapped in a PNG.

| Figure | What it answers |
|---|---|
| `clause_heatmap` | Clause × eval-axis. **Which** clauses internalized, not just how much. With exactly two recipes a third diverging panel shows the difference. |
| `retrieval_vs_application` | One point per clause. **Off-diagonal mass is the thesis** — clauses the model can name but does not act on. |
| `compliance_vs_tension` | Complying without recognizing = memorized behavior. |
| `ood_decay` | One line per recipe, **faceted by distance axis, never pooled** — averaging a translation effect with a fiction-framing effect describes neither. |
| `robustness_delta` | Paired stressed − clean, by wrapper and by clause. |
| `side_effect_panel` | Over-refusal, persona drift, reasoning retention, capability. **Always run** — trait data leaks. |
| `checkpoint_trajectory` | Every headline metric vs training step. |

**Conventions.** 95% CI on every aggregate (Wilson for proportions, percentile bootstrap for
means). A recipe keeps the same color in every figure, and the color follows the recipe rather than
its rank — filtering a report down to two recipes does not repaint the survivors. The categorical
palette is validated for colorblind separation; asking for more series than can be told apart is an
error, never a generated hue. Axes are oriented so higher is always better, except on the
side-effect panel, where each series keeps its native direction and says so on its own label.

**Capability regression is ingested, not reimplemented.** Point `side_effects.capability_path` at a
flat JSON of `{"mmlu": 0.72, "gpqa": 0.31}` produced by whatever harness already runs those. A
second implementation would drift from the numbers everyone else quotes.

---

## Extending it

```python
from constieval import register

@register("judge", "my_axis")          # + an entry under `axes:` in control/prompts/rubrics.yaml
class MyJudge(RubricJudge):
    axis = "my_axis"
```

Same shape for `builder` (a new item family), `transform` (a new stress), `llm` (a new provider),
and `plot` (a new figure). `uv run python -m constieval.cli registry` lists everything registered.
Config validation fails loudly *before* any money is spent: an axis declared in the rubrics with no
registered judge, an undeclared pressure wrapper, an OOD axis with no distance-0 anchor, and an
unknown provider are all caught up front.

---

## Tier B — NOT implemented (requires additional training runs)

Deliberately out of scope: each of these needs training runs this package cannot perform, so
shipping a half-version would produce numbers that look like Tier A's and mean something else. The
groundwork that costs nothing is already in place, and noted per item.

1. **Counterfactual causality.** Train on a deliberately *inverted* clause; check that behavior
   flips on that clause and its entailments **only**. One run per inverted clause, 2–3 clean,
   semantically distant clauses. *Already in place:* every clause declares `entailments`, which is
   the ground truth a spillover matrix would be scored against.
2. **Held-out clause generalization.** Evaluate the excluded clauses against a model whose training
   data genuinely omitted them. *Already in place:* 5 clauses are marked `held_out`, decided before
   generation, and `analysis.held_out_vs_trained` computes the comparison — but under Tier A both
   groups were in the training data, so the number is descriptive only.
3. **Data-recipe ablations / scaling curves.** N runs by construction. *Already in place:* `recipe`
   is a first-class store column and every figure is a pairwise comparison over it.
4. **Persistence.** Does the trait survive continued SFT/RL. *Already in place:* `checkpoint_step`
   and the trajectory plot.

**Planned figures:** spillover matrix (inverted clause × all other clauses), held-out vs trained
paired bars, recipe scaling curves.

## Tier B-lite — NOT implemented (requires model internals)

1. **Linear probes** on clause-relevant activations. Needs hidden states, so it needs the weights
   locally rather than an OpenAI-compatible endpoint — the `hf` provider is the natural hook.
2. **Self-report vs behavior consistency.** Separates "represents the value" from "acts on it".
   Closest Tier A analogue is `retrieval_vs_application`, which is behavioral on both axes.

**Planned figure:** probe accuracy vs behavioral compliance, per clause.

---

## Sequencing

1. Freeze the clause list and item set — `items build`, then **pin the printed id** into
   `itemset.id` in your run config. ✅
2. Ship Tier A end to end on an existing checkpoint; confirm the plots render. ✅ (`--smoke` does
   this offline in ~10s.)
3. Wire Tier A into the checkpoint loop — `checkpoint.yaml`, one `run` per saved step.
4. Tier B once a recipe comparison is worth the compute.
