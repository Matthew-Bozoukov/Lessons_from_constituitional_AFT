<!-- ABOUTME: The generator ablation — what was built to swap Anthropic for xAI in the -->
<!-- ABOUTME: difficult-advice recipe, what the corpora measure, and what confounds remain. -->

# Generator ablation: difficult advice, written by xAI instead of Anthropic

The megadoc asks for one control under "Controls (before anything else)":

> **Generator ablation.** Same pipeline, >=2 generator models. See how much the generation
> pipeline vs model spec vs generator model matter.

and lists, under cross-cutting gaps, *"Generator model never ablated anywhere."* This is
that control. It began as an attempt to replace Anthropic with Gemini, which failed, and
finished as two grok corpora — one of which is a usable ablation and one of which is not.

**Read this before training on either corpus.** Neither is a drop-in replacement for
`difficult-advice-v2`, and the ways they differ are measured below rather than guessed at.

## The two arms

| | `difficult_advice_grok_716.yaml` | `difficult_advice_grok_responder_716.yaml` |
|---|---|---|
| what grok writes | scenarios, prompts AND responses | responses only |
| what is reused | nothing | the baseline's stages 1–4, frozen |
| rows delivered | 673 / 716 | **703 / 716** |
| contract vs baseline | `retries` 5→12, `max_fail_pct` 2→15 | **identical** |
| length ratio vs baseline | 2.64x | **1.67x** |
| length-only AUC vs baseline | 0.975 | **0.864** |
| cost | $61.79 | **$15.96** |
| HF | `LASR-Callum/2026-08-20-difficult-advice-grok-716` | `LASR-Callum/2026-08-21-difficult-advice-grok-responder-716` |

**Use the responder arm.** The all-grok arm regenerates the questions as well as the
answers, so it differs from the baseline in its situations, domains, trait balance and
user-turn length on top of the variable it is supposed to isolate. It is kept as a record
of what an all-xAI pipeline produces, not as an experiment.

### Why "responder swap" is the ablation

`load_source_run` freezes the baseline's first half. Only `draft_responses` and
`revise_responses` are paid for:

```
                      all-grok                 responder swap
1 chunk_constitution  same 9 principles        same 9 principles
2 write_scenarios     grok-4.3  (regenerated)  baseline  (REUSED)
3 draft_prompts       grok-4.3  (regenerated)  baseline  (REUSED)
4 revise_prompts      grok-4.6  (regenerated)  baseline  (REUSED)
5 draft_responses     grok-4.6                 grok-4.6   <- the swap
6 revise_responses    grok-4.6                 grok-4.6   <- the swap
```

In the baseline recipe Sonnet 5's job *is* `revise_responses` — it writes the final answer
that trains. So "how much did Sonnet matter" is exactly "how much does the model writing
the answer matter", and this arm varies that and nothing else.

`scratch/build_da716_prompt_source.py` stages the inputs. It does not merely draw from the
same pool as the `da716` training arm — it replays that arm's own selection (`pick_balanced`,
seed 0, no RNG draw before it) and independently reproduces the two statistics its train
config documents: trait counts `80/80/80/80/80/79/79/79/79` and 635 distinct domains. The
corpora therefore answer the same questions **scenario id for scenario id**, which is what
makes the paired comparison below possible.

## The headline result

With the questions held identical, the generators diverge at the **revision** step, and
directionally:

| | draft | revised | revision effect |
|---|---|---|---|
| baseline | 2242 (Haiku 4.5) | 2670 (Sonnet 5) | **1.19x — lengthens** |
| grok arm | 1964 (grok-4.6) | 1568 (grok-4.6) | **0.80x — shortens** |

At the draft stage the two nearly match (1.14x). Essentially the whole final gap is made
at revision: Sonnet expands the draft, grok-4.6 compresses it. Sonnet's contribution is not
that it writes long from scratch — it is that its revision pass *adds*.

Where the extra length goes: sentence **count** 1.14x, sentence **length** 1.35x. Sonnet
writes a similar number of longer sentences, not more points.

## What else differs (paired, 703 shared questions)

Rates are per 1,000 characters, so they are length-independent — this is what separates
"longer, so more of everything" from a real stylistic difference.

| feature | Sonnet | grok | per-1k ratio |
|---|---|---|---|
| contractions | 21 | 1 | **14.9x** |
| offer phrases ("I can", "instead") | 2 | 1 | **3.9x** |
| em-dashes | 8 | 2 | **2.4x** |
| second person ("you", "your") | 13 | 6 | 1.3x |
| refusal phrases | 1 | 1 | **0.39x** (grok refuses more densely) |
| first person | 5 | 3 | 1.02x (same rate) |
| numbers | 1 | 1 | 1.06x (same rate) |

So grok is not simply a terser Sonnet. Per unit of text it **refuses more** and **offers
alternatives less**, while matching Sonnet exactly on first-person framing and on citing
concrete numbers from the prompt.

### Punctuation is a perfect fingerprint

| character | Sonnet | grok |
|---|---|---|
| ASCII apostrophe `'` | 100% of docs | 2.7% |
| Unicode apostrophe `’` | **0%** | 58.9% |
| ASCII quote `"` | 94.5% | 1.7% |
| curly quote `“` | **0%** | 72.3% |

The two corpora use disjoint punctuation. A model trained on either inherits its
typography. Note this does NOT explain the separability: normalising every curly character
to ASCII leaves bag-of-words AUC at 0.9999 — the vocabularies genuinely differ.

## Confounds that remain

Measured with `scratch/compare_generator_arms.py`, which reports the same AUC the
2026-08-17 PAR/PC investigation used.

1. **Length, AUC 0.864.** The one that matters. Verbosity is a nuisance variable, not a
   value, and 0.864 is where peer-critique was called defective (0.85 on length alone —
   and a model was trained on it before the check ran). Train only with length reported as
   a covariate, or compare at the draft stage where the arms nearly match.
2. **grok-4.6 drafts AND revises.** The baseline drafts with Haiku and revises with Sonnet,
   so it gets a second model's critique; this arm does not. grok-4.3 and grok-4.20 were both
   measured for the drafter slot and each leaves ~8% of rows unanswerable at the length floor.
3. **Hidden reasoning.** grok-4.6 cannot disable it (the endpoint 400s); Haiku and Sonnet
   ran non-thinking.
4. **13 rows short** of 716, in t6/t7/t8.

Bag-of-words AUC ~1.0 is **not** on this list. Any two generators have distinct lexical
fingerprints; the 0.70 gate was built for within-corpus arm leakage, and the megadoc itself
notes it "is probably the wrong bar for a substance contrast".

## Why the length floor bites xAI and not Anthropic

`draft_responses` and `revise_responses` enforce `min_chars: 700`. That was calibrated to
sit just under Haiku's natural floor — **zero** of the baseline's 1,968 rows fall below it,
and its median sits 3.8x above. For grok the same number lands mid-distribution, so it cuts,
and it cuts hardest where the principle's natural answer is shortest: t7 ("honour operator
adjustments") wants a boundary, t8 ("be substantively helpful") wants an enumerated
alternative. In the all-grok arm t7 finished at 30/80 while t8 finished at 69/80.

Three things that do **not** fix it, all measured:

- **Raising reasoning effort makes output shorter, not longer** (single-shot pass on
  matched inputs: grok-4.3 reasoning off 31%, effort low 0%, effort high 19%). The model
  spends itself in the hidden trace and then answers tersely.
- **Retry failures correlate per prompt**, so `(1-p)^n` understates them ~3x — 23.3% of rows
  failed all three attempts where independence predicts 8.1%. Judge any retry-budgeted stage
  by per-row all-attempts-fail rate, never per-call.
- **Lowering the floor plateaus** near 6.7% from 500 down to 300, because the residual rows
  fail on banned vocabulary or a missing tag, not on length.

What does work is a model with fewer unsalvageable prompts: re-running every stuck row 10
more times, grok-4.20 has 4/30 that never pass and grok-4.6 has 1/30.

## Reproducing

```bash
# 1. stage the baseline's exact 716 prompts (free)
uv run python scratch/build_da716_prompt_source.py

# 2. generate the responder arm (~$16, ~30 min)
uv run scripts/data/synth/build_dataset.py \
  --config configs/data/synth/difficult_advice_grok_responder_716.yaml

# 3. diagnose it before training on it
uv run python scratch/compare_generator_arms.py \
  --arm output/synthdoc_grok_responder_716/<ts>/dataset.jsonl \
  --arm_label grok --base_label sonnet
```

`--smoke` on step 2 does not truncate a loaded source (the source is whatever the staged
directory holds), so point it at a small one:
`--overrides "source.local_dir=data/da716_prompt_source_smoke"`.

## Two resume traps found building this

Both cost a run each and neither is obvious:

1. **A completed `stage_N_*.jsonl` snapshot makes resume reuse the stage wholesale.** Raising
   `retries` and resuming changed nothing until the snapshot was deleted and the
   `.partial.jsonl` checkpoint kept — that combination re-runs only the rows that failed.
2. **The same trap at `export_sft`** silently exported 620 rows from a 673-row stage 7,
   reporting success.

## Why not Gemini

The arm was first attempted with Gemini and abandoned. Gemini's safety layer blocks the
hardest principle-4 (harm, CBRN/cyber-adjacent) `draft_prompts` calls persistently — 26/716
survived **zero of six** resamples on 3.7-flash even at `safety_settings: BLOCK_NONE` — and
those are the corpus's most valuable rows. xAI exposes no request-side safety knob and needed
none: 90 probe calls on principle 4 blocked zero, and the live run completed 716/716
scenarios and 716/716 draft_prompts with no content filtering at all. Every grok failure in
this work was a length lint, never a refusal.
