<!-- ABOUTME: The current best baseline for each arm family — what a new experiment compares against. -->
<!-- ABOUTME: Short by design. One entry per family; change it only with a measured reason and say what. -->

# Current baselines

**If you are generating a new corpus or training a new arm, this file says what to build on
and what to compare against.** It is the answer to "which difficult-advice thing is the
baseline?", which has three plausible answers in the repo's history and only one right one.

Change an entry only when a measurement says to, and record the measurement in the entry.

---

## Difficult advice — **principle-scoped 702**

| | |
|---|---|
| **recipe** | `configs/data/synth/2026-08-01_difficult_advice.yaml` (this IS the principle-scoped recipe) |
| **corpus** | `LASR-Callum/2026-08-21-sonnet45-difficult-advice-principle-scoped-constitution-716` |
| **mixture** | `LASR-Callum/2026-08-21-table2-9284-difficult-advice-principle-scoped-702-train-mixture` → `t2_9284_da_chunk_only_702.jsonl` (9,986 rows, 702 DA = 7.03%) |
| **train config** | `configs/train/2026-08-21_lora_qwen36_table2_9284_difficult_advice_principle_scoped_702_dynbatch.yaml` (seed 0) + `configs/train/2026-08-31_lora_qwen36_table2_9284_difficult_advice_principle_scoped_702_dynbatch_seed_{42,69}.yaml` |
| **adapter** | `LASR-Callum/qwen3.6-27b-lora-t2-9284-da-chunk-only-702-r64-dynbatch` |
| **ODCV** | 11.5% [6.2, 19.6] as published on 65 cells; 10.5% [3.5, 19.3] re-scored on the 57 cells its siblings share |

**Use this, not `da716` and not `synthdoc-716`.**

### Why

The difficult-advice recipe pasted the WHOLE constitution into exactly two of its five LLM
stages — `revise_prompts` and `revise_responses`. Every other stage already saw only its one
target principle. On 2026-08-21 those two injections were deleted as an ablation, and on
2026-08-24 the ablation became the default, because a matched-pair test found no detectable
cost:

| recipe | ODCV MR (65 cells) |
|---|---|
| principle-scoped | **11.5%** [6.2, 19.6] |
| full constitution (`da716`) | 16.3% [10.0, 21.8] |

Overlapping intervals, so the honest claim is "no measurable difference", not "principle-scoped is
better". It wins on being simpler and cheaper at equal measured quality.

The comparison is unusually clean: the principle-scoped run **resumed a copy of the da716 run
directory** with stages 1–4 already in place, so the two arms share byte-identical scenarios
and draft prompts and differ only in the two refine prompts.

### What removing the injection also removed

The constitution's **preamble** — the priority / conflict-resolution section saying how
principles trade off — belongs to no principle chunk, so it reached the generator ONLY through
those two slots. Principle-scoped therefore drops the trade-off guidance entirely, not just the other
eight principles. If a future result turns on how the model resolves conflicts between
principles, this is the first thing to look at.

It is also why the arm is **702 rows, not 716**: without the constitution's framing the content
filter deterministically refused 6 `revise_prompts` calls (they failed again on resample) and 2
rows were lost at `revise_responses`, leaving 708 → 9 × 78 = 702 as the largest trait-balanced
draw. That refusal pattern is itself a measured effect of the change, not noise to engineer
around.

### The two you should NOT start from

| | what it is | why not |
|---|---|---|
| `synthdoc-716` <br>("original", root A) | The first build of the recipe, 3–4 Aug. Confusingly published as `matboz/synthdoc-v2-difficult-advice` — the "v2" is the *generator package* version, not the corpus. | Superseded: it has the four defects v2 fixed (scenario diversity, voice lints, refine-stage metadata, stock openers). |
| `da716` <br>("v2", "diverse", root B) | Root A's defects fixed, 14 Aug. | Superseded by its own principle-scoped fork. Keep it alive only as the control for the existing generator sweep (below). |

### The one live reason to still touch da716

**Every generator-comparison arm is built on da716, not on principle-scoped.** grok DA, GPT DA and
Sonnet concise each freeze da716's stages 1–4 (which include the constitution-injected prompt
revision) and regenerate only the reply. So those three can only be read against da716.

That means: a *new* generator arm should freeze **principle-scoped's** stages 1–4 instead, and will
not be directly comparable to the existing three. Regenerating the existing swaps on principle-scoped
is the clean fix when it is worth the spend.

### Building a NEW arm on top of it

A derived arm (responder swap, length cap, stakes change, any ablation) does not regenerate
the corpus — it **freezes the baseline's early stages and re-runs only the stage it is
changing**, so the two differ in one thing. That is declared in the synth config as:

```yaml
source:
  local_dir: "data/<arm>_source"          # staged from the BASELINE's published stages
  snapshot: "stage_6_draft_responses.jsonl"   # the last stage you are NOT changing
```

Stage that `local_dir` from the principle-scoped run
(`LASR-Callum/2026-08-21-sonnet45-difficult-advice-principle-scoped-constitution-716`), the way
`scratch/sonnet_concise/build_source.py` does for its parent. `load_source_run` hard-asserts
the constitution sha matches, so a source from the wrong parent fails loudly rather than
silently crossing arms.

**Do not stage a new arm from da716's run directory.** Every existing generator swap does,
which is why they can only be read against da716 and not against the baseline.

### Comparing against it in a plot

`scratch/gpt_seeds/plot_seed_mean.py` names the baseline once, as `BASELINE_ARM`, and draws
it as a rule across the whole axis so every other arm reads as a distance from it. A new plot
should import that constant rather than hardcode an arm key; `tests/test_baselines.py`
asserts it matches this file.

---

## Controls (unchanged)

| control | what it is | adapter |
|---|---|---|
| 0% synthetic | Table2 9,284 alone | `LASR-Callum/qwen3.6-27b-lora-table2-only-9284-r64` |
| no-SFT | the base model, fp8 | — |
| matched non-DA | numina control 716 | `matboz/qwen3.6-27b-lora-9284-numina-control-716-r64` |

## Standing rules for any new arm

1. **One pass per seed** when averaging seeds, and score every arm on the cells they all
   kept — `src/eval/misalignment/odcv/odcv.py` has `pick_most_complete_pass` and
   `shared_cells` for exactly this.
2. **Three seeds or it cannot be ranked.** Between-seed spread on ODCV is 1.2–9.4 pp, so a
   single-run gap under ~10 pp is not a difference.
3. **Never rebuild a mixture to reproduce an arm** — copy the published bytes. The builder's
   shuffle depends on the corpus it reads.
