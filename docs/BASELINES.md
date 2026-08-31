<!-- ABOUTME: The current best baseline for each arm family — what a new experiment compares against. -->
<!-- ABOUTME: Short by design. One entry per family; change it only with a measured reason and say what. -->

# Current baselines

**If you are generating a new corpus or training a new arm, this file says what to build on
and what to compare against.** It is the answer to "which difficult-advice thing is the
baseline?", which has three plausible answers in the repo's history and only one right one.

Change an entry only when a measurement says to, and record the measurement in the entry.

---

## Difficult advice — **chunk-only 702**

| | |
|---|---|
| **recipe** | `configs/data/synth/difficult_advice.yaml` (this IS the chunk-only recipe) |
| **corpus** | `LASR-Callum/2026-08-21-sonnet45-difficult-advice-chunk-only-constitution-716` |
| **mixture** | `LASR-Callum/2026-08-21-table2-9284-difficult-advice-chunk-only-702-train-mixture` → `t2_9284_da_chunk_only_702.jsonl` (9,986 rows, 702 DA = 7.03%) |
| **train config** | `configs/train/lora_qwen36_t2_9284_da_chunk_only_702_dynbatch_2xh200.yaml` (+ `2026-08-31_*_s42/_s69` replicates) |
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
| chunk-only | **11.5%** [6.2, 19.6] |
| full constitution (`da716`) | 16.3% [10.0, 21.8] |

Overlapping intervals, so the honest claim is "no measurable difference", not "chunk-only is
better". It wins on being simpler and cheaper at equal measured quality.

The comparison is unusually clean: the chunk-only run **resumed a copy of the da716 run
directory** with stages 1–4 already in place, so the two arms share byte-identical scenarios
and draft prompts and differ only in the two refine prompts.

### What removing the injection also removed

The constitution's **preamble** — the priority / conflict-resolution section saying how
principles trade off — belongs to no principle chunk, so it reached the generator ONLY through
those two slots. Chunk-only therefore drops the trade-off guidance entirely, not just the other
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
| `da716` <br>("v2", "diverse", root B) | Root A's defects fixed, 14 Aug. | Superseded by its own chunk-only fork. Keep it alive only as the control for the existing generator sweep (below). |

### The one live reason to still touch da716

**Every generator-comparison arm is built on da716, not on chunk-only.** grok DA, GPT DA and
Sonnet concise each freeze da716's stages 1–4 (which include the constitution-injected prompt
revision) and regenerate only the reply. So those three can only be read against da716.

That means: a *new* generator arm should freeze **chunk-only's** stages 1–4 instead, and will
not be directly comparable to the existing three. Regenerating the existing swaps on chunk-only
is the clean fix when it is worth the spend.

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
