<!-- ABOUTME: The current best baseline for each arm family — what a new experiment compares against. -->
<!-- ABOUTME: Short by design. One entry per family; change it only with a measured reason or an explicit project decision, and say which. -->

# Current baselines

**If you are generating a new corpus or training a new arm, this file says what to build on
and what to compare against.** It is the answer to "which difficult-advice thing is the
baseline?", which has four plausible answers in the repo's history and only one right one.

Change an entry only when a measurement or an explicit project decision says to, and record
which in the entry.

---

## Difficult advice — **neutral 752**

| | |
|---|---|
| **recipe** | `configs/data/synth/da.yaml`: principle-scoped (no stage sees more than its one principle), Haiku 4.5 writes scenarios, draft prompts and draft responses, Sonnet 5 revises the prompts and the responses, and no model or developer name survives in any exported field |
| **constitution** | `constitutions/claude_distilled_09_principles/constitution.md`: the nine principles at full length, Claude and Anthropic neutralised |
| **corpus** | `dougalldeepmind/2026-09-14-da-synth` @ `013886238fca238c4d54ace96530f444bb2b2f02`: 752 rows (t1 78, t2 85, t3 85, t4 83, t5 85, t6 84, t7 83, t8 84, t9 85) |
| **mixture** | `dougalldeepmind/2026-09-15-da-7-mix` @ `c8a65ab574bef277aaefc55664c1d4dff03b4ef5`, from `configs/data/mixture/da.yaml`: 10,000 rows = the 2026-09-08 nosynth base (9,300) + exactly 700 DA (78 × t1–t7, 77 × t8–t9) |
| **train config** | the shared recipe `configs/train/sft.yaml` with `model=qwen36 data_repo=dougalldeepmind/2026-09-15-da-7-mix data_revision=c8a65ab574bef277aaefc55664c1d4dff03b4ef5 seed=0` (global batch 16 on any GPU count) |
| **adapter** | `dougalldeepmind/2026-09-15-qwen36-0-da-7` @ `903c47ef0d5bc63bc2b55b4bdb23a8ba3af58dc0`: seed 0, trained 2026-09-15 on 1× H200 (625 steps, about 4 h), `thinking: true` |
| **ODCV** | not yet measured |

**Use this for all new difficult-advice work**, not principle-scoped 702, `da716` or `synthdoc-716`.

### Why it changed on 2026-09-14

**A project decision, not a measurement.** Every earlier DA corpus was generated against
`constitutions/archive/claude_distilled_12_principles_mid/constitution.md`, which names Claude
and Anthropic: 21 of 708 rows of the principle-scoped corpus name either in trained text. The
organisms are not that model, so for the paper every arm is regenerated against a constitution
that names neither and retrained on the 2026-09-08 nosynth base mix.

The recipe is principle-scoped 702's with three changes: the neutral full-length constitution
(chosen over `constitutions/abridged/` so the alignment target changes in identity, not in
length); a `\bclaude\b` / `\banthropic\b` lint at the last stage that writes each exported
field; and one sentence in both revise prompts, "never name the model you are, or the company
that built you". It is a fresh generation — new scenarios, new prompts — not a fork of the
principle-scoped run, so the two arms share no rows. Details and costs: `docs/LOG.md`,
2026-09-14.

**Until it has an ODCV number, the last measured DA number is principle-scoped 702's** —
11.5% [6.2, 19.6] on 65 cells — and nothing yet says the two are equivalent. Measure it before
drawing a conclusion from a gap to it.

### Why the recipe is principle-scoped

The difficult-advice recipe used to paste the WHOLE constitution into exactly two of its five
LLM stages — `revise_prompts` and `revise_responses`. Every other stage already saw only its
one target principle. On 2026-08-21 those two injections were deleted as an ablation, and on
2026-08-24 the ablation became the default, because a matched-pair test found no detectable
cost:

| recipe | ODCV MR (65 cells) |
|---|---|
| principle-scoped | **11.5%** [6.2, 19.6] |
| full constitution (`da716`) | 16.3% [10.0, 21.8] |

Overlapping intervals, so the honest claim is "no measurable difference", not "principle-scoped
is better". It wins on being simpler and cheaper at equal measured quality. That test resumed a
copy of the da716 run directory with stages 1–4 in place, so the two arms shared byte-identical
scenarios and draft prompts and differed only in the two refine prompts.

**What it withholds.** The constitution's **preamble** — the priority / conflict-resolution
section saying how principles trade off — belongs to no principle chunk, so at `principle`
granularity no stage sees it. If a future result turns on how the model resolves conflicts
between principles, this is the first thing to look at.

**What it costs at generation time.** Without the constitution framing the request as
alignment-training data, Anthropic's content filter refuses some `revise_prompts` calls, and the
refusals repeat on retry: 13 of 765 scenarios in neutral 752 (11 at `revise_prompts`, 6 of them
principle t1), 8 of 716 in principle-scoped 702. Generate a few percent over the target, and
draw the exact count in the mixture (`balance_by: trait_id`).

### The three you should NOT start from

| | what it is | why not |
|---|---|---|
| principle-scoped 702 <br>("chunk-only") | This recipe against the Claude-naming mid document, 2026-08-21: corpus `LASR-Callum/2026-08-21-sonnet45-difficult-advice-principle-scoped-constitution-716`, mixture `LASR-Callum/2026-08-21-table2-9284-difficult-advice-principle-scoped-702-train-mixture`, adapter `LASR-Callum/qwen3.6-27b-lora-t2-9284-da-chunk-only-702-r64-dynbatch` (seeds 42 and 69 trained 2026-08-31), ODCV 11.5% [6.2, 19.6] on 65 cells. | Superseded 2026-09-14 by neutral 752: its constitution and 21 of its rows name Claude or Anthropic, and it was trained on the table-2 base, not the nosynth one. Its number stays the reference only until neutral 752 is measured. |
| `synthdoc-716` <br>("original", root A) | The first build of the recipe, 3–4 Aug. Confusingly published as `matboz/synthdoc-v2-difficult-advice` — the "v2" is the *generator package* version, not the corpus. | Superseded: it has the four defects v2 fixed (scenario diversity, voice lints, refine-stage metadata, stock openers). |
| `da716` <br>("v2", "diverse", root B) | Root A's defects fixed, 14 Aug. | Superseded by its own principle-scoped fork. Keep it alive only to read the existing generator swaps (below). |

**Every existing generator-comparison arm is built on da716.** grok DA, GPT DA and Sonnet concise
each freeze da716's stages 1–4 and regenerate only the reply, so those three can only be read
against da716 — and they carry the Claude-naming constitution too. Rebuild them on neutral 752's
stages before they go into a comparison with it.

### Building a NEW arm on top of it

A derived arm (responder swap, length cap, stakes change, any ablation) does not regenerate
the corpus — it **freezes the baseline's early stages and re-runs only the stage it is
changing**, so the two differ in one thing. That is declared in the synth config as:

```yaml
source:
  local_dir: "data/<arm>_source"              # staged from the BASELINE's published stages
  snapshot: "stage_6_draft_responses.jsonl"   # the last stage you are NOT changing
```

Stage that `local_dir` from neutral 752's run: every snapshot is published under `stages/` in
`dougalldeepmind/2026-09-14-da-synth` (`stage_5_revise_prompts.jsonl`,
`stage_6_draft_responses.jsonl`, `stage_7_revise_responses.jsonl`, …), the way
`scratch/sonnet_concise/build_source.py` does for its parent. `load_source_run` hard-asserts
the constitution sha matches, so the derived config's `constitution:` must be
`constitutions/claude_distilled_09_principles/constitution.md`, and a source from the wrong
parent fails loudly rather than silently crossing arms.

### Comparing against it in a plot

`scratch/gpt_seeds/plot_seed_mean.py` names the baseline once, as `BASELINE_ARM`, and draws
it as a rule across the whole axis so every other arm reads as a distance from it. A new plot
should import that constant rather than hardcode an arm key; `tests/test_baselines.py`
asserts it matches this file. Until neutral 752 has an ODCV run its arm is declared there
with no seeds, so the figure draws no baseline rule; add the seeds when the run is published.

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
