# CoT verbosity expansion — prompt iteration on 5 sampled rows

Target: think tokens 4.95x (that is what takes the 716 difficult-advice rows' trainable
total to 3x, with the answer and user turns untouched). Model `anthropic/claude-sonnet-5`,
temperature 0.7, 5 rows from `LASR-Callum/2026-08-14-table2-9284-difficult-advice-716-train`.

`concept_diff` = LLM autorater listing every proposition in the expansion absent from the
original. "major" = a fact, example, option, hedge or conclusion that changes the argument.

| variant | what changed | multiple | fidelity |
|---|---|---|---|
| v1 `expand_prompt.md` | global word target, CoT + reply shown | 2.40x | 3/5 pass, 2 major, **imported a phrase from the reply** |
| v1 + top-up passes | re-expand until in band | 3.98x | 3/5 pass, 5 major |
| v1 over-ask | ask 2x the target | 3.14x | not audited |
| v2 `expand_prompt_v2.md` | cache-ordered, reply and user turn removed | 2.33x | 4/5 pass, 1 major |
| v3 `expand_prompt_v3.md` | per-paragraph paragraph-count budget | 3.04x | not audited |
| v4 `expand_prompt_v4.md` | fixed `<p>` slot sequence | **1.69x** | worst row 1.01x — copied its source |
| v5 `expand_prompt_v5.md` | v3 + budget quoted per source sentence | 3.63x | 4/5 pass, 1 major |
| v5 @ ask 6.6 | over-ask on top of v5 | 3.74x / 3.32x | one row echoed verbatim |
| **v6 = v5 prompt + split-then-allocate plan** | long paragraphs split at sentence bounds before budgeting | **3.64x** | **5/5 pass, 0 major** |

v6 is the recommendation: same prompt file as v5, `MAX_ALLOC = 3` in `pilot_expand.py`.

## What the iteration established

**A global word target is unsteerable.** v1/v2 asked for one number covering the whole
rewrite and returned ~48% of it, at every asked multiple. Over-asking to compensate raises
variance without raising the mean much (v1 over-ask 3.14x, per-row spread 2.47–4.48x).

**Budget locally, and quote the budget per source sentence.** v3 gave each source paragraph
a paragraph count and the model ignored it whenever the paragraph was short — runs told to
write 510 words came back at 206, 220, 236. It was not refusing length; it could not see
three paragraphs of material in a 63-word paragraph. v5 quotes the same budget divided by the
paragraph's sentence count ("4 sentences, about 85 words of thinking each") and compliance
jumps: best row 501/510, 463/510, 604/680.

**Do not enforce the count structurally.** v4 replaced the budget with an explicit sequence
of `<p>` slots. It collapsed to 1.69x, and its worst row came back with per-paragraph word
counts of 71/109/107/63 against a source of 69/109/107/63 — the tags turned expansion into
segmentation and the model filled the containers by copying.

**Run size is the binding constraint, not the asked multiple.** Runs budgeted at or under
3 output paragraphs (~510 words) land on target; larger ones under-deliver by roughly the
amount they are over. Raising the ask inflates every run's budget and so makes this worse,
which is why v5 @ 4.95 (3.63x) and v5 @ 6.6 (3.74x, 3.32x on a rerun) are indistinguishable.
v6 caps run size instead, by splitting long paragraphs at sentence boundaries *before*
allocating. Splitting before allocating matters: dividing a paragraph's allocation evenly
after the fact hands a 28-word unit the same budget as a 121-word one (a 12.1x local ask
next to a 2.8x one).

**Smaller runs buy fidelity, not just length.** v6 holds the same 3.6x as v5 but goes
4/5 → 5/5 with zero major additions. A run that can only draw on two or three sentences has
much less room to generalise, narrate, or invent a counterfactual — the three failure modes
v2 had to forbid by name.

**The echo failure is discrete and cheap to detect.** At a high ask, rows do not undershoot
gradually — one came back 98.8% character-identical to its source. `RESAMPLE_BELOW = 3.0`
separates it from a genuine short rewrite cleanly; resampling clears it. Two rows needed it
in the v6 run.

**Prompt caching works, once warm.** v2's diagnosis was right: all constant instruction must
precede `<<<cache>>>`. v6 reports 15,640 of 25,413 prompt tokens cached (~62%). The first
call of a run is always a cache write, so firing all rows concurrently from cold wastes it —
send one row, then fan out.

## Where it stands

3.64x against a 4.95x target, with fidelity now clean. Single-pass prompting has plateaued
here: five prompt generations moved 2.33x → 3.64x, and the last three all land at 3.0–3.6x
with heavy per-row variance (v6 rows: 4.61, 4.08, 2.61, 3.23, 4.07). One row (`t5_b08_s000`)
resisted three resamples.

Closing the remaining 1.3x needs something other than prompt wording. The cheapest candidate
is the convergence loop already written in `topup_expand.py` (unrun): with v6's per-row
fidelity at 0 major additions there is now headroom to spend on a second pass, which was not
true when top-up was tried on v1 output and fidelity fell to 3/5. Note that `topup_expand.py`
puts its per-row payload before its constant instruction and carries no `<<<cache>>>` marker,
so it currently caches nothing.

Caveat carried forward: the 4.95x target derives from a think/answer token split
(420,447 / 410,185) that no script in this directory reproduces, and the answer figure
implies 1.05 tokens/word against the sample where think implies 1.37. Worth re-deriving over
all 716 rows before committing to the multiple.

---

# Retarget: 3x the CoT itself (2026-08-25)

The goal changed from 3x total trainable tokens (which needed 4.95x on the think block) to
3x the think block. That is a much easier target, and it is met.

| run | ask | achieved | fidelity |
|---|---|---|---|
| v6 @ ask 4.0 | 4.0 | 2.75x | not audited |
| **v6 @ ask 4.6** | 4.6 | **2.98x** | **5/5 pass, 0 major additions** |

Per row: 2.20x, 3.32x, 2.75x, 3.97x, 2.63x. `v6_3x_out.json`, audit in
`concept_diff_v6_3x.json`.

Config: `expand_prompt_v5.md`, `ASK = 4.6`, `PARA_WORDS = 170`, `MAX_ALLOC = 3`,
`RESAMPLE_BELOW = 2.0`, `anthropic/claude-sonnet-5` at temperature 0.7.

The ask stays well above the target because the achieved/asked ratio is ~0.65 and stable:
4.0 -> 2.75x, 4.6 -> 2.98x. Calibrate the ask, do not expect the model to hit the number.

What this does to the dataset: think 420,447 -> ~1,253,000 tokens, answer unchanged at
410,185, so trainable tokens go 830,632 -> ~1,663,000, a 2.0x on the training total.

## Prefix caching

Fixed. The marker placement was already right in v2; what was missing is that a cache
breakpoint only pays from the *second* call onward — the first call writes it. Firing all
rows concurrently from cold made every one of them race that write and miss, which is why
every earlier run reported `cached_tokens: 0`. `main()` now expands one row alone, then fans
out. Cold start (the ask-4.0 run) hit on 4 of 5 calls: 7,820 = 4 x 1,955, the warm-up being
the one miss. The ask-4.6 run hit on 5 of 5 — 9,775 = 5 x 1,955 — because the ask-4.0 run had
already written the same prefix and it was still live, which is also the reason the two runs
report different hit percentages for an identical prompt.

At 716 rows this is 715 hits and one write. Do not raise `max_workers` past the point where
the warm-up is still serial, and do not reword the instruction block between rows — any edit
above `<<<cache>>>` invalidates the prefix for every row after it.

## Cost at scale

$0.256 for 5 rows = ~$0.051/row at full rate, so ~$37 for 716 rows, less a few dollars of
cache discount that `cost_of` does not model (it prices cached reads at full rate
deliberately, as a conservative floor). Above the ~$20 flag threshold in CLAUDE.md.

---

# Productionised into synth (2026-08-25)

The recipe is now `configs/data/synth/2026-08-25_verbose_cot.yaml`, run with
`uv run synth run --config configs/data/synth/2026-08-25_verbose_cot.yaml [--smoke]`. The scratch
drivers above are kept as the record of how the recipe was found; nothing depends on them.

## Engine additions (`src/data/synth/`)

| addition | why the existing seam did not fit |
|---|---|
| `lint: {ratio_of, min_word_ratio, max_word_ratio}` | `min_chars`/`max_chars` are absolute and `lint_problems` never saw the record, so "2-4.5x of the source field" was inexpressible |
| `verify: [{model, prompts, accept_field, accept_values, save_as}]` | `lint` decides with a regex; "introduces nothing the source did not say" needs a model. Takes a LIST, for the reason below |
| `derive: {fn, args}` | `prompt_vars` handles literals and field-conditional cases, not computation, and the paragraph plan is real code |
| `strip_patterns: {tag: [regex]}` | the prompt buys compliance by making the model label the parts of its answer; those labels must not reach the corpus |

## What the gates caught, before any of it was trusted

**The judge gate had to be built twice.** A first version false-failed 4/5 known-clean
expansions and 5/5 deliberately-inert mutants, flagging "unsupported detail" — exactly the
elaboration the contract permits. On 716 records that would have failed nearly everything,
exhausted every retry, and handed back the control arm at full price. The fix was telling
the judge outright that B is three times longer and that difference is the assignment, then
narrowing the question to "would this change what the assistant DECIDES".

**Two judges, not one.** With additions, omissions and contradictions asked in one call,
planted truncations were detected 0/5 — the omission question loses to the addition
question when they share a prompt. Alone in its own call the same question detects 5/5.

**Measured detection (planted defects, 5 records):**

| planted defect | detected |
|---|---|
| decision-changing addition | 4/5 |
| truncated tail (25%) | 5/5 |
| contiguous middle third deleted | 2/5 |
| one paragraph deleted | 1/5 — ambiguous, see below |
| *inert* elaboration (must NOT fire) | 5/5 correctly ignored |
| unmodified control (must NOT fire) | 5/5 clean on 3, 2 real defects found |

Single-paragraph deletion being undetectable is a property of the data, not a judge
failure: the expansion re-derives its conclusions, so one paragraph of it often carries no
unique content. Truncation — a model running out of steam before the resolution — is the
failure that actually occurs, and it is caught every time.

Two of the five "clean" controls turned out not to be clean. gpt-5.6-terra found an
invented mirror-case and a scenario-contradicting detail in expansions Sonnet had passed
5/5. That is the argument for a different-family judge, measured rather than asserted.

## Reasoning settings: the biggest cost lever, and a trap

Extended thinking is ~70% of the output bill (6,630 output tokens per call against ~1,900
of visible rewrite). Four settings, same 20 records:

| setting | multiple | sd | fallbacks | $/20 |
|---|---|---|---|---|
| default | 3.03x, 3.29x | 0.83, 0.73 | 0-1 | $2.39, $2.05 |
| `{enabled: false}` | 2.45x | **0.20** | 0 | **$0.87** |
| `{effort: low}` | 2.05x | 0.46 | 3 | $1.01 |
| `{max_tokens: 2000}` | 3.29x | 0.73 | 0 | $2.05 |

**`reasoning: {max_tokens: N}` is silently ignored** — output stayed 6,628 tokens/call,
indistinguishable from default. Only `enabled` and `effort` take effect. Two of the runs
above that looked like "capped thinking" were just default thinking with sampling noise.

Thinking off is beautifully behaved — a quarter of the price and a quarter of the variance
— but it SATURATES: raising the ask 4.6 -> 5.6 moved it only 2.45x -> 2.61x while variance
tripled. It cannot reach a 3x target at all. Thinking is what buys the length.

Haiku-4.5 as the expander: 2.04x with **7/20 records failing the fidelity gate outright**.
Half the price, unusable.

## Smoke, and the bug it found

The measurement script reported `0 fallbacks` while the minimum per-row multiple was
1.00x — the fallback row was present but mismarked, because `also` (which stamps
`expansion_status: expanded` on every record) was merged AFTER `on_exhausted.mark`, erasing
it. A fallback row was reporting itself as a clean one, which is the single thing that path
exists to make countable. Fixed and covered by a test.

## The full run overran the budget, and the guard did not stop it

`budget_usd: 68.0` was set and did not fire. `pipeline.run` checks the budget BETWEEN
stages; the expansion is a single stage over 716 records, so nothing was checked until it
was over — and it never got there, because the stage then died on `max_fail_pct`.

Reconstructed spend for that run: ~1,040 expand calls at ~6,630 output tokens each
(1,890 visible + ~4,740 thinking) plus ~1,650 judge calls, so **~$85 (range $84-90)**
against a $68 guard and an $80 authorisation that already had $11 spent against it.

Two things made the smoke a bad predictor:

1. **Retry rate.** The 20-record smoke retried ~34% of records. The full run logged 371
   failed attempts over 716 records — ~52%. Small-sample retry rate is not a reliable
   estimator of large-sample retry rate here, and retries are the whole cost variable.
2. **Content-filter refusals never appeared at smoke scale.** 34/716 (4.7%) of prompts
   were refused OUTRIGHT by Anthropic's own filter (`finish_reason=content_filter`), and
   0/20 were refused in each of seven smokes.

**The guard to actually use:** a per-stage budget check, or a check inside `run_items`,
not one that only runs at stage boundaries. A single-stage pipeline over a large corpus is
exactly the shape `budget_usd` fails to protect, and this config is that shape.

**State left behind:** `output/verbose_cot/20260825_042004/stage_2_expand.partial.jsonl`
holds 682 of 716 records at a corpus multiple of 2.884x, resumable with
`uv run synth run --config configs/data/synth/2026-08-25_verbose_cot.yaml --resume <that dir>`.
The 34 refused records now have a landing place (`on_exhausted.mark_refused`) and would
fall back to their original traces on the next attempt, costing ~$3 to finish.
