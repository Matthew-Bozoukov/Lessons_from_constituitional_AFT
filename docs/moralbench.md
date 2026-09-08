<!-- ABOUTME: What a published MoralBench run contains on Hugging Face, how it gets there, -->
<!-- ABOUTME: and how the dashboard reads it. Reference doc; the eval itself is src/eval/misalignment/moralbench/. -->

# MoralBench on Hugging Face

MoralBench (Ji et al., [arXiv:2406.04428](https://arxiv.org/abs/2406.04428)) is the
declarative values probe: 88 fixed A/B items over Moral Foundations Theory, scored
mechanically against a released human answer key. No judge, no docker. This document is
about the **published artifact** — what a run puts on the Hub and why — not about the
scoring, which lives in `src/eval/misalignment/moralbench/moralbench.py`.

## Running one

```bash
# turnkey: rents a pod, runs, tears it down (src/eval/managed.py)
uv run moralbench <hf_adapter_or_model>

# or the ordinary framework path, against a pod you already have
uv run evals --name moralbench --target <hf> --server root@<ip>:<port> --ssh-key ~/.ssh/<key>

# no GPU at all — an API target, the cheapest way to check the wiring
uv run moralbench openrouter:qwen/qwen3-32b
```

A ladder shares one pod and one server, which is what makes two arms comparable — the
LoRA is swapped, not the process:

```bash
uv run evals --name moralbench --target Qwen/Qwen3.6-27B <adapter> --server <host> mode=think
```

`mode=think` is required when a **base model** joins a think-stamped ladder: a full model
has no `training_meta.json`, so it would resolve to its template default, and comparison
code refuses to pair arms whose modes differ.

## What lands on the Hub

The repo id is `<HF_ORG>/<today>-moralbench-<model_key>`, minted by the naming law
(`src/utils.py`) and pushed by `run_eval`'s epilogue. It obeys the published-layout
contract (`src/eval/layout.py`) like every other eval:

```
rollouts/records.jsonl        one row per (item, repetition) — 88 x reps rows
results/results.json          the canonical summary (epilogue-written)
results/metrics.json          the same numbers as produced by the eval
results/scores.md             greppable markdown mirror
metadata/run_meta.json        epilogue provenance: git SHA, config, target, mode
metadata/moralbench_run_meta.json   the eval's own: repetitions, swap, system prompt,
                                    pinned upstream commit
README.md                     the dataset card (below)
```

The eval's run_meta is **namespaced**. `run_eval`'s epilogue moves its own file to
`metadata/run_meta.json`; an eval that writes that bare name collides — `FileExistsError`
on Windows, and on POSIX a silent overwrite that loses the eval's provenance.

### `rollouts/records.jsonl`

Self-contained per the "logs means ROLLOUTS" rule: the prompt the model saw AND what it
did with it. One row per (item, repetition):

| field | meaning |
| --- | --- |
| `item_id` | `<source>/<name>`, e.g. `MFQ_30/harm_1` |
| `rep` | repetition index (0-based) |
| `dataset` / `assessment` | `MFQ`\|`MFV` / `binary`\|`comparative` |
| `foundation` / `foundation_stem` | canonical name / upstream's (`care` / `harm`) |
| `part` | `relevance`, `agreement` (MFQ's two halves) or `vignette` |
| `prompt` | verbatim, exactly as sent — swapped if `swap_options` |
| `option_A` / `option_B` | option texts **as presented** |
| `scores` | the released per-option score map for this presentation |
| `correct_option` | higher-scoring option, or `TIE` |
| `raw` / `think` / `answer` | full completion / reasoning trace / visible answer |
| `think_words` | trace length — the empty-`<think>` check (gotcha 1) |
| `parsed` / `parse_tier` | extracted letter (or null) and which rule matched |
| `score` | the released score for `parsed`; **0.0 when unparsed** |
| `finish_reason` | `stop`, `length` (truncated), or `error` |

`think` is recorded for diagnostics and is **never** given to the scorer —
`resolve_trace` splits it off before `parse_answer` sees anything.

### `results/results.json`

Four blocks that are never summed together, because binary (a weighted human-agreement
score) and comparative (accuracy against a key) share no scale:

```jsonc
{
  "MFQ_binary":      { "total": 51.35, "n_items": 20,
                       "min_possible": 37.57, "max_possible": 62.43,
                       "normalized": 0.554,
                       "by_foundation": { "care": { ... }, ... } },
  "MFV_binary":      { ... },                       // scale max 4.0, not 5.0
  "MFQ_comparative": { ... },                       // floor 1.0: ingroup_2 is a tie
  "MFV_comparative": { ...,                          // per-item ceiling 24...
                       "max_possible_deterministic": 23.0 },  // ...but 23 in practice
  "totals_by_repetition": { "0": 121.03, ... },
  "parse": { "parse_rate": 0.939, "invalid_rate": 0.061,
             "tiers": { "exact": 313, "labeled": 93, ... },
             "answer_balance": { "A": 230, "B": 183 } },
  "mode": "think", "repetitions": 5, "swap_options": false
}
```

**`normalized` is the number to compare across arms**, not `total`. Both binary options
score, so the floor is 60% of the ceiling on MFQ and 74% on MFV — a raw total hides most
of the difference. `min_possible`/`max_possible` travel with every block so a reader can
never mistake the scale for 0..max.

`max_possible_deterministic` appears only where duplicate upstream prompts make the
per-item ceiling unreachable — MFV comparative, where two byte-identical questions carry
opposite labels. Its presence in a block *is* the flag.

### The card

`run_eval` writes the standard fields (`experiment`, `date_generated`, `constitution`,
`source_repo` @ commit, `models`, `generation_config`, `schema`, `provenance`).
`constitution` is deliberately **`none`**: MoralBench's taxonomy predates our
constitution and is not aligned to it, which is exactly why a shift measured in it is
evidence of transfer rather than of the model reciting spec-shaped text.

Hub-indexed tags, which are the discovery route:

```
eval-run · eval:moralbench · model:<model_key> · mode:<think|nothink|default>
```

## What is deliberately NOT published

**The 88 prompts and the four answer JSONs never leave this repo.** Upstream publishes no
licence (verified at the pinned commit: no `LICENSE`, no metadata field, no README
statement), and `out_dir` is uploaded verbatim — so the corpus is vendored under
`src/eval/misalignment/moralbench/assets/` and excluded from every push. Item **ids**,
model responses and scores are ours to publish; the prompt corpus is not. See
`assets/NOTICE.md`.

A consequence worth knowing: a published run is not self-contained for someone outside
this repo. They can see which item scored what, but not read the item. That is the
intended trade until upstream adds a licence.

## How the dashboard reads it

`dashboard/lib/evalRuns.ts` discovers runs through the Hub's tag filter
(`/api/datasets?author=<org>&filter=eval-run`) and keeps a `moralbench` adapter for the
generic explorer's rollout browser. The dedicated view is
`dashboard/app/moralbench/page.tsx` + `MoralBenchExplorer.tsx`, reading through
`dashboard/lib/moralbench.ts`:

- **any number of runs side by side** (capped at 6, past which grouped bars stop being
  readable and the table is the better tool);
- **overall** (the four blocks) or **by foundation** (one block, broken out over the six);
- bars drawn against each block's **reachable range**, with the chance baseline marked in
  the same coordinate space so "below chance" is visible rather than arithmetic;
- a run-health table — parse rate, invalid rate, A/B balance — because an unparsed answer
  scores 0, which is below every reachable binary score, so a run with a high invalid
  rate can undershoot a block's own floor and look like a moral finding.

Only **public** repos are visible: the site is token-less by design.

## Reading the numbers honestly

1. **Higher is not better.** The score rewards agreement with the MFQ/MFV human norming
   sample (WEIRD, skewed liberal and educated). A constitution-trained arm that
   downweights "conformed to the traditions of society" scores *worse* here, and that may
   be the constitution working. This measures where a checkpoint's moral weighting sits.
2. **The comparative half is at chance.** Every model in the paper's own tables sits
   within one standard deviation of random guessing, and the first checkpoint we ran was
   below chance on both halves. Report it; do not build on it.
3. **Four items per foundation.** A per-foundation level is coarse. The paired delta
   between arms on identical items is the quantity worth reading.
4. **A single arm is a coordinate, not a finding.** Publish the base alongside it.

## Upstream defects

Three, all preserved rather than corrected, all pinned by `tests/test_moralbench.py` so a
re-copy that changes them fails the suite instead of silently moving a number:
a duplicated MFV Care vignette carrying the wrong text, two byte-identical MFV comparative
questions with opposite labels, and a duplicated MFV Fairness pair. The MFQ comparative
`ingroup_2` tie (A=B=1.0) looks *deliberate* rather than broken — all ten MFQ pivots are
order-consistent and the tie matches its pivot's human mean exactly. Full detail in
`src/eval/misalignment/moralbench/assets/NOTICE.md`.
