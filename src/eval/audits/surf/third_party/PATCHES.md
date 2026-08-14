# SURF — vendored copy and local patches

Upstream: <https://github.com/seoirsem/SURF> @ `7d3fe912612290de0b4d4155fab73058189c2056`
(the tool from *Chunky Post-Training*, arXiv 2602.05910). Vendored 2026-08-07.

Vendored rather than pip-installed for the same reason as
`agentic_misalignment/third_party/` and `odcv/third_party/`: it carries local patches, and a
re-clone silently drops them. **If you re-clone SURF, re-apply everything below.**

## Patches

### 1. `--data-files` on `prepare-dataset` (REQUIRED, not cosmetic)

`surf/cli/main.py`, `surf/extraction/extractor.py`.

Upstream calls `load_dataset(dataset, split="train")` with no file selector. Our corpus repo
`LASR-Callum/2026-08-04-synthdoc-difficult-advice-9-principles` ships **two** JSONL files —
`stage_7_sft.jsonl` (the SFT records) and `stage_6_final.jsonl` (the pre-render stage, a
different schema). An unqualified load tries to merge them and dies:

```
DatasetGenerationError: An error occurred while generating the dataset
```

Verified before vendoring: plain load fails; `data_files="stage_7_sft.jsonl"` yields 2,203
rows with columns `['messages', 'metadata']`, first row roles `['system', 'user', 'assistant']`.
`AttributeExtractor._extract_first_turn` needs exactly that `messages` shape, so the corpus
needs no conversion — only file selection.

`--data-files ''` restores upstream behaviour for single-file repos such as Tulu.

### 2. `prepare-dataset` defaults retargeted at the difficult-advice corpus

| option | upstream | here |
| --- | --- | --- |
| `--dataset` | `allenai/tulu-3-sft-mixture` | `LASR-Callum/2026-08-04-synthdoc-difficult-advice-9-principles` |
| `--data-files` | *(did not exist)* | `stage_7_sft.jsonl` |
| `--num-samples` | `None` (all) | `2203` (the whole corpus) |
| `--n-clusters` | `25000` | `1000` |
| `--extract-model` | `anthropic:claude-opus-4-5-20251101` | `openrouter:openai/gpt-5.6-terra` |
| `--summarize-model` | `anthropic:claude-opus-4-5-20251101` | `openrouter:openai/gpt-5.6-terra` |

`--n-clusters 1000` keeps upstream's attributes-per-cluster ratio. Clustering runs over
*attributes*, not prompts: upstream clusters 50,000 Tulu prompts x 10 attributes = 500k
attributes into 25,000 clusters, ~20 per cluster. Here 2,203 x 10 = ~22k attributes into
1,000 clusters is ~22 per cluster — the same granularity at a twentieth of the scale.

**Deliberately NOT changed:** the standalone `extract` command and the `sweep` / `run-em`
judge and query models still carry upstream's defaults. Only `prepare-dataset` was
retargeted; the judge model is a separate decision and Opus remains upstream's
recommendation for scoring.

### 3. Target `max_tokens` raised 2048 -> 8192 for reasoning targets

`surf/em_loop/loop.py`.

Upstream caps target generations at 2048 tokens, which is ample for a non-reasoning model.
Our targets are Qwen3.6 adapters trained with `thinking: true`: the trace consumes the
budget first, so a 2048 cap truncates inside the think block and the judge scores a
response that has no answer in it at all — as *compliance*, since nothing violates a
principle when nothing was said. Same failure as the "reasoning models need token headroom"
gotcha in the root `CLAUDE.md`.

Judge and query caps are untouched (2048 / 512); only the target needed headroom.

### 4. `prompt_cache_key` passed through on OpenAI-compatible calls

`surf/core/models.py` (`QueryParams`, `_call_openai_impl`, the `openrouter` branch),
`surf/em_loop/judge.py`.

Every judge call resends the same ~2.2k-token prefix — the task line plus the whole rubric —
and only the query/response tail differs. The judge sets one `prompt_cache_key` derived from a
hash of the rubric text, so every call in a sweep shares it. The key is omitted entirely when
None, so no other call site changes behaviour.

**Measured 2026-08-09, and the result is negative for OpenAI models — read this before
assuming caching helps.** Against `openai/gpt-5.6-terra` on OpenRouter:

| access pattern | cached tokens |
| --- | --- |
| byte-identical prompt, repeated | 2,194 of 2,197 (cost $0.0030 -> $0.0004) |
| **shared 2.2k prefix, unique tail (what a sweep actually does)** | **0, on 5/5 calls** |
| same, rubric moved to a system message | 0 on 4/5; the one hit routed to a pricier backend |

So the caching is effectively exact-match on this route, not prefix-based, and a sweep — where
every query/response pair is unique — gets no benefit. `prompt_cache_key` is retained because
it is free and correct, but do not budget for a discount with an OpenAI-family judge.

Anthropic models *do* cache by prefix here, via explicit `cache_control` breakpoints:
`anthropic/claude-sonnet-4.5` cached 2,146 of 2,251 tokens on unique tails, $0.0094 on the
cache-write call then $0.0018 steady state. `anthropic/claude-haiku-4.5` reported 0 cached
under the same test. Using that path would mean splitting the judge prompt into a cached
system block plus a per-example user turn; it was measured but not adopted, since a
Sonnet-with-caching judge and an uncached terra judge cost about the same (~$21 vs ~$22 per
7,200 calls).

### 5. Judge errors are reported instead of silently scoring 0.0

`surf/em_loop/judge.py`.

`SingleJudgeSystem.score` catches every exception and returns `0.0`. That score is
indistinguishable from a genuine "no violation", so a rate-limited or failing judge quietly
teaches the EM loop that a whole region of attribute space is clean, and the run looks
healthy. Upstream's control flow is kept (raising would abort a multi-hour sweep on one bad
call); what is added is an `error_count` and a loud stderr line per failure. A run whose log
contains `!!! judge error` has holes in it and its buffer is not trustworthy — check the
count before reading the results.

### 6. The attribute corpus is loaded once and shared across runs

`surf/em_loop/sampling.py`.

`AttributeFileLoader` materialises every dataset row as a Python dict in `self.data`, and
`SweepRunner` builds one `EMLoop` — and therefore one loader — per run, all inside a single
process (`asyncio.gather` in `sweep.py`). `--num-runs 3` over
`seoirsem/CHUNKY-tulu3-SFT-25k-attributes` therefore held 938,074 records three times and
was OOM-killed after 7m53s on a 15GB machine, having printed nothing but three
"Loading from HuggingFace" lines. A module-level `_CORPUS_CACHE` keyed by
`(source, attribute_column)` returns the already-loaded list instead.

Sharing is safe because the records are read-only after load: `sample_random_entry` builds a
new dict from `entry.get(...)` and `random.sample` returns a new list, so no run can mutate
what another reads. The key includes the column because the same dataset read through a
different attribute column filters to a different subset. Covered by
`scratch/test_surf_corpus_cache.py` (identity sharing, per-column isolation, sampling
unchanged under a fixed seed, corpus unmutated after 200 draws).

### 7. Attribute extraction works on non-chat datasets

`surf/extraction/extractor.py`.

`_process_record` reads `record["messages"]` and returns `None` for anything else, so
extracting from an instruction dataset — `nvidia/OpenCodeInstruct` has `input`/`output` —
produced an **empty attribute file with no error**. Added `_extract_plain_fields`, which
falls back to the usual single-turn field names (`input`/`instruction`/`prompt`/…,
`output`/`response`/`answer`/…). Unknown schemas still skip, so nothing is silently invented.

### 8. The `<1>..<10>` attribute parse tolerates missing closing tags

`surf/extraction/extractor.py`.

`SINGLE_ATTRIBUTION_PROMPT` asks for attributes "listed in XML tags from `<1>` to `<10>`"
but never asks for **closing** tags, while `parse_xml_tags_optional` requires a matched
`<n>...</n>` pair. Claude closes its tags, so upstream never hit this; `gemini-3.7-flash`
reads `<n>` as a delimiter and closes nothing, so **18 of 20 rows** parsed to zero attributes
and were written out as empty lists with no error. `_parse_numbered` now tries the strict
form and a delimiter split, returning whichever recovers more — taking the longer result
rather than preferring strict, so a partially-closed reply does not silently lose its
unclosed entries.

### 9. `--num-samples` draws a seeded random subset, not the head

`surf/extraction/extractor.py`, `surf/cli/main.py` (`--sample-seed`).

Upstream processes rows `0..num_samples-1`. On `nvidia/OpenCodeInstruct` (5M rows) the head
is **100% `domain=generic` and 100% `generation_algorithm=self-instruct`**, where a random
draw of the same size is 72/28 and 52/48 — so the head silently omits an entire domain and
an entire generation algorithm. Now takes a seeded random subset when `num_samples` is
smaller than the dataset; the seed keeps a resumed run on the same subset.

### 10. A rubric may supply its own query-generation prompt

`surf/em_loop/loop.py`, read from the rubric YAML as `query_gen_prompt`.

The query generator is principle-blind *and* format-blind: it only renders attributes into
queries, so no configuration could ask for a particular SHAPE of scenario. That blocks any
sweep whose object is the framing itself — e.g. `rubrics/constitution_in_code.yaml`, which
tests whether the model applies its values when a norm-violating request arrives as code
rather than prose. A rubric may now carry its own generator prompt, keeping the generator
and the judge that scores its output paired in one file. Absent the field, behaviour is
unchanged.

## Added

`rubrics/traits.yaml` — scores violations of the nine principles in
`constitutions/claude_distilled_09_principles_mid_20260804/constitution.md`.

It is deliberately **two-sided**. Principles 8 and 9 make over-refusal, hedging, moralising
and paternalism violations in their own right, so a rubric that only hunted for harmful or
deceptive output would push SURF toward eliciting refusals and score the resulting caution as
success. The rubric says so explicitly and gives worked examples on both sides, plus the
constitution's own "when this does NOT apply" exclusions as false-positive guards.

## Environment

Own nested env, like the other audit tooling (see the root `CLAUDE.md` note that
`src/eval/audits/` is exempt from the eval-framework contract):

```bash
cd src/eval/audits/surf/third_party/SURF
uv sync
cp ../../../../../../.env .env      # OPENROUTER_API_KEY + HF_TOKEN
```
