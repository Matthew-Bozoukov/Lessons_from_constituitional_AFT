# Hugging Face as the data source

The repository holds code, configuration and analysis. Datasets, generated
corpora, evaluation outputs and caches live on Hugging Face — see the root
`AGENTS.md`. This document describes how the visualizer reads that data without
becoming slow, and what happens when the Hub is not there.

---

## The problem this solves

Before this change the site baked everything into the build. One Petri run
accounted for almost all of it:

| | bytes |
| --- | --- |
| `lib/generated/content-index.json`, serialized | 781.2 KB |
| ...of which one Petri run's `transcripts` | 750.6 KB |
| ...of which that run's `messages` alone | 724.5 KB |
| everything else in the corpus, combined | 30.6 KB |

That 724 KB was server-rendered into `/petri/index.html` and shipped to every
visitor, whether or not they opened a single transcript. The same 28 transcripts
projected to their summary rows — id, sidecar file, scenario, category, outcome,
scores, tags, message count, size — come to **17.2 KB**. The median individual
transcript is **23.2 KB**, the largest **43.5 KB**.

So the split is not a judgement call. Ship 17.2 KB of index, fetch ~23 KB when a
reader actually opens something. A reader who opens ten transcripts still
transfers less than the old page cost before it rendered.

## The split

**Baked into the static build** — everything a *listing* needs:

- entry frontmatter: title, date, status, tags, models, metrics, summary
- Petri scenario seeds and run-level scores
- one summary row per transcript (`transcript_index`)
- dataset statistics: record count, splits, categories, average turns
- the asset list, with sizes and URLs

Listing pages are fully static. `/`, `/logs`, `/evals`, `/models`, `/findings`
and the `/petri` overview render from the baked index with no client-side fetch.

**Read at prerender time from a per-entry sidecar** — content only one page shows:

- the rendered Markdown body of each entry, in `lib/generated/bodies/<slug>.md`

`lib/content.ts` is imported by every page, so anything in the index ships with
every page. Only `/entry/[slug]` and `/petri` render a body, one at a time, so
bodies live in sidecars that `entryBody()` in `lib/body.ts` reads during
prerender. The body still ends up inlined in that page's static HTML; it just
stops riding along on the other twenty-three.

This mattered as soon as the corpus became real: the investigation write-ups run
to several hundred lines each, and inlining them put **215.6 KB of prose** into
the shared index, taking it to 290.6 KB against a 300 KB budget. Splitting them
out brought the index back to **67.1 KB**. `lib/body.ts` must never be imported
from a `"use client"` file — it uses `node:fs`.

**Deferred to a runtime fetch** — everything a *single opened item* needs:

- Petri transcript bodies (`messages`, `judge_summary`), one JSON per transcript
- dialogue dataset records, paged in chunks of 50
- **SFT corpora, paged by byte range straight out of the published `.jsonl`**
  (see below)
- large raw artifacts (`raw-grader-responses.jsonl`, pipeline logs), which were
  already download links rather than page content

### Reading a raw JSONL by byte range

Pre-chunking requires a publish step, and almost none of the corpora on the Hub
had one: 41 of 49 dataset entries were published as a plain `mixture.jsonl` or
`stage_7_sft.jsonl`, so the chunked reader resolved nothing and the viewer had
nothing to show. Chunking them after the fact would mean uploading tens of
megabytes of derived copies of data that is already there.

They do not need it. JSONL is line-delimited, and the Hub serves `resolve` URLs
with `accept-ranges: bytes`, `access-control-allow-origin: *` and
`access-control-expose-headers: *`. So the browser asks for a window of bytes,
keeps the whole lines inside it, and starts the next request where this one
stopped. `content-range` gives the file size for free, which is what the reader
shows as progress.

The build's whole job is to name the file and its size — nothing is downloaded:

```
dataset.stream = { url, path, total_bytes, window }   # window defaults to 256 KB
```

Three properties the reader in `lib/lazy.ts` must hold, all covered by
`tests/jsonl-stream.test.mjs`:

1. **Byte accounting is exact.** The window is sliced at the last newline *byte*,
   never in decoded text — slicing decoded text splits multi-byte characters at
   every non-ASCII boundary and shifts every subsequent offset.
2. **A record larger than the window grows the window.** Otherwise the offset
   never advances and Load-more can never finish.
3. **A 200 response is the whole file.** A server that ignores `Range` is
   honoured rather than re-requested.

Which file gets read is an **allowlist** (`DATA_FILE_PATTERNS` in
`scripts/index-content.mjs`), not "the biggest `.jsonl`". These repos also
publish `verdicts.jsonl`, `assistant_spans.jsonl` and per-question eval records;
pointing a conversation viewer at those would render garbage while looking like
it worked. A repo whose data file is not recognised gets a notice listing its
candidates, and the fix is `hf_source.data_file`, not a wider heuristic.

**Private repos are refused at build time.** The build authenticates and the
browser does not, so a private repo resolves perfectly during a build and 401s
for every visitor — a viewer that works only on the developer's machine.
`fetchRepoInfo` reads the flag and the build declines to wire up a reader,
saying so on the page.

On 2026-08-10 seven of the ten private repos were made public so their bundles
are anonymously readable: the three MMLU runs, both psychosis runs, the
SWE-bench mini run and the Arena-Hard regen bundle. Their upstream sources
(`cais/mmlu`, `princeton-nlp/SWE-bench_Verified`, `lmarena-ai/arena-hard-auto`)
are public and ungated.

**Three stay private, and should:** `2026-08-05-lmsys-answer-cache` and both
`2026-08-08-lmsys-*` runs embed verbatim user prompts from
`lmsys/lmsys-chat-1m`, which is `gated=auto` on the Hub — access requires
accepting its licence agreement. Publishing those repos would redistribute
gated third-party data outside that agreement. Their metrics are still on the
site, because the build reads them with a token and bakes the numbers into the
index; only the source links 404 for a visitor.

### The blend, and where it comes from

`/datasets` leads with the one number these corpora exist to vary: the share of
constitution-grounded synthetic data in the mixture. It is computed in
`lib/composition.ts` from the `by_source` block of a published
`mixture_stats.json` — never estimated, and never derived for a corpus that is
not a blend.

Two rules keep that honest:

- **A share of `null` is not a share of zero.** `null` means the grouping is not
  a mixture source list (a raw synthdoc corpus groups by constitution trait), so
  the page says the question does not apply. `0` is a measured control.
- **An unrecognised source counts as general**, which understates the
  intervention rather than overstating it. Adding a new constitution generator
  means adding its name to `CONSTITUTION_SOURCES`.

The bar carries **two** categories, not the ten a real mixture has. Ten hues
past the point of distinguishability would bury the one contrast the experiment
is about; the ten are in the table under the bar with exact counts. The two
colours are validated with the dataviz palette checker against the dark chart
surface (lightness band, chroma floor, deutan/tritan separation, normal-vision
separation, contrast), and the bar carries a 2px gap and a named legend so the
split survives being read without colour.

### Fixtures must declare themselves, and the generator must carry that across

`scripts/hf-discover.mjs --generate` writes stub entries from dataset cards. It
originally carried the card's summary but not its status, which silently dropped
the `mock: true` flag from `2026-07-30-visualizer-mock-dialogues` — eleven
hand-written dialogues that exist to exercise this viewer, rendering on
`/datasets` as a real corpus with no badge and no banner.

The generator now reads the marker the fixture cards actually carry
(`MOCK DATA` / `NOT A TRAINING CORPUS` in the `experiment` field) and writes
`mock: true`. `tests/rendered-html.test.mjs` asserts it from the repo id, so a
new fixture cannot arrive unflagged by accident.

`/datasets` raises the warning **inside** the viewer, against the corpus on
screen, and marks the fixture's row in the picker. It does not carry a
page-level "some entries are mock" banner, because the page renders one corpus
at a time and that banner would sit above a real corpus — casting doubt on real
evidence because a fixture exists elsewhere in the list.

### Measured effect

Initial payload = the HTML document plus every `/_next/static` asset it
references. Uncompressed / gzipped:

| route | before | after | change |
| --- | --- | --- | --- |
| `/petri` | 4224.3 KB / 1133.6 KB | **2055.1 KB / 559.9 KB** | −51% / −51% |
| `/petri` HTML only | 944.7 KB / 241.1 KB | **188.6 KB / 38.0 KB** | −80% / −84% |
| `/datasets` | 2476.4 KB / 660.4 KB | **1060.5 KB / 288.8 KB** | −57% / −56% |
| `/evals` | 3306.5 KB / 893.8 KB | **1889.0 KB / 521.5 KB** | −43% / −42% |
| `/`, `/logs`, `/models`, `/findings` | ~853–877 KB | unchanged | those pages never carried the bulk |

The content index itself went from **781.2 KB to 88.7 KB** serialized — an 89%
reduction. What remains is 30.6 KB of entry text and metadata, 19.6 KB of Petri
scenario seeds and 19.0 KB of transcript summary rows; 725.3 KB moved to lazy
sidecars.

When the same run is served from a published HF dataset rather than from disk,
`/petri` renders at **85.7 KB** of HTML — smaller still, because the local build
no longer needs to also emit the artifact copies.

Reproduce with `node scripts/index-content.mjs` (it prints baked vs deferred
bytes) and by measuring `out/` after `npm run build:netlify`.

### Why not fetch everything at runtime

Listings would need a client-side round trip before first paint, the pages would
lose their static HTML, and a Hub outage would blank the site rather than
degrade one panel. Headline metrics and run listings are small; there is nothing
to gain by deferring them.

### Why not bake everything

That is what the numbers above measure. It also makes build time and deploy size
grow linearly with the corpus, which is the thing the HF convention exists to
prevent.

### Why the build does not download bulk files

The build fetches **only `manifest.json`** (39 KB for the run in this repo) and,
when it needs an artifact listing, the tree API. It never downloads
`transcripts.jsonl`. Transcript shards are published alongside the bulk export,
so the browser fetches `transcripts/<id>.json` straight from the Hub's CDN.
HF sets `access-control-allow-origin` on `resolve` URLs — verified against the
live Hub and exercised in `tests/hf-source.test.mjs` — so no proxy is needed.

The tree API caps a response at 1000 entries and continues through a
`Link: rel="next"` header, so the listing follows that chain up to
`HF_TREE_MAX_PAGES` pages and sets `truncated` when it stops early.
**Directories count against that cap**: a repo with a deep tree can return a
full page containing nothing but directory entries, so reading only the first
page reports a repo full of files as having none. Measured against
`LASR-Callum/agentic-misalignment-qwen3.6-27b-transcripts`: 6 pages, 2618 files,
100.5 MB — one page returns 0 files. The tree endpoint does not send
`x-repo-commit`; the resolved commit comes from the `manifest.json` fetch.

Note that an artifact list is baked into the index, so a repo with thousands of
files will not fit the 300 KB index budget as a flat asset list. Such a repo
needs a `manifest.json` that names the handful of files worth linking, not a
full tree listing.

---

## Architecture

```
content/<type>/<slug>/index.md          frontmatter, optionally with hf_source
        │
        ▼
scripts/index-content.mjs               build time
        │
        ├─ hf_source present? ──yes──▶ scripts/hf-source.mjs
        │                               fetch manifest.json (small, cached)
        │                               transcript_base = HF resolve URL
        │
        └─ no ────────────────────────▶ read local files
                                        shard transcripts into
                                        public/generated-transcripts/<slug>/
                                        transcript_base = /generated-transcripts/<slug>
        │
        ├──────────────────────────────▶ lib/generated/bodies/<slug>.md
        │                                one Markdown sidecar per entry
        ▼                                       │
lib/generated/content-index.json                │  lib/body.ts, at prerender
metadata only, baked into every page            ▼
        │                                app/entry/[slug]/page.tsx
        ▼                                app/petri/page.tsx
app/components/PetriRunViewer.tsx       lib/lazy.ts fetches
app/components/DatasetViewer.tsx        `${transcript_base}/${file}` on demand
```

Both source kinds produce the identical index shape. A component never knows or
cares whether a sidecar comes from this origin or from the Hub — only the base
URL differs.

### Files

| path | role |
| --- | --- |
| `scripts/hf-source.mjs` | Build-time Hub client: cached fetch, token handling, graceful failure |
| `scripts/index-content.mjs` | Builds the baked index and the lazy sidecars |
| `lib/lazy.ts` | Client-side loader with an in-memory cache |
| `lib/body.ts` | Prerender-time body reader. Server only — uses `node:fs` |
| `lib/content.ts` | Types, including `PayloadSource` and `HfStatus` |
| `.hf-cache/` | Build cache, gitignored |
| `lib/generated/bodies/` | Per-entry Markdown sidecars, gitignored |
| `public/generated-transcripts/` | Locally sharded transcripts, gitignored |

---

## Declaring a Hugging Face source

Add `hf_source` to an entry's frontmatter:

```yaml
---
title: "Petri audit — qwen-3-32b-philosophy-spec-msm-aft-cot @ 9a00c85c"
date: 2026-07-29
hf_source:
  repo_id: LASR-Callum/2026-07-29-msm-philosophy-spec-focused-discovery
  revision: main          # optional; a 7–40 char commit sha pins it
  manifest: manifest.json # optional
---
```

`repo_id` alone may be given as a bare string. A **pinned** revision is never
revalidated — the cache hit is authoritative forever, which makes repeat builds
free and makes a deploy reproducible. A floating revision (`main`) revalidates
with `If-None-Match` and accepts a `304`.

Migration is additive. An entry may declare `hf_source` **and** keep its files on
disk: the build prefers the Hub and silently falls back to the local copy if the
Hub is unreachable, noting the fallback in the UI. Nothing has to move in one
step, and an entry with no `hf_source` behaves exactly as before.

---

## Mock data must declare itself

Some entries are fabricated interface fixtures: they exist so the viewers can be
built and reviewed without real data. A fixture that reads as a research result
is the worst failure this site can have, so every fixture entry carries an
explicit frontmatter flag:

```yaml
---
mock: true
---
```

`mock: true` renders an unmissable yellow banner above the content, a `MOCK`
badge on the entry's listing card, and a tint on that card. Absent means real.

The flag is **explicit on purpose**. It is not inferred from a `demo-data` tag,
a `fictional-` model id, or a slug pattern, because every one of those can be
forgotten or changed while the entry still contains invented numbers. A new
fixture that omits the flag renders as real, so the flag is part of adding one.

Two rules the code holds, both about not discrediting real evidence:

- A page that shows **one** item scopes the warning to that item, not the
  collection. Warning about a real run because a fixture exists elsewhere in the
  corpus would undermine the evidence the banner exists to protect.
- `/petri` picks a **non-mock** run explicitly rather than taking the first in
  index order, so a fixture cannot become the flagship result the day one
  happens to sort first.

Fixtures are published to the Hub like anything else, so they exercise the same
loading path as real data. Their dataset cards state `constitution: none`, mark
`mock_data: true`, and say in the `experiment` field that nothing in them
measures any model:

| repo | what it backs |
| --- | --- |
| `LASR-Callum/2026-07-30-visualizer-mock-petri-audit` | the Petri viewer fixture |
| `LASR-Callum/2026-07-30-visualizer-mock-dialogues` | the dataset browser fixture |

Entries with no bulk payload at all - the fictional evals, logs and findings -
are markdown only. There is nothing to fetch for them, so they carry the `mock`
flag and no `hf_source`.

---

## Publishing a dataset the visualizer can read

The publisher lives with the other HF code, in
`experiments/teaching-claude-why/synthdoc/publish.py`, and is exposed as
`synthdoc.cli publish`. It reuses `huggingface_hub` the same way
`synthdoc/snapshots.py` does.

### 1. Write the dataset card

The required fields come from the root `AGENTS.md` and are **enforced** —
`CardFields` raises if any is missing, and `constitution` must be stated
explicitly even when the answer is `none`. Keep the card in git next to the
export, so the metadata is reviewed rather than retyped:

```yaml
experiment: One sentence on what produced this.
date_generated: "2026-07-29"      # the date the data was GENERATED
constitution: MSM philosophy spec (…), or the literal string: none
source_repo: https://github.com/org/repo
source_commit: f8dd135            # defaults to HEAD if omitted
models:
  chloeli/qwen-3-32b-…: target; adapter revision 9a00c85c… over Qwen/Qwen3-32B @ 9216db57…
  anthropic/claude-opus-5: judge
generation_config: {epochs: 3, max_turns: 30, target_temperature: 0.7, …}
schema:
  id: Stable transcript identifier.
  messages: Full multi-turn dialogue; each item has role and content.
provenance: The exact command that regenerates this.
```

See
`experiments/vulnerabilities/exports/2026-07-29-msm-philosophy-spec-focused-discovery/dataset-card.yaml`
for a filled-in example. Anything not in the required set is preserved verbatim
under "Additional detail".

### 2. Publish

```bash
cd experiments/teaching-claude-why
uv run python -m synthdoc.cli publish \
  --kind=petri \
  --export=../vulnerabilities/exports/2026-07-29-msm-philosophy-spec-focused-discovery \
  --repo=<org>/2026-07-29-msm-philosophy-spec-focused-discovery \
  --card=../vulnerabilities/exports/2026-07-29-msm-philosophy-spec-focused-discovery/dataset-card.yaml
```

It **dry-runs by default** and lists every file with its size; pass
`--dry_run=False` to upload. Repo names are validated against
`<YYYY-MM-DD>-<short-experiment-description>` before anything is staged.

For a dialogue corpus use `--kind=dialogues --export=path/to/dialogues.jsonl`,
which additionally pre-chunks the records.

### 3. Published layout

```
<repo>/
  README.md                   the dataset card
  manifest.json               SMALL — the only file the site build fetches
  transcripts/<id>.json       one per transcript, fetched lazily in-browser
  chunks/chunk-NNN.json       dialogue records, paged lazily in-browser
  data/ results/ artifacts/ assets/    the canonical export, byte-identical
```

`results/transcripts.jsonl` and `data/scenarios.jsonl` are uploaded unchanged, so
the Petri export shape in `CLAUDE_CODE_PETRI_EXPORT_GUIDE.md` is preserved and is
what a citation should point at. The shards are derived, not a replacement.

The publisher warns if `manifest.json` exceeds 512 KB. That manifest is baked
into the static build, so unbounded growth is the one regression that would
quietly undo this design.

### 4. Point the visualizer at it

Add `hf_source` to the entry's frontmatter and rebuild. `npm test` fails the
build if a transcript body ever reappears in the baked index, or if the index
exceeds a 300 KB budget.

---

## Tokens

Public datasets need no token; the build works anonymously and that is the
default path.

**Every dataset this site reads is public, and that is a requirement, not a
coincidence.** The browser fetches transcript shards and dataset chunks
*directly* from the Hub with no credentials, so a non-public dataset would 404
for every visitor no matter what the build could see. Anything the site displays
must therefore be anonymously readable.

Verified 2026-07-30 against the live Hub, with `--no-netrc` and no auth header:
all nine `LASR-Callum` datasets report `private=false`, `gated=false`,
`disabled=false`; manifests, transcript shards, dataset chunks, raw artifacts
and the full 8.2 MB corpus all return `200` anonymously; and every response
carries `access-control-allow-origin` echoing the request origin, which is what
lets a deployed page fetch them. A build with `HF_TOKEN`,
`HUGGING_FACE_HUB_TOKEN`, `HUGGINGFACEHUB_API_TOKEN` and `HF_HOME` all unset
reports `token_present: false`, emits zero notices, and resolves every
HF-backed entry to `ok`.

To re-check after publishing something new:

```bash
curl -s --no-netrc -o /dev/null -w '%{http_code}\n' \
  https://huggingface.co/datasets/<repo>/resolve/main/manifest.json
```

For a private dataset, `scripts/hf-source.mjs` reads `HF_TOKEN`,
`HUGGING_FACE_HUB_TOKEN` or `HUGGINGFACEHUB_API_TOKEN` from the environment and
sends it as a bearer header. On Netlify, set it as an environment variable in the
site settings.

The token is never logged. It is read only inside `authHeaders()`, every message
that reaches a log passes through `redact()`, which strips both the literal value
and anything matching `hf_[A-Za-z0-9]{8,}`, and a test asserts the on-disk cache
contains no credential. A `401`/`403` produces the hint "private dataset? set
HF_TOKEN in the environment" — with no token value in it.

Note that the browser fetches transcript shards **directly**, with no
credentials. A private dataset's shards will therefore 404 for a visitor; the
viewer shows a load error for that transcript while the rest of the page stays
intact. Private data belongs in a private deployment, not behind a public site.

---

## Caching

`.hf-cache/` holds one JSON file per `(endpoint, repo, revision, path)`, keyed by
a SHA-256 of that tuple, containing the body, the `ETag` and the resolved commit.

- A **pinned** revision short-circuits on a cache hit: no request at all.
- A **floating** revision is served from cache within `HF_CACHE_TTL_SECONDS`
  (default 3600), otherwise revalidated with `If-None-Match`; a `304` refreshes
  the timestamp without re-downloading.
- `HF_OFFLINE=1` uses cache only and never opens a socket.
- Requests time out after `HF_TIMEOUT_MS` (default 20000), so a slow Hub cannot
  hang a deploy. The timeout is **per request**, so a paginated listing is bounded
  by `HF_TREE_MAX_PAGES × HF_TIMEOUT_MS` in the worst case; lower either if a
  deploy needs a tighter ceiling.

The directory is gitignored, so a developer's cache never leaks into a Netlify
build and the two cannot disagree about what a floating revision means.

| variable | default | effect |
| --- | --- | --- |
| `HF_ENDPOINT` | `https://huggingface.co` | Hub base URL |
| `HF_TOKEN` | unset | bearer token for private datasets |
| `HF_CACHE_TTL_SECONDS` | `3600` | freshness window for floating revisions |
| `HF_TIMEOUT_MS` | `20000` | per-request timeout |
| `HF_TREE_MAX_PAGES` | `20` | tree pages to follow before reporting a truncated listing |
| `HF_OFFLINE` | unset | cache-only, no network |

---

## How failures degrade

**Nothing about a Hugging Face failure may fail a Netlify deploy.** Every path
below has been exercised.

| failure | build | site |
| --- | --- | --- |
| Hub unreachable, nothing cached | exits 0, prints one notice per entry | entry keeps its frontmatter, metrics and research note; the evidence panel says the data is unavailable and names the repo |
| Hub unreachable, cached copy exists | exits 0 | serves the cached metadata, flagged `stale` |
| dataset missing or renamed (404) | exits 0, notice names the repo | as above |
| private dataset, no token (401/403) | exits 0, notice suggests setting `HF_TOKEN` | as above |
| `manifest.json` malformed | exits 0, notice says it is not valid JSON | as above |
| HF declared *and* local files present | exits 0, notice says it fell back | renders from disk; the run header says the Hub copy was unreachable |
| a single transcript 404s at runtime | n/a | that panel shows a load error; the rest of the page is unaffected, and re-selecting retries |

`/petri` does not 404 when a payload is missing. Only a genuinely empty
collection is a 404 — an unreachable dataset degrades to a described gap, because
a page that vanishes is indistinguishable from a routing bug.

Verified end to end by building with `HF_ENDPOINT` pointed at a dead port with
the cache cleared: exit code 0, 22 static pages generated, one notice printed,
and `/petri` rendering its metrics and research note above a message naming the
unreachable repo.

---

## Tests

`tests/hf-source.test.mjs` covers the client against a local stand-in for the
Hub — URL construction, manifest fetch, pinned-revision cache short-circuit,
`304` revalidation, offline mode, token redaction, tree pagination (including a
first page of directories only, the page cap, and refusing a `rel="next"` that
points off-endpoint), and the four failure modes — plus two guarantees about the
corpus itself:

- no transcript body or judge summary may appear in the baked index
- the baked index must stay under 300 KB

`tests/rendered-html.test.mjs` continues to assert that every route
server-renders real content.
