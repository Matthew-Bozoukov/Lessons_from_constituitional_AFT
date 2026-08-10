// Builds lib/generated/content-index.json - the only content the static build
// bakes in - plus the lazily-fetched sidecars that everything heavy lives in.
//
// The rule this script enforces:
//
//   Anything a LISTING needs is baked. Anything a single opened ITEM needs is
//   deferred to a sidecar the browser fetches on demand.
//
// So the index carries titles, dates, models, metrics, scenario seeds and a
// per-transcript summary row, but never a transcript's `messages`. On the
// corpus in this repo that is the difference between a 781 KB index and a
// 55 KB one, almost all of it one Petri run.
//
// An entry's payload can come from disk (`content/<type>/<slug>/`) or from a
// Hugging Face dataset declared as `hf_source` in frontmatter. Both produce the
// same index shape. HF is consulted for METADATA ONLY; bulk files are linked,
// never downloaded, so build time does not scale with corpus size.

import { promises as fs } from "node:fs";
import path from "node:path";
import matter from "gray-matter";
import {
  contentRoot,
  contentTypeFor,
  projectRoot,
  slugify,
  titleFromMarkdown,
  toPosix,
  walk,
} from "./content-utils.mjs";
import {
  fetchRepoInfo,
  fetchRepoJson,
  fetchRepoListing,
  offline,
  redact,
  repoUrl,
  resolveUrl,
  tokenPresent,
} from "./hf-source.mjs";

const outputDirectory = path.join(projectRoot, "lib", "generated");
const outputFile = path.join(outputDirectory, "content-index.json");
/** Per-entry Markdown sidecars, kept out of the shared index. */
const bodyRoot = path.join(outputDirectory, "bodies");
const publicAssetRoot = path.join(projectRoot, "public", "content-assets");
const publicDatasetRoot = path.join(projectRoot, "public", "generated-datasets");
const publicTranscriptRoot = path.join(projectRoot, "public", "generated-transcripts");

/** Collected for the build report; also surfaced in the index for the UI. */
const notices = [];

function warn(message) {
  const text = redact(message);
  notices.push(text);
  console.warn(`  ! ${text}`);
}

function normalize(value) {
  return JSON.parse(JSON.stringify(value));
}

function bytes(value) {
  return Buffer.byteLength(JSON.stringify(value ?? null));
}

/**
 * Point `./assets/...` and `./artifacts/...` links in an entry's body at
 * wherever its files actually ended up. For a local entry that is the copy
 * under `public/content-assets/`; for an entry served entirely from the Hub it
 * is the HF resolve URL, because nothing was copied locally to link to.
 */
function rewriteAssetLinks(body, prefix) {
  return body
    .replace(/(\]\()\.\/(assets|artifacts)\//g, `$1${prefix}$2/`)
    .replace(/(<img[^>]+src=["'])\.\/(assets|artifacts)\//g, `$1${prefix}$2/`);
}

/**
 * Write one entry's rendered Markdown to its own sidecar.
 *
 * The body used to live in `content-index.json`, which `lib/content.ts` imports
 * and therefore every page in the site carries. With the fabricated fixtures
 * replaced by the real investigation write-ups - several of them 400-800 lines -
 * that meant every page shipped ~216 KB of prose it would never render, and the
 * index came within 3% of its 300 KB budget.
 *
 * Only `/entry/[slug]` and `/petri` render a body, one at a time, and both are
 * server components that prerender. So a body is a per-item payload in exactly
 * the sense the Hugging Face split already established, and it belongs in a
 * sidecar for the same reason.
 */
async function writeBody(slug, body) {
  await fs.mkdir(bodyRoot, { recursive: true });
  await fs.writeFile(path.join(bodyRoot, `${slug}.md`), body, "utf8");
}

async function copyEntryAssets(entryDirectory, type, slug) {
  const files = await walk(entryDirectory);
  const copied = [];
  for (const file of files) {
    if (file.endsWith(".md")) continue;
    // The publish card describes the entry; it is not evidence a reader
    // downloads. Excluded here for the same reason `hfAssets` excludes it, so
    // both backends produce the same artifact list.
    if (/^dataset-card\.(ya?ml|json)$/.test(path.basename(file))) continue;
    const relative = path.relative(entryDirectory, file);
    if (relative.startsWith("..")) continue;
    const destination = path.join(publicAssetRoot, type, slug, relative);
    await fs.mkdir(path.dirname(destination), { recursive: true });
    await fs.copyFile(file, destination);
    const stat = await fs.stat(file);
    copied.push({
      name: path.basename(file),
      path: `/content-assets/${type}/${slug}/${toPosix(relative)}`,
      size_bytes: stat.size,
      kind: path.extname(file).slice(1).toLowerCase() || "file",
    });
  }
  return copied;
}

async function readJsonl(file) {
  const raw = await fs.readFile(file, "utf8");
  return parseJsonl(raw, file);
}

function parseJsonl(raw, label) {
  return raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, index) => {
      try {
        return JSON.parse(line);
      } catch {
        throw new Error(`Invalid JSONL at ${label}:${index + 1}`);
      }
    });
}

function messagesFor(record) {
  const candidate =
    record.messages || record.conversation || record.turns || record.dialogue;
  if (Array.isArray(candidate)) return candidate;
  if (record.prompt !== undefined || record.response !== undefined) {
    return [
      { role: "user", content: String(record.prompt || "") },
      { role: "assistant", content: String(record.response || "") },
    ];
  }
  return [];
}

// ---------------------------------------------------------------------------
// Hugging Face sources
// ---------------------------------------------------------------------------

/**
 * Normalize the `hf_source` frontmatter block.
 *
 * ```yaml
 * hf_source:
 *   repo_id: LASR-Callum/2026-07-29-msm-philosophy-spec-focused-discovery
 *   revision: 9a00c85c            # optional; a pinned sha skips revalidation
 *   manifest: manifest.json       # optional
 *   data_file: mixture_think.jsonl  # optional; the JSONL the viewer pages
 * ```
 * A bare string is accepted as shorthand for `repo_id`.
 *
 * `data_file` overrides the filename allowlist in `pickDataFile`. It exists so
 * that a repo publishing several corpora - filtered and unfiltered, thinking
 * and not - can say which one the viewer should read, rather than having the
 * build guess and be quietly wrong.
 */
function normalizeHfSource(raw) {
  if (!raw) return null;
  const source = typeof raw === "string" ? { repo_id: raw } : raw;
  const repoId = String(source.repo_id || source.repo || "")
    .replace(/^hf:\/\//, "")
    .replace(/^\/+|\/+$/g, "");
  if (!repoId) return null;
  return {
    repo_id: repoId,
    revision: String(source.revision || source.rev || "main"),
    manifest: String(source.manifest || "manifest.json"),
    data_file: source.data_file ? String(source.data_file) : "",
  };
}

/**
 * Pull the small manifest for an HF-backed entry.
 *
 * Returns `{ ok, manifest, commit }` or `{ ok: false, error }`. Callers must
 * treat a failure as "render what frontmatter already gives us", never as a
 * build error - a missing dataset must not break a Netlify deploy.
 */
async function loadHfManifest(source, label) {
  const result = await fetchRepoJson(source.repo_id, source.revision, source.manifest);
  if (!result.ok) {
    warn(`${label}: ${result.error}`);
    return { ok: false, error: result.error };
  }
  if (!result.json || typeof result.json !== "object") {
    warn(`${label}: ${source.manifest} is not a JSON object`);
    return { ok: false, error: "manifest is not a JSON object" };
  }
  return {
    ok: true,
    manifest: result.json,
    commit: result.commit || "",
    cached: Boolean(result.cached),
    stale: Boolean(result.stale),
  };
}

/** Turn an HF file listing into the entry's downloadable-artifact list. */
async function hfAssets(source, label) {
  const listing = await fetchRepoListing(source.repo_id, source.revision);
  if (!listing.ok) {
    warn(`${label}: could not list files (${listing.error})`);
    return [];
  }
  if (listing.truncated) {
    // Say so rather than presenting a partial list as complete: a reader has no
    // way to tell a short artifact list from an exhaustive one.
    warn(
      `${label}: file listing truncated at ${listing.pages} pages ` +
        `(${listing.files.length} files); raise HF_TREE_MAX_PAGES to list the rest`,
    );
  }
  if (listing.files.length === 0) {
    warn(`${label}: ${source.repo_id} listed no files at revision ${source.revision}`);
  }
  return listing.files
    // The card, the card's source and the entry body are all already rendered on
    // the page. Offering them again as downloads is noise, not evidence.
    .filter(
      (file) =>
        !/^(README\.md|\.gitattributes|manifest\.json|index\.md|dataset-card\.(ya?ml|json))$/.test(
          file.path,
        ),
    )
    // Per-transcript shards are an implementation detail of lazy loading, not
    // artifacts a reader downloads one by one.
    .filter((file) => !file.path.startsWith("transcripts/"))
    .filter((file) => !file.path.startsWith("chunks/"))
    .map((file) => ({
      name: path.posix.basename(file.path),
      path: resolveUrl(source.repo_id, source.revision, file.path),
      size_bytes: file.size,
      kind: path.posix.extname(file.path).slice(1).toLowerCase() || "file",
      remote: true,
    }));
}

// ---------------------------------------------------------------------------
// Petri runs
// ---------------------------------------------------------------------------

/**
 * The summary row for one transcript: everything the explorer's list and
 * filters need, and nothing else. `messages` and `judge_summary` are the bulk
 * and both ride along in the sidecar instead.
 */
function transcriptSummary(record, file, sizeBytes) {
  return {
    id: String(record.id),
    file,
    scenario_id: String(record.scenario_id || ""),
    category: String(record.category || "uncategorized"),
    outcome: String(record.outcome || "unknown"),
    scores: record.scores && typeof record.scores === "object" ? normalize(record.scores) : {},
    tags: Array.isArray(record.tags) ? record.tags.map(String) : [],
    message_count: Number(record.message_count ?? messagesFor(record).length),
    size_bytes: Number(sizeBytes || 0),
  };
}

/** File name a transcript's sidecar is stored under, in both backends. */
function transcriptFile(id) {
  return `${slugify(String(id))}.json`;
}

/**
 * Write one JSON sidecar per transcript, so opening a transcript costs one
 * small request instead of shipping the whole run.
 */
async function shardTranscripts(slug, records) {
  const destination = path.join(publicTranscriptRoot, slug);
  await fs.rm(destination, { recursive: true, force: true });
  await fs.mkdir(destination, { recursive: true });
  const sizes = new Map();
  let total = 0;
  for (const record of records) {
    const name = transcriptFile(record.id);
    const payload = `${JSON.stringify(record)}\n`;
    await fs.writeFile(path.join(destination, name), payload, "utf8");
    sizes.set(String(record.id), Buffer.byteLength(payload));
    total += Buffer.byteLength(payload);
  }
  return { base: `/generated-transcripts/${slug}`, sizes, total };
}

/** Build a Petri manifest from files on disk. */
async function localPetriManifest(entryDirectory, slug) {
  const files = await walk(entryDirectory);
  const scenarioFile = files.find((file) => file.endsWith("scenarios.jsonl"));
  const transcriptSource = files.find((file) => file.endsWith("transcripts.jsonl"));
  const scoreFile = files.find((file) => file.endsWith("scores.json"));
  if (!scenarioFile && !transcriptSource && !scoreFile) return null;

  const transcripts = transcriptSource ? await readJsonl(transcriptSource) : [];
  const { base, sizes, total } = await shardTranscripts(slug, transcripts);

  return {
    source: { kind: "local" },
    scenarios: scenarioFile ? await readJsonl(scenarioFile) : [],
    scores: scoreFile ? JSON.parse(await fs.readFile(scoreFile, "utf8")) : {},
    transcript_index: transcripts.map((record) =>
      transcriptSummary(record, transcriptFile(record.id), sizes.get(String(record.id))),
    ),
    transcript_base: base,
    transcript_count: transcripts.length,
    deferred_bytes: total,
  };
}

/**
 * Build a Petri manifest from a published HF dataset, using only the small
 * manifest file. Transcript bodies stay on the Hub and are fetched by the
 * browser from `transcripts/<id>.json` when a reader opens one.
 */
function hfPetriManifest(source, manifest, commit) {
  const transcripts = Array.isArray(manifest.transcripts) ? manifest.transcripts : [];
  return {
    source: {
      kind: "hf",
      repo_id: source.repo_id,
      revision: source.revision,
      commit,
      url: repoUrl(source.repo_id),
    },
    scenarios: Array.isArray(manifest.scenarios) ? normalize(manifest.scenarios) : [],
    scores:
      manifest.scores && typeof manifest.scores === "object" ? normalize(manifest.scores) : {},
    transcript_index: transcripts.map((record) =>
      transcriptSummary(
        record,
        String(record.file || transcriptFile(record.id)),
        record.size_bytes,
      ),
    ),
    transcript_base: resolveUrl(source.repo_id, source.revision, "transcripts"),
    transcript_count: transcripts.length,
    deferred_bytes: transcripts.reduce((sum, item) => sum + Number(item.size_bytes || 0), 0),
  };
}

// ---------------------------------------------------------------------------
// Dialogue datasets
// ---------------------------------------------------------------------------

function datasetStats(records) {
  const turns = records.map((record) => messagesFor(record).length);
  const roleCounts = {};
  const splits = {};
  const categories = {};
  for (const record of records) {
    for (const message of messagesFor(record)) {
      const role = String(message.role || "unknown");
      roleCounts[role] = (roleCounts[role] || 0) + 1;
    }
    const metadata = record.metadata || {};
    const split = String(metadata.split || record.split || "unspecified");
    const category = String(metadata.category || record.category || "uncategorized");
    splits[split] = (splits[split] || 0) + 1;
    categories[category] = (categories[category] || 0) + 1;
  }
  return {
    average_turns:
      turns.length > 0
        ? Number((turns.reduce((sum, value) => sum + value, 0) / turns.length).toFixed(1))
        : 0,
    role_counts: roleCounts,
    splits,
    categories,
  };
}

async function localDatasetManifest(entryDirectory, slug) {
  const files = await walk(entryDirectory);
  const dataFile = files.find(
    (file) => file.endsWith(".jsonl") && file.includes(`${path.sep}data${path.sep}`),
  );
  if (!dataFile) return null;

  const records = await readJsonl(dataFile);
  const chunkSize = 50;
  const chunks = [];
  const destinationRoot = path.join(publicDatasetRoot, slug);
  await fs.mkdir(destinationRoot, { recursive: true });

  let deferred = 0;
  for (let index = 0; index < records.length; index += chunkSize) {
    const chunkNumber = Math.floor(index / chunkSize);
    const name = `chunk-${String(chunkNumber).padStart(3, "0")}.json`;
    const payload = `${JSON.stringify(records.slice(index, index + chunkSize))}\n`;
    await fs.writeFile(path.join(destinationRoot, name), payload, "utf8");
    deferred += Buffer.byteLength(payload);
    chunks.push(`/generated-datasets/${slug}/${name}`);
  }

  return {
    source: { kind: "local" },
    source_file: `/content-assets/datasets/${slug}/${toPosix(path.relative(entryDirectory, dataFile))}`,
    format: "jsonl",
    record_count: records.length,
    chunk_size: chunkSize,
    chunks,
    stats: datasetStats(records),
    deferred_bytes: deferred,
  };
}

/**
 * Dataset manifest from a published HF dataset. The publisher pre-chunks the
 * records into `chunks/chunk-NNN.json`, so the build only reads the summary and
 * the browser pages through chunks exactly as it does for local datasets.
 */
function hfDatasetManifest(source, manifest, commit) {
  const dataset =
    manifest.dataset && typeof manifest.dataset === "object" ? manifest.dataset : manifest;
  const chunks = Array.isArray(dataset.chunks) ? dataset.chunks : [];
  return {
    source: {
      kind: "hf",
      repo_id: source.repo_id,
      revision: source.revision,
      commit,
      url: repoUrl(source.repo_id),
    },
    source_file: resolveUrl(
      source.repo_id,
      source.revision,
      String(dataset.source_file || "data/dialogues.jsonl"),
    ),
    format: String(dataset.format || "jsonl"),
    record_count: Number(dataset.record_count || 0),
    chunk_size: Number(dataset.chunk_size || 50),
    chunks: chunks.map((chunk) =>
      /^https?:\/\//.test(chunk) ? chunk : resolveUrl(source.repo_id, source.revision, chunk),
    ),
    stats:
      dataset.stats && typeof dataset.stats === "object"
        ? normalize(dataset.stats)
        : { average_turns: 0, role_counts: {}, splits: {}, categories: {} },
    deferred_bytes: Number(dataset.total_bytes || 0),
  };
}

// ---------------------------------------------------------------------------
// Raw JSONL on the Hub, read by byte range
// ---------------------------------------------------------------------------
//
// Almost every SFT corpus in this project was published as a plain `.jsonl`
// with no chunking step, so the chunked path above resolves nothing for them
// and the viewer had nothing to show. They do not need chunking: the Hub serves
// byte ranges on `resolve` URLs, so the browser can page a 28 MB mixture
// directly. The build's whole job here is to name the right file and its size.

/**
 * Files that hold a conversation corpus, best first.
 *
 * An allowlist, not "the biggest .jsonl in the repo". These repos also contain
 * `verdicts.jsonl`, `assistant_spans.jsonl`, `cluster_summaries.jsonl` and
 * per-question eval records - pointing a conversation viewer at those would
 * render garbage while looking like it worked. A repo whose data file is not
 * recognised gets a notice naming its candidates, so the fix is to declare
 * `data_file:` in frontmatter rather than to widen a heuristic.
 *
 * `mixture_think` outranks `mixture`: same records, but the thinking variant
 * carries the reasoning traces that make the think/nothink distinction visible.
 */
const DATA_FILE_PATTERNS = [
  /^sft_dataset(_[\w-]+)?\.jsonl$/,
  /^stage_7_sft\.jsonl$/,
  /^sft_[\w-]+\.jsonl$/,
  /^mixture_think\.jsonl$/,
  /^mixture\.jsonl$/,
  /^mixture_\d+_\d+\.jsonl$/,
  /^mixture_filtered\.jsonl$/,
  /^difficult_advice_pool\.jsonl$/,
  /^tulu3_replay\.jsonl$/,
  /^data\/dialogues\.jsonl$/,
  /^stage_6_final\.jsonl$/,
];

/** Small published statistics files, in the order they are trusted. */
const STATS_FILES = ["mixture_stats.json", "stats.json", "dataset_stats.json"];

function pickDataFile(files, declared) {
  if (declared) {
    const match = files.find((file) => file.path === declared);
    return match || null;
  }
  for (const pattern of DATA_FILE_PATTERNS) {
    const match = files.find((file) => pattern.test(file.path));
    if (match) return match;
  }
  return null;
}

/**
 * Record count and composition from a published statistics sidecar.
 *
 * Every value here is read from the file; nothing is estimated. A corpus with
 * no sidecar gets `record_count: 0`, which the viewer renders as unknown - a
 * guessed count on a page whose whole job is provenance would be worse than no
 * count at all.
 */
function statsFromSidecar(json) {
  if (!json || typeof json !== "object") return null;
  const total = json.total && typeof json.total === "object" ? json.total : json;
  const count = Number(total.examples ?? total.rows ?? total.record_count ?? 0);
  const categories = {};
  const bySource = json.by_source && typeof json.by_source === "object" ? json.by_source : {};
  for (const [name, value] of Object.entries(bySource)) {
    const examples = typeof value === "object" ? Number(value?.examples ?? 0) : Number(value);
    if (Number.isFinite(examples) && examples > 0) categories[name] = examples;
  }
  if (!count && !Object.keys(categories).length) return null;
  return { record_count: Number.isFinite(count) ? count : 0, categories };
}

/** Bytes fetched per page. ~20-90 records for the record sizes in this corpus. */
const STREAM_WINDOW = 256 * 1024;

/**
 * Returns `{ dataset }` on success and `{ reason }` on failure.
 *
 * The reason is carried back rather than only warned, because it is what the
 * reader sees. Falling back to the manifest fetch's error made every one of
 * these entries blame a missing `manifest.json`, when the actual situation was
 * usually "this repo holds adapters and logs, not a conversation corpus" - an
 * accurate-sounding error pointing at the wrong thing.
 */
async function hfStreamDataset(source, label, commit) {
  const info = await fetchRepoInfo(source.repo_id, source.revision);
  // The build authenticates and the browser does not, so a private repo would
  // resolve here and 401 for every visitor. Refuse it at the build instead of
  // shipping a reader that only works on the developer's machine.
  if (info.ok && info.private) {
    const reason =
      `${source.repo_id} is private, and the browser reads the Hub without ` +
      `credentials, so its records cannot be shown here`;
    warn(`${label}: ${reason}`);
    return { reason };
  }

  const listing = await fetchRepoListing(source.repo_id, source.revision);
  if (!listing.ok) {
    warn(`${label}: could not list ${source.repo_id}: ${listing.error}`);
    return { reason: listing.error };
  }

  const dataFile = pickDataFile(listing.files, source.data_file);
  if (!dataFile) {
    const candidates = listing.files
      .filter((file) => file.path.endsWith(".jsonl"))
      .map((file) => file.path)
      .slice(0, 6);
    const reason = candidates.length
      ? `${source.repo_id} publishes no recognised conversation file; it holds ` +
        `${candidates.join(", ")}, none of which is a dialogue corpus`
      : `${source.repo_id} publishes no JSONL records to browse`;
    warn(
      `${label}: ${reason}` +
        (candidates.length ? "; declare one as hf_source.data_file to override" : ""),
    );
    return { reason };
  }

  let stats = null;
  for (const name of STATS_FILES) {
    if (!listing.files.some((file) => file.path === name)) continue;
    const loaded = await fetchRepoJson(source.repo_id, source.revision, name);
    stats = loaded.ok ? statsFromSidecar(loaded.json) : null;
    if (stats) {
      stats.from = name;
      break;
    }
  }

  return {
    dataset: {
    source: {
      kind: "hf",
      repo_id: source.repo_id,
      revision: source.revision,
      commit: commit || listing.commit,
      url: repoUrl(source.repo_id),
    },
    source_file: resolveUrl(source.repo_id, source.revision, dataFile.path),
    format: "jsonl",
    record_count: stats?.record_count || 0,
    chunk_size: 0,
    chunks: [],
    stream: {
      url: resolveUrl(source.repo_id, source.revision, dataFile.path),
      total_bytes: Number(dataFile.size || 0),
      window: STREAM_WINDOW,
      path: dataFile.path,
    },
    stats: {
      average_turns: 0,
      role_counts: {},
      splits: {},
      categories: stats?.categories || {},
      ...(stats?.from ? { categories_source: stats.from } : {}),
    },
    // Nothing is downloaded at build time, so this costs the bundle nothing.
    deferred_bytes: 0,
    },
  };
}

// ---------------------------------------------------------------------------
// Index
// ---------------------------------------------------------------------------

const markdownFiles = (await walk(contentRoot)).filter((file) => file.endsWith(".md"));
const entries = [];
const seenIds = new Set();
const seenSlugs = new Map();
let deferredTotal = 0;

if (offline()) console.log("HF_OFFLINE is set: using only cached Hugging Face metadata.");

for (const file of markdownFiles) {
  const type = contentTypeFor(file);
  if (!type) continue;

  const raw = await fs.readFile(file, "utf8");
  const parsed = matter(raw);
  const entryDirectory = path.dirname(file);
  const directorySlug = path.basename(entryDirectory);
  const slug = slugify(parsed.data.slug || directorySlug || path.basename(file, ".md"));
  const id = String(parsed.data.id || `${type}:${slug}`);
  if (seenIds.has(id)) throw new Error(`Duplicate content id: ${id}`);
  seenIds.add(id);
  // The id is type-scoped, so two entries of DIFFERENT types could share a slug
  // and pass the check above - while the body sidecar is `bodies/<slug>.md` and
  // `/entry/<slug>` resolves by slug alone. That happened: an auto-generated
  // eval stub collided with a hand-written dataset entry of the same name, and
  // whichever indexed last silently overwrote the other's body. Nothing failed;
  // the page just showed one entry's prose under the other's metadata.
  if (seenSlugs.has(slug)) {
    throw new Error(
      `Duplicate slug: ${slug} (${seenSlugs.get(slug)} and ${type}). Slugs must be ` +
        "unique across types - the body sidecar and the /entry route both key on them.",
    );
  }
  seenSlugs.set(slug, type);

  const source = normalizeHfSource(parsed.data.hf_source);
  const label = `${type}/${slug}`;

  // Fetch the small manifest first: it can also supply entry-level fields for
  // entries whose dataset card is the system of record.
  let hf = null;
  let hfStatus;
  if (source) {
    const loaded = await loadHfManifest(source, label);
    if (loaded.ok) {
      hf = loaded;
      hfStatus = {
        state: loaded.stale ? "stale" : "ok",
        repo_id: source.repo_id,
        revision: source.revision,
        commit: loaded.commit,
        url: repoUrl(source.repo_id),
        cached: loaded.cached,
      };
    } else {
      hfStatus = {
        state: "unavailable",
        repo_id: source.repo_id,
        revision: source.revision,
        url: repoUrl(source.repo_id),
        message: loaded.error,
      };
    }
  }

  const stat = await fs.stat(file);

  // Assets: remote when HF answered and nothing is on disk, local otherwise.
  // Falling back to local files is what makes the migration additive - an entry
  // that has both keeps rendering when the Hub is unreachable.
  const localFiles = (await walk(entryDirectory)).filter((f) => !f.endsWith(".md"));
  const remoteAssets = Boolean(hf) && localFiles.length === 0;
  const assets = remoteAssets
    ? await hfAssets(source, label)
    : await copyEntryAssets(entryDirectory, type, slug);
  const assetPrefix = remoteAssets
    ? `${resolveUrl(source.repo_id, source.revision, "")}`
    : `/content-assets/${type}/${slug}/`;

  let dataset;
  let petri;
  if (type === "datasets") {
    dataset = hf ? hfDatasetManifest(source, hf.manifest, hf.commit) : undefined;
    // A `manifest.json` that describes a GENERATION run rather than a published
    // corpus yields no chunks, and so did every repo that was never chunked at
    // all. Both fall through to reading the raw JSONL by byte range.
    if (source && !dataset?.chunks?.length) {
      const streamed = await hfStreamDataset(source, label, hf?.commit);
      if (streamed.dataset) {
        dataset = streamed.dataset;
        // The repo demonstrably resolved, so an "unavailable" status left over
        // from a missing manifest.json would now be a false alarm on the page.
        if (!hf) {
          hfStatus = {
            state: "ok",
            repo_id: source.repo_id,
            revision: source.revision,
            commit: dataset.source.commit,
            url: repoUrl(source.repo_id),
          };
        }
      } else if (!hf && hfStatus) {
        // Say why the records are missing, not that a manifest 404'd.
        hfStatus = { ...hfStatus, message: streamed.reason };
      }
    }
    if (!dataset) dataset = (await localDatasetManifest(entryDirectory, slug)) || undefined;
    if (!dataset && source) {
      warn(`${label}: no dataset records available from ${source.repo_id} or on disk`);
    }
  }
  if (type === "petri-runs") {
    petri = hf ? hfPetriManifest(source, hf.manifest, hf.commit) : undefined;
    if (!petri) {
      // Either no HF source, or HF was declared but unreachable. Fall back to
      // whatever is on disk so the page still renders, and say so.
      petri = (await localPetriManifest(entryDirectory, slug)) || undefined;
      if (petri && source) {
        petri.source = { ...petri.source, fallback_from: source.repo_id };
        warn(`${label}: using on-disk copy because ${source.repo_id} is unavailable`);
      }
    }
  }
  deferredTotal += Number(dataset?.deferred_bytes || 0) + Number(petri?.deferred_bytes || 0);

  const card = hf?.manifest || {};
  const rawDate = parsed.data.date || card.date_generated || stat.mtime;
  const date =
    rawDate instanceof Date
      ? rawDate.toISOString().slice(0, 10)
      : String(rawDate).slice(0, 10);
  const title = String(
    parsed.data.title || titleFromMarkdown(parsed.content, path.basename(file, ".md")),
  );

  await writeBody(slug, rewriteAssetLinks(parsed.content, assetPrefix));

  entries.push({
    ...normalize(parsed.data),
    id,
    slug,
    type,
    title,
    date,
    summary: String(parsed.data.summary || card.experiment || ""),
    status: String(parsed.data.status || "unknown"),
    tags: Array.isArray(parsed.data.tags) ? parsed.data.tags.map(String) : [],
    models: Array.isArray(parsed.data.models)
      ? parsed.data.models.map(String)
      : parsed.data.model_id
        ? [String(parsed.data.model_id)]
        : [],
    metrics:
      parsed.data.metrics && typeof parsed.data.metrics === "object"
        ? normalize(parsed.data.metrics)
        : {},
    // `body` is deliberately NOT here. It lives in a per-slug sidecar under
    // lib/generated/bodies/, because the index is imported by every page and
    // only two server components ever render a body. See writeBody below.
    body_bytes: Buffer.byteLength(rewriteAssetLinks(parsed.content, assetPrefix)),
    source_path: toPosix(path.relative(projectRoot, file)),
    assets,
    ...(hfStatus ? { hf: hfStatus } : {}),
    ...(dataset ? { dataset } : {}),
    ...(petri ? { petri } : {}),
  });
}

entries.sort((a, b) => b.date.localeCompare(a.date) || a.title.localeCompare(b.title));

await fs.mkdir(outputDirectory, { recursive: true });
const index = {
  generated: true,
  generated_at: new Date().toISOString(),
  entry_count: entries.length,
  hf: {
    endpoint: process.env.HF_ENDPOINT || "https://huggingface.co",
    token_present: tokenPresent(),
    offline: offline(),
    notices,
  },
  entries,
};
await fs.writeFile(outputFile, `${JSON.stringify(index, null, 2)}\n`, "utf8");

const bakedBytes = bytes(index);
console.log(
  `Indexed ${entries.length} research entries. ` +
    `Baked ${(bakedBytes / 1024).toFixed(1)} KB; ` +
    `deferred ${(deferredTotal / 1024).toFixed(1)} KB to lazy sidecars.`,
);
if (notices.length > 0) {
  console.log(
    `${notices.length} Hugging Face notice(s); the build continued with the ` +
      "content that was reachable.",
  );
}
