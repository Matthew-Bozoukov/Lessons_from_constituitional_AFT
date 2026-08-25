"use client";

// ABOUTME: Client-side access to the org's training-data repos on Hugging Face: tag-filtered
// ABOUTME: discovery, data-file resolution (card configs, else an allowlist), stats sidecars.
//
// The dataset twin of lib/evalRuns.ts. Every corpus publisher in the research repo —
// synth's StageCache, mix's push, properties/ablate — stamps its card with
// `training_data_tags` (src/huggingface.py):
//
//   training-data  kind:<synth|mixture|ablation|fixture>  pipeline:<name>
//   constitution:<slug>  [stage:<unfiltered|filtered|final>]  [smoke]  [mock]
//
// and names the rows file as the default entry of the card's `configs:` block. So one
// listing call — /api/datasets?author=<org>&filter=training-data with cardData expanded —
// yields every corpus AND the file to stream, with no build step and no content entry to
// keep in sync. Only PUBLIC repos are visible: the site is token-less by design
// (netlify.toml). Untagged legacy repos are invisible until their cards are backfilled
// (scratch/backfill_training_data_tags.py in the research repo).

import type { DatasetManifest } from "./content";
// Explicit extension: tests/training-data.test.mjs loads this module under Node's own
// ESM resolver, which does not guess extensions; the bundler resolves it the same way.
import { cached, loadJsonDoc } from "./lazy.ts";

const ENDPOINT = "https://huggingface.co";
export const TRAINING_DATA_ORG = "LASR-Callum";
export const TRAINING_DATA_TAG = "training-data";
/** Bytes fetched per page. ~20-90 records for the record sizes in this corpus. */
export const STREAM_WINDOW = 256 * 1024;

type Json = Record<string, unknown>;

export interface TrainingDataRepo {
  repo: string;
  /** `kind:` facet — which publisher produced the rows. */
  kind: string;
  /** `pipeline:` facet — the synth document type, mixture config, or ablation tag. */
  pipeline: string;
  /** `constitution:` facet — the constitution slug, or `none`. */
  constitution: string;
  /** `stage:` facet on mixture pushes (unfiltered / filtered / final), else "". */
  stage: string;
  smoke: boolean;
  /** Interface fixture: hand-written rows that exist to exercise the viewer. */
  mock: boolean;
  tags: string[];
  createdAt: string;
  lastModified: string;
  /** Repo-relative rows file from the card's default config, or "" when undeclared. */
  dataFile: string;
}

export type TreeFile = { path: string; size: number };

export type SidecarStats = {
  record_count: number;
  categories: Record<string, number>;
  /** Which sidecar the numbers came from. */
  from: string;
};

/** Value of a `key:value` facet tag, or "" when the card carries none. */
export function facet(tags: string[], key: string): string {
  const hit = tags.find((t) => t.startsWith(`${key}:`));
  return hit ? hit.slice(key.length + 1) : "";
}

/** Config names that denote THE corpus when no `default: true` is set. */
const CORPUS_CONFIG_NAMES = new Set(["default", "dataset", "train"]);

/**
 * The rows file a card's `configs:` block declares.
 *
 * The `default: true` config wins, failing that one named `default`/`dataset`/`train`.
 * A card whose only configs are stage snapshots (a synth run that has not published
 * its `dataset.jsonl` yet) declares no corpus, and must not be read as its first
 * stage. `data_files` is a string in the publishers' cards and a `[{split, path}]`
 * list in cards the Hub UI or older tooling wrote, so both shapes are read. A glob
 * cannot be streamed by byte range, so it counts as undeclared too.
 */
export function dataFileFromConfigs(cardData: unknown): string {
  const configs = (cardData as Json | null | undefined)?.configs;
  if (!Array.isArray(configs) || configs.length === 0) return "";
  const chosen = (configs.find((c) => (c as Json)?.default === true) ||
    configs.find((c) => CORPUS_CONFIG_NAMES.has(String((c as Json)?.config_name)))) as Json | undefined;
  let files: unknown = chosen?.data_files;
  if (Array.isArray(files)) files = files[0];
  if (files && typeof files === "object") files = (files as Json).path;
  const file = typeof files === "string" ? files.trim() : "";
  return file && !/[*?[]/.test(file) ? file : "";
}

/**
 * Files that hold a conversation corpus, best first — the fallback for a repo whose
 * card declares no default config (legacy pushes, hand uploads).
 *
 * An allowlist, not "the biggest .jsonl in the repo": these repos also hold
 * `verdicts.jsonl`, `assistant_spans.jsonl` and per-question eval records, and a
 * conversation viewer pointed at those renders garbage while looking like it worked.
 * `mixture_think` outranks `mixture`: same records, with the reasoning traces.
 */
export const DATA_FILE_PATTERNS: readonly RegExp[] = [
  /^dataset\.jsonl$/,
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

/** Root-level stage snapshot that IS the export: `stage_8_export_sft.jsonl`, `stage_6_final.jsonl`. */
const STAGE_EXPORT = /^stage_(\d+)_[\w-]*?(sft|final)[\w-]*\.jsonl$/;
/** JSONL files that are never the rows: sidecars beside a corpus, and eval-record dumps. */
const SIDECAR = /(verdict|label|scan|rating|score|summar|attribute|span|^records\.jsonl$)/i;

/**
 * The file to stream, in order of confidence:
 *
 * 1. the declared file, when the repo actually holds it — a declared file that is
 *    missing (a synth run whose `dataset.jsonl` is not published yet) falls through;
 * 2. the first allowlist match;
 * 3. the highest-numbered root stage export (the pre-contract synth layouts kept
 *    every stage at the root, and the last `*sft*`/`*final*` one is the corpus);
 * 4. a lone root JSONL that is not a sidecar (hand-pushed mixtures named after
 *    their arm, e.g. `t2_9284_da716_10k.jsonl`) — one file leaves nothing to guess;
 * 5. "".
 */
export function pickDataFile(paths: string[], declared = ""): string {
  if (declared && paths.includes(declared)) return declared;
  for (const pattern of DATA_FILE_PATTERNS) {
    const hit = paths.find((p) => pattern.test(p));
    if (hit) return hit;
  }
  const stages = paths
    .map((p) => ({ p, m: STAGE_EXPORT.exec(p) }))
    .filter((x) => x.m)
    .sort((a, b) => Number(b.m![1]) - Number(a.m![1]));
  if (stages.length) return stages[0].p;
  const lone = paths.filter((p) => p.endsWith(".jsonl") && !p.includes("/") && !SIDECAR.test(p));
  return lone.length === 1 ? lone[0] : "";
}

/** Small published statistics files, in the order they are trusted. */
export const STATS_FILES = ["mixture_stats.json", "stats.json"];

/** The sidecars to try for a corpus: the shared names, then `<rows file>.stats.json`. */
export function statsCandidates(file = ""): string[] {
  return file ? [...STATS_FILES, `${file}.stats.json`] : [...STATS_FILES];
}

/**
 * Record count and composition from a published statistics sidecar.
 *
 * Two schemas are real. `uv run mix` writes `{total: {examples}, by_source: {name:
 * {examples}}}`; the hand-pushed arm mixtures write `{total: 9987, per_source: {name:
 * count}}`. Every value here is read from the file; nothing is estimated. A corpus
 * with no sidecar gets no count, which the viewer renders as unknown — a guessed
 * count on a page whose whole job is provenance would be worse than none.
 */
export function statsFromSidecar(json: unknown): Omit<SidecarStats, "from"> | null {
  if (!json || typeof json !== "object") return null;
  const doc = json as Json;
  const total = (doc.total && typeof doc.total === "object" ? doc.total : doc) as Json;
  const count = Number(
    typeof doc.total === "number" ? doc.total : total.examples ?? total.rows ?? total.record_count ?? 0,
  );
  const categories: Record<string, number> = {};
  const sources = ((typeof doc.by_source === "object" && doc.by_source) ||
    (typeof doc.per_source === "object" && doc.per_source) || {}) as Json;
  for (const [name, value] of Object.entries(sources)) {
    const examples =
      value && typeof value === "object" ? Number((value as Json).examples ?? 0) : Number(value);
    if (Number.isFinite(examples) && examples > 0) categories[name] = examples;
  }
  if (!count && !Object.keys(categories).length) return null;
  return { record_count: Number.isFinite(count) ? count : 0, categories };
}

/** One Hub listing row, reduced to the facets the explorer works with. */
/**
 * Fold a string for picker search: lower-cased, with the separators repo names,
 * file names and tags use (`-` `_` `.` `/` `:`) collapsed to single spaces.
 */
export function searchText(text: string): string {
  return text.toLowerCase().replace(/[-_./:\s]+/g, " ").trim();
}

/**
 * Whether a corpus matches a picker query.
 *
 * Every whitespace-separated term of the query must occur somewhere in the
 * corpus's fields, in any order, as a substring — so `advice 716`, `716 advice`,
 * `difficult-advice-716` and `difficult_advice` all find
 * `2026-08-14-table2-9284-difficult-advice-716-train`. Each term is tried
 * against the separator-folded text AND the same text with separators removed,
 * so `da716` matches the rows file `t2_9284_da716_10k.jsonl` and
 * `difficultadvice` matches `difficult-advice`.
 */
export function corpusMatches(fields: Array<string | undefined | null>, query: string): boolean {
  const terms = searchText(query).split(" ").filter(Boolean);
  if (!terms.length) return true;
  const spaced = searchText(fields.filter((f): f is string => Boolean(f)).join(" "));
  const joined = spaced.replace(/ /g, "");
  return terms.every((term) => {
    const bare = term.replace(/ /g, "");
    return spaced.includes(term) || joined.includes(bare);
  });
}

export function parseRepo(row: {
  id: string;
  tags?: string[];
  cardData?: unknown;
  createdAt?: string;
  lastModified?: string;
}): TrainingDataRepo {
  const tags = row.tags || [];
  const kind = facet(tags, "kind") || "unknown";
  return {
    repo: row.id,
    kind,
    pipeline: facet(tags, "pipeline"),
    constitution: facet(tags, "constitution"),
    stage: facet(tags, "stage"),
    smoke: tags.includes("smoke") || /-smoke$/.test(row.id),
    mock: tags.includes("mock") || kind === "fixture",
    tags,
    createdAt: row.createdAt || "",
    lastModified: row.lastModified || "",
    dataFile: dataFileFromConfigs(row.cardData),
  };
}

export function repoDate(repo: string): string {
  const m = /(\d{4}-\d{2}-\d{2})/.exec(repo.split("/")[1] || repo);
  return m ? m[1] : "";
}

export function resolveUrl(repo: string, path: string): string {
  return `${ENDPOINT}/datasets/${repo}/resolve/main/${path}`;
}

export function repoUrl(repo: string): string {
  return `${ENDPOINT}/datasets/${repo}`;
}

/** Every public training corpus in the org, newest (by dated name) first. */
export function listTrainingData(org: string = TRAINING_DATA_ORG): Promise<TrainingDataRepo[]> {
  const url =
    `${ENDPOINT}/api/datasets?author=${org}&filter=${TRAINING_DATA_TAG}&limit=500` +
    "&expand[]=cardData&expand[]=tags&expand[]=createdAt&expand[]=lastModified";
  return cached(url, async () => {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status} listing ${org} training-data repos`);
    const rows = (await res.json()) as Parameters<typeof parseRepo>[0][];
    return rows
      .map(parseRepo)
      .sort(
        (a, b) =>
          repoDate(b.repo).localeCompare(repoDate(a.repo)) ||
          b.lastModified.localeCompare(a.lastModified),
      );
  });
}

/** The repo's file tree, following the Hub's `Link: rel="next"` pagination. */
export function listFiles(repo: string): Promise<TreeFile[]> {
  return cached(`tree:${repo}`, async () => {
    const out: TreeFile[] = [];
    let url: string | null = `${ENDPOINT}/api/datasets/${repo}/tree/main?recursive=1`;
    for (let page = 0; url && page < 10; page += 1) {
      const res: Response = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status} listing files of ${repo}`);
      const items = (await res.json()) as Array<{ type: string; path: string; size?: number }>;
      for (const it of items) if (it.type === "file") out.push({ path: it.path, size: it.size || 0 });
      const link = res.headers.get("link") || "";
      const next = /<([^>]+)>;\s*rel="next"/.exec(link);
      url = next && next[1].startsWith(ENDPOINT) ? next[1] : null;
    }
    return out;
  });
}

/**
 * The statistics sidecar, or null when the corpus publishes none.
 *
 * With a file listing only the sidecars that exist are fetched; without one each
 * candidate is probed on the CDN and a 404 is the ordinary outcome for a synth corpus,
 * whose publisher writes a generation manifest rather than mixture statistics.
 */
export function loadStats(repo: string, file = "", files?: TreeFile[]): Promise<SidecarStats | null> {
  return cached(`stats:${repo}`, async () => {
    for (const name of statsCandidates(file)) {
      if (files && !files.some((f) => f.path === name)) continue;
      try {
        const stats = statsFromSidecar(await loadJsonDoc<unknown>(resolveUrl(repo, name)));
        if (stats) return { ...stats, from: name };
      } catch {
        /* not published under this name: try the next */
      }
    }
    return null;
  });
}

function manifestFor(repo: TrainingDataRepo, file: string, size: number, stats: SidecarStats | null): DatasetManifest {
  return {
    source: { kind: "hf", repo_id: repo.repo, url: repoUrl(repo.repo) },
    source_file: resolveUrl(repo.repo, file),
    format: "jsonl",
    record_count: stats?.record_count || 0,
    stream: { url: resolveUrl(repo.repo, file), total_bytes: size, window: STREAM_WINDOW, path: file },
    stats: {
      average_turns: 0,
      role_counts: {},
      splits: {},
      categories: stats?.categories || {},
      ...(stats ? { categories_source: stats.from } : {}),
    },
  };
}

/**
 * A manifest the DatasetViewer can page.
 *
 * A card that declares its rows file costs no API call: the stream URL is known, the
 * size arrives with the first byte-range response, and only a mixture-shaped corpus is
 * probed for a stats sidecar. A card that declares nothing pays one tree listing, so
 * the allowlist can choose and the sidecar probe can be exact. Throws, with the repo's
 * JSONL candidates named, when there is nothing recognisable to browse.
 */
export function corpusManifest(repo: TrainingDataRepo): Promise<DatasetManifest> {
  return cached(`corpus:${repo.repo}`, async () => {
    if (repo.dataFile) {
      const stats = repo.kind === "synth" ? null : await loadStats(repo.repo, repo.dataFile);
      return manifestFor(repo, repo.dataFile, 0, stats);
    }
    const files = await listFiles(repo.repo);
    const paths = files.map((f) => f.path);
    const file = pickDataFile(paths);
    if (!file) {
      const jsonl = paths.filter((p) => p.endsWith(".jsonl")).slice(0, 6);
      throw new Error(
        jsonl.length
          ? `${repo.repo} publishes no recognised conversation file; it holds ${jsonl.join(", ")}`
          : `${repo.repo} publishes no JSONL records to browse`,
      );
    }
    const stats = await loadStats(repo.repo, file, files);
    return manifestFor(repo, file, files.find((f) => f.path === file)?.size || 0, stats);
  });
}
