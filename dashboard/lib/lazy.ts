"use client";

// Runtime loader for the payloads the build deliberately did NOT bake in.
//
// A sidecar is addressed as `${base}/${file}`, where `base` is either a path on
// this site (locally sharded content) or a Hugging Face `resolve` URL (published
// datasets). HF sets permissive CORS on those, so the same fetch works for both
// and no proxy or token is needed for public datasets.

import type { PetriTranscript } from "./content";

/** In-memory cache, so re-selecting a transcript is instant and free. */
const cache = new Map<string, Promise<unknown>>();

/**
 * Run `make` once per key and share the promise.
 *
 * A failed load must not poison the cache: the reader should be able to retry
 * by selecting the item again, so a rejected promise is evicted. Shared by the
 * sidecar loaders here and the Hub listings in evalRuns / trainingData.
 */
export function cached<T>(key: string, make: () => Promise<T>): Promise<T> {
  const existing = cache.get(key);
  if (existing) return existing as Promise<T>;
  const pending = make();
  pending.catch(() => cache.delete(key));
  cache.set(key, pending);
  return pending;
}

export function sidecarUrl(base: string, file: string) {
  return `${String(base).replace(/\/+$/, "")}/${String(file).replace(/^\/+/, "")}`;
}

function loadJson<T>(url: string): Promise<T> {
  return cached(url, async () => {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} loading ${url}`);
    }
    return response.json() as Promise<T>;
  });
}

/** Fetch one Petri transcript body on demand. */
export function loadTranscript(base: string, file: string) {
  return loadJson<PetriTranscript>(sidecarUrl(base, file));
}

/** Fetch a single JSON document (an object, not a chunk array) on demand. */
export function loadJsonDoc<T>(url: string) {
  return loadJson<T>(url);
}

// ---------------------------------------------------------------------------
// Streaming a raw JSONL by byte range
// ---------------------------------------------------------------------------
//
// Pre-chunking a corpus into `chunks/chunk-NNN.json` requires a publish step,
// and almost none of the training mixtures on the Hub have had one. They do not
// need it: JSONL is line-delimited, and the Hub serves `resolve` URLs with
// `accept-ranges: bytes`, `access-control-allow-origin: *` and
// `access-control-expose-headers: *` (verified against the live Hub, exercised
// in tests/jsonl-window.test.mjs). So a browser can page a 28 MB mixture by
// asking for a window of bytes and keeping the whole lines inside it.
//
// The byte accounting has to be exact, because the next request starts where
// this one stopped. It is done on the raw bytes rather than on decoded text:
// slicing at the last newline BYTE means a window can never split a multi-byte
// character, which slicing decoded text would do at every non-ASCII boundary.

export type JsonlWindow = {
  records: unknown[];
  /** Byte offset the next window should start at. */
  nextOffset: number;
  /** Size of the whole file, from `content-range`. */
  totalBytes: number;
  done: boolean;
  /** Bytes fetched, including the trailing partial line that was not used. */
  fetchedBytes: number;
};

/** Ceiling on growing the window for one oversized record. */
const MAX_WINDOW = 8 * 1024 * 1024;

function parseTotal(response: Response): number {
  const range = response.headers.get("content-range");
  const total = range ? Number(range.split("/").pop()) : NaN;
  if (Number.isFinite(total) && total > 0) return total;
  const length = Number(response.headers.get("content-length"));
  return Number.isFinite(length) ? length : 0;
}

/**
 * Read whole JSONL records from `[offset, offset + windowBytes)`.
 *
 * A record longer than the window would otherwise return nothing and stall the
 * pager forever, so the window doubles until the record fits or `MAX_WINDOW` is
 * reached. A line that does not parse is skipped rather than thrown: one
 * malformed record must not blank the viewer.
 */
export async function loadJsonlWindow(
  url: string,
  offset: number,
  windowBytes: number,
): Promise<JsonlWindow> {
  let size = Math.max(1024, windowBytes);
  for (;;) {
    const response = await fetch(url, {
      headers: { range: `bytes=${offset}-${offset + size - 1}` },
    });
    if (!response.ok) throw new Error(`HTTP ${response.status} loading ${url}`);

    const totalBytes = parseTotal(response);
    const bytes = new Uint8Array(await response.arrayBuffer());
    // A server that ignores `Range` answers 200 with the whole file. Honour it
    // rather than re-slicing: the data is correct, just larger than asked for.
    const atEof =
      response.status !== 206 || (totalBytes > 0 && offset + bytes.length >= totalBytes);

    let consumed = bytes.length;
    if (!atEof) {
      consumed = bytes.lastIndexOf(0x0a) + 1;
      if (consumed === 0) {
        // No line ended inside the window: one record is bigger than it.
        if (size >= MAX_WINDOW) {
          throw new Error(
            `A single record at byte ${offset} exceeds ${MAX_WINDOW} bytes and cannot be paged.`,
          );
        }
        size = Math.min(size * 2, MAX_WINDOW);
        continue;
      }
    }

    const text = new TextDecoder().decode(bytes.subarray(0, consumed));
    const records: unknown[] = [];
    for (const line of text.split("\n")) {
      if (!line.trim()) continue;
      try {
        records.push(JSON.parse(line));
      } catch {
        // Skip: a truncated or malformed line is not worth losing the page over.
      }
    }

    const nextOffset = offset + consumed;
    return {
      records,
      nextOffset,
      totalBytes,
      done: totalBytes > 0 ? nextOffset >= totalBytes : bytes.length === 0,
      fetchedBytes: bytes.length,
    };
  }
}

/**
 * Fetch ONE known JSONL record by its exact byte range.
 *
 * `loadJsonlWindow` pages forward through a file when you do not know where the
 * rows are. When you do - because a published sidecar indexes `id -> [offset,
 * length]` - a single exact range turns a 24 MB corpus into an ~8 KB request,
 * which is what makes per-row inspection viable in the browser.
 *
 * A server that ignores `Range` answers 200 with the whole file; that is treated
 * as a failure here rather than parsed, because slicing a full-file response by
 * the requested offsets would appear to work while the bytes were never the ones
 * asked for on a file that had since changed.
 */
export async function loadJsonlRow<T>(url: string, offset: number, length: number): Promise<T> {
  const key = `${url}#${offset}+${length}`;
  const existing = cache.get(key);
  if (existing) return existing as Promise<T>;

  const pending = (async () => {
    const response = await fetch(url, {
      headers: { range: `bytes=${offset}-${offset + length - 1}` },
    });
    if (!response.ok) throw new Error(`HTTP ${response.status} loading ${url}`);
    if (response.status !== 206) {
      throw new Error(`range request was not honoured (HTTP ${response.status})`);
    }
    return JSON.parse(new TextDecoder().decode(await response.arrayBuffer())) as T;
  })();

  pending.catch(() => cache.delete(key));
  cache.set(key, pending);
  return pending;
}

/** Fetch a whole JSONL and parse every line. For sidecars small enough to hold. */
export async function loadJsonlAll<T>(url: string): Promise<T[]> {
  const existing = cache.get(url);
  if (existing) return existing as Promise<T[]>;

  const pending = (async () => {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status} loading ${url}`);
    const text = await response.text();
    return text.split("\n").filter((line) => line.trim()).map((line) => JSON.parse(line) as T);
  })();

  pending.catch(() => cache.delete(url));
  cache.set(url, pending);
  return pending as Promise<T[]>;
}

/** Human-readable failure text, with no internals leaked. */
export function describeLoadError(error: unknown, source: string) {
  const detail = error instanceof Error ? error.message : String(error);
  return `Could not load this item from ${source}. ${detail}`;
}
