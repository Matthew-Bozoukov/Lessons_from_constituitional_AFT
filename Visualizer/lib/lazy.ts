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

export function sidecarUrl(base: string, file: string) {
  return `${String(base).replace(/\/+$/, "")}/${String(file).replace(/^\/+/, "")}`;
}

async function loadJson<T>(url: string): Promise<T> {
  const existing = cache.get(url);
  if (existing) return existing as Promise<T>;

  const pending = (async () => {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} loading ${url}`);
    }
    return response.json();
  })();

  // A failed load must not poison the cache: the reader should be able to retry
  // by selecting the item again.
  pending.catch(() => cache.delete(url));
  cache.set(url, pending);
  return pending as Promise<T>;
}

/** Fetch one Petri transcript body on demand. */
export function loadTranscript(base: string, file: string) {
  return loadJson<PetriTranscript>(sidecarUrl(base, file));
}

/** Fetch one dataset chunk on demand. Chunk entries are already full URLs. */
export function loadChunk<T>(url: string) {
  return loadJson<T[]>(url);
}

/** Human-readable failure text, with no internals leaked. */
export function describeLoadError(error: unknown, source: string) {
  const detail = error instanceof Error ? error.message : String(error);
  return `Could not load this item from ${source}. ${detail}`;
}
