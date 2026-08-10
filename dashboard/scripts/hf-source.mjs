// Build-time Hugging Face client for the research log.
//
// Only ever fetches SMALL metadata: a dataset's `manifest.json` and its file
// listing. Bulk payloads (transcript bodies, per-sample rows) are never pulled
// at build time - the browser fetches those straight from the HF CDN on demand,
// so they cost the build nothing and never enter the static bundle.
//
// Three properties this module must hold, because the Netlify deploy depends on
// them:
//   1. It never throws. A missing dataset, a dead network, a 401 - all degrade
//      to a described failure the caller can render.
//   2. It caches on disk, keyed by revision, so repeat builds do no network work
//      and CI does not trip HF rate limits.
//   3. It never logs the token.

import { promises as fs } from "node:fs";
import { createHash } from "node:crypto";
import path from "node:path";
import { projectRoot } from "./content-utils.mjs";

export const HF_ENDPOINT =
  process.env.HF_ENDPOINT?.replace(/\/+$/, "") || "https://huggingface.co";

export const cacheRoot = path.join(projectRoot, ".hf-cache");

/** Milliseconds a cached response for a floating revision stays fresh. */
const DEFAULT_TTL_MS = Number(process.env.HF_CACHE_TTL_SECONDS || 3600) * 1000;

/** Per-request timeout. A slow Hub must not hang a deploy. */
const TIMEOUT_MS = Number(process.env.HF_TIMEOUT_MS || 20000);

/**
 * Read the token from the environment without ever returning it to a caller
 * that might log it. Only `authHeaders` sees the value.
 */
function token() {
  return (
    process.env.HF_TOKEN ||
    process.env.HUGGING_FACE_HUB_TOKEN ||
    process.env.HUGGINGFACEHUB_API_TOKEN ||
    ""
  );
}

export function tokenPresent() {
  return Boolean(token());
}

function authHeaders() {
  const value = token();
  return value ? { authorization: `Bearer ${value}` } : {};
}

/** True when the build should do no network at all. */
export function offline() {
  return /^(1|true|yes)$/i.test(process.env.HF_OFFLINE || "");
}

/**
 * Redact anything token-shaped before a message reaches a log. Belt and
 * braces: we never interpolate the token into a message, but an HTTP client
 * error string could echo a request header.
 */
export function redact(message) {
  const value = token();
  let text = String(message ?? "");
  if (value) text = text.split(value).join("<redacted>");
  return text.replace(/\bhf_[A-Za-z0-9]{8,}\b/g, "hf_<redacted>");
}

/** A revision that names an immutable commit needs no revalidation. */
function isPinned(revision) {
  return /^[0-9a-f]{7,40}$/i.test(revision || "");
}

function cachePath(repoId, revision, filePath) {
  const key = createHash("sha256")
    .update(`${HF_ENDPOINT}\0${repoId}\0${revision}\0${filePath}`)
    .digest("hex")
    .slice(0, 32);
  return path.join(cacheRoot, `${key}.json`);
}

async function readCache(file) {
  try {
    return JSON.parse(await fs.readFile(file, "utf8"));
  } catch {
    return null;
  }
}

async function writeCache(file, payload) {
  try {
    await fs.mkdir(path.dirname(file), { recursive: true });
    await fs.writeFile(file, JSON.stringify(payload), "utf8");
  } catch {
    // A read-only or full cache directory must not fail a build.
  }
}

/**
 * Public URL a browser can fetch a repo file from. HF sets
 * `access-control-allow-origin` on these, so client-side lazy loads work from
 * the deployed origin without a proxy.
 */
export function resolveUrl(repoId, revision, filePath) {
  const rev = encodeURIComponent(revision || "main");
  const clean = String(filePath).replace(/^\/+/, "");
  return `${HF_ENDPOINT}/datasets/${repoId}/resolve/${rev}/${clean}`;
}

async function request(url, extraHeaders = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    return await fetch(url, {
      headers: { ...authHeaders(), ...extraHeaders },
      signal: controller.signal,
      redirect: "follow",
    });
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Fetch one small text file from a dataset repo, through the disk cache.
 *
 * Returns `{ ok, text, revision, commit, cached, error }`. Never throws and
 * never rejects: a caller can always keep building.
 */
export async function fetchRepoFile(repoId, revision, filePath) {
  const rev = revision || "main";
  const file = cachePath(repoId, rev, filePath);
  const cached = await readCache(file);
  const pinned = isPinned(rev);

  // A pinned commit can never change, so a hit is authoritative forever.
  if (cached?.ok && pinned) {
    return { ...cached, cached: true };
  }
  const fresh =
    cached?.ok && Date.now() - (cached.fetched_at || 0) < DEFAULT_TTL_MS;
  if (cached?.ok && (fresh || offline())) {
    return { ...cached, cached: true };
  }
  if (offline()) {
    return {
      ok: false,
      cached: false,
      error: `HF_OFFLINE is set and ${repoId}:${filePath} is not cached`,
    };
  }

  const url = resolveUrl(repoId, rev, filePath);
  try {
    const response = await request(
      url,
      cached?.etag ? { "if-none-match": cached.etag } : {},
    );

    if (response.status === 304 && cached?.ok) {
      const refreshed = { ...cached, fetched_at: Date.now() };
      await writeCache(file, refreshed);
      return { ...refreshed, cached: true };
    }
    if (!response.ok) {
      if (cached?.ok) return { ...cached, cached: true, stale: true };
      const hint =
        response.status === 401 || response.status === 403
          ? " (private dataset? set HF_TOKEN in the environment)"
          : "";
      return {
        ok: false,
        cached: false,
        error: `HTTP ${response.status} for ${repoId}:${filePath}${hint}`,
      };
    }

    const payload = {
      ok: true,
      text: await response.text(),
      etag: response.headers.get("etag") || "",
      commit: response.headers.get("x-repo-commit") || "",
      revision: rev,
      fetched_at: Date.now(),
    };
    await writeCache(file, payload);
    return { ...payload, cached: false };
  } catch (error) {
    if (cached?.ok) return { ...cached, cached: true, stale: true };
    return {
      ok: false,
      cached: false,
      error: redact(`${error?.name || "Error"}: ${error?.message || error}`),
    };
  }
}

/** Fetch and parse a small JSON file from a dataset repo. */
export async function fetchRepoJson(repoId, revision, filePath) {
  const result = await fetchRepoFile(repoId, revision, filePath);
  if (!result.ok) return result;
  try {
    return { ...result, json: JSON.parse(result.text) };
  } catch (error) {
    return {
      ok: false,
      cached: result.cached,
      error: redact(`${repoId}:${filePath} is not valid JSON: ${error.message}`),
    };
  }
}

/**
 * Pages the tree API will follow before giving up. The Hub caps a page at 1000
 * entries, so this bounds a listing at 20k. A repo deeper than that gets a
 * truncated list and says so - a build must not walk an unbounded tree.
 */
const MAX_TREE_PAGES = Number(process.env.HF_TREE_MAX_PAGES || 20);

/**
 * The `rel="next"` target of a `Link` header, or `null`.
 *
 * Only a URL on the configured endpoint is followed. The header is server-
 * controlled, and a listing loop is not the place to chase an arbitrary host.
 */
export function nextPageUrl(response) {
  const header = response.headers.get("link");
  if (!header) return null;
  for (const part of header.split(",")) {
    const match = /^\s*<([^>]+)>\s*;\s*rel="?next"?/.exec(part);
    if (!match) continue;
    return match[1].startsWith(`${HF_ENDPOINT}/`) ? match[1] : null;
  }
  return null;
}

/**
 * Describe a dataset repo without downloading it: file list with sizes, and the
 * commit the revision currently points at.
 *
 * The tree API caps a response at 1000 entries and hands back the rest through
 * a `Link: rel="next"` header, so this follows that chain. Directories count
 * against the cap: a repo with a deep tree can fill an entire page with nothing
 * but directory entries, which is how a single-request version of this returned
 * an empty file list for a repo that was full of files.
 */
export async function fetchRepoListing(repoId, revision) {
  const rev = revision || "main";
  const file = cachePath(repoId, rev, "::tree");
  const cached = await readCache(file);
  const pinned = isPinned(rev);
  if (cached?.ok && (pinned || offline() || Date.now() - (cached.fetched_at || 0) < DEFAULT_TTL_MS)) {
    return { ...cached, cached: true };
  }
  if (offline()) {
    return { ok: false, error: `HF_OFFLINE is set and ${repoId} tree is not cached` };
  }

  let url = `${HF_ENDPOINT}/api/datasets/${repoId}/tree/${encodeURIComponent(rev)}?recursive=1`;
  try {
    const files = [];
    let commit = "";
    let pages = 0;
    let truncated = false;

    while (url) {
      const response = await request(url);
      if (!response.ok) {
        if (cached?.ok) return { ...cached, cached: true, stale: true };
        return { ok: false, error: `HTTP ${response.status} listing ${repoId}` };
      }
      if (!commit) commit = response.headers.get("x-repo-commit") || "";
      const entries = await response.json();
      for (const entry of Array.isArray(entries) ? entries : []) {
        if (entry.type !== "file") continue;
        files.push({
          path: entry.path,
          size: Number(entry.size ?? entry?.lfs?.size ?? 0),
        });
      }
      pages += 1;
      const next = nextPageUrl(response);
      if (next && pages >= MAX_TREE_PAGES) {
        truncated = true;
        break;
      }
      url = next;
    }

    const payload = {
      ok: true,
      files,
      commit,
      revision: rev,
      pages,
      truncated,
      fetched_at: Date.now(),
    };
    await writeCache(file, payload);
    return { ...payload, cached: false };
  } catch (error) {
    if (cached?.ok) return { ...cached, cached: true, stale: true };
    return { ok: false, error: redact(`${error?.name}: ${error?.message}`) };
  }
}

/**
 * Repo-level flags: is it private, is it gated.
 *
 * This matters for one reason. The build authenticates; the browser does not.
 * A private repo therefore resolves perfectly at build time and 401s for every
 * visitor, which surfaces as a viewer that is permanently "loading" on the
 * deployed site and works fine on the developer's machine. Reading the flag
 * lets the build refuse to wire up a reader it knows the public cannot use.
 */
export async function fetchRepoInfo(repoId, revision) {
  const rev = revision || "main";
  const file = cachePath(repoId, rev, "::info");
  const cached = await readCache(file);
  if (cached?.ok && (offline() || Date.now() - (cached.fetched_at || 0) < DEFAULT_TTL_MS)) {
    return { ...cached, cached: true };
  }
  if (offline()) {
    return { ok: false, error: `HF_OFFLINE is set and ${repoId} info is not cached` };
  }
  try {
    const response = await request(`${HF_ENDPOINT}/api/datasets/${repoId}`);
    if (!response.ok) {
      if (cached?.ok) return { ...cached, cached: true, stale: true };
      return { ok: false, error: `HTTP ${response.status} describing ${repoId}` };
    }
    const info = await response.json();
    const payload = {
      ok: true,
      private: Boolean(info?.private),
      gated: Boolean(info?.gated),
      fetched_at: Date.now(),
    };
    await writeCache(file, payload);
    return { ...payload, cached: false };
  } catch (error) {
    if (cached?.ok) return { ...cached, cached: true, stale: true };
    return { ok: false, error: redact(`${error?.name}: ${error?.message}`) };
  }
}

/** Human-readable link to the dataset page, for provenance in the UI. */
export function repoUrl(repoId) {
  return `${HF_ENDPOINT}/datasets/${repoId}`;
}
