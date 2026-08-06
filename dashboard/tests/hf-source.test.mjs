// Exercises the build-time Hugging Face client against a local stand-in for the
// Hub, plus the payload guarantee the whole design rests on: no transcript body
// may ever reach the baked index.

import assert from "node:assert/strict";
import { createServer } from "node:http";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import test, { after, before } from "node:test";
import { fileURLToPath } from "node:url";

let server;
let endpoint;
let requests = [];
let etagCounter = 0;

/** Files the fake Hub serves, keyed by `<repo>/<path>`. */
const files = new Map([
  [
    "acme/2026-07-29-demo-run/manifest.json",
    JSON.stringify({
      manifest_version: 1,
      kind: "petri-run",
      experiment: "Demo",
      date_generated: "2026-07-29",
      scenarios: [{ id: "seed-1", category: "demo", hypothesis: "h", expected_signal: "s" }],
      scores: { retained_transcripts: 1 },
      transcripts: [
        {
          id: "audit-001",
          file: "audit-001.json",
          scenario_id: "seed-1",
          category: "demo",
          outcome: "pass",
          scores: { concerning_behavior: 0 },
          tags: ["demo"],
          message_count: 2,
          size_bytes: 128,
        },
      ],
    }),
  ],
  ["acme/2026-07-29-demo-run/not-json.json", "{ this is not json"],
]);

/** A repo the stand-in serves across two tree pages. */
const deepTreeRepo = "acme/2026-07-30-deep-tree";

before(async () => {
  server = createServer((request, response) => {
    requests.push(request.url);
    const url = new URL(request.url, "http://localhost");

    const tree = url.pathname.match(/^\/api\/datasets\/(.+?)\/tree\/(.+)$/);
    if (tree) {
      const repo = tree[1];

      // A repo whose tree does not fit one page. Page 1 is nothing but
      // directory entries, which is exactly how the Hub answers a deeply
      // nested repo and exactly what a single-request client gets wrong.
      if (repo === deepTreeRepo) {
        const cursor = url.searchParams.get("cursor");
        if (!cursor) {
          const directories = Array.from({ length: 3 }, (_, index) => ({
            type: "directory",
            path: `group-${index}`,
            size: 0,
          }));
          response.writeHead(200, {
            "content-type": "application/json",
            "x-repo-commit": "deadbeefcafe",
            link: `<${endpoint}${url.pathname}?recursive=true&cursor=page2>; rel="next"`,
          });
          response.end(JSON.stringify(directories));
          return;
        }
        response.writeHead(200, { "content-type": "application/json" });
        response.end(
          JSON.stringify([
            { type: "file", path: "group-0/results.jsonl", size: 4096 },
            { type: "file", path: "group-1/results.jsonl", size: 2048 },
          ]),
        );
        return;
      }

      const listing = [...files.keys()]
        .filter((key) => key.startsWith(`${repo}/`))
        .map((key) => ({ type: "file", path: key.slice(repo.length + 1), size: 10 }));
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify(listing));
      return;
    }

    const resolve = url.pathname.match(/^\/datasets\/(.+?)\/resolve\/(.+?)\/(.+)$/);
    if (resolve) {
      const key = `${resolve[1]}/${resolve[3]}`;
      const body = files.get(key);
      if (body === undefined) {
        response.writeHead(404).end("not found");
        return;
      }
      const etag = `"etag-${etagCounter}"`;
      if (request.headers["if-none-match"] === etag) {
        response.writeHead(304, { etag }).end();
        return;
      }
      response.writeHead(200, { etag, "x-repo-commit": "deadbeefcafe" });
      response.end(body);
      return;
    }
    response.writeHead(404).end("not found");
  });

  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  endpoint = `http://127.0.0.1:${server.address().port}`;
  process.env.HF_ENDPOINT = endpoint;
  // Isolate the cache so a developer's real cache cannot mask a failure.
  process.env.HF_CACHE_TTL_SECONDS = "0";
});

after(() => server?.close());

async function freshModule() {
  // A fresh module instance per test: the cache TTL and offline flag are read
  // at call time, but the cache directory is resolved at import.
  const url = new URL("../scripts/hf-source.mjs", import.meta.url);
  url.searchParams.set("t", `${process.pid}-${Math.random()}`);
  return import(url.href);
}

test("resolveUrl builds a browser-fetchable Hub URL", async () => {
  const hf = await freshModule();
  assert.equal(
    hf.resolveUrl("acme/demo", "main", "transcripts/a.json"),
    `${endpoint}/datasets/acme/demo/resolve/main/transcripts/a.json`,
  );
});

test("fetches a small manifest and reports the commit", async () => {
  const hf = await freshModule();
  const result = await hf.fetchRepoJson(
    "acme/2026-07-29-demo-run",
    "main",
    "manifest.json",
  );
  assert.equal(result.ok, true);
  assert.equal(result.commit, "deadbeefcafe");
  assert.equal(result.json.transcripts[0].id, "audit-001");
});

test("a missing dataset degrades to a described failure, never a throw", async () => {
  const hf = await freshModule();
  const result = await hf.fetchRepoFile("acme/does-not-exist", "main", "manifest.json");
  assert.equal(result.ok, false);
  assert.match(result.error, /HTTP 404/);
});

test("an unreachable Hub degrades to a described failure", async () => {
  const url = new URL("../scripts/hf-source.mjs", import.meta.url);
  url.searchParams.set("t", `unreachable-${Math.random()}`);
  const previous = process.env.HF_ENDPOINT;
  // A port nothing is listening on: the connection is refused immediately.
  process.env.HF_ENDPOINT = "http://127.0.0.1:1";
  try {
    const hf = await import(url.href);
    const result = await hf.fetchRepoFile("acme/whatever", "main", "manifest.json");
    assert.equal(result.ok, false);
    assert.ok(result.error.length > 0);
  } finally {
    process.env.HF_ENDPOINT = previous;
  }
});

test("invalid JSON is reported rather than crashing the build", async () => {
  const hf = await freshModule();
  const result = await hf.fetchRepoJson(
    "acme/2026-07-29-demo-run",
    "main",
    "not-json.json",
  );
  assert.equal(result.ok, false);
  assert.match(result.error, /not valid JSON/);
});

test("a pinned revision is served from cache without a second request", async () => {
  const hf = await freshModule();
  const pinned = "deadbeefcafe";
  requests = [];
  const first = await hf.fetchRepoFile("acme/2026-07-29-demo-run", pinned, "manifest.json");
  assert.equal(first.ok, true);
  assert.equal(first.cached, false);
  const countAfterFirst = requests.length;

  const second = await hf.fetchRepoFile("acme/2026-07-29-demo-run", pinned, "manifest.json");
  assert.equal(second.ok, true);
  assert.equal(second.cached, true);
  assert.equal(
    requests.length,
    countAfterFirst,
    "a pinned commit must never be revalidated",
  );
});

test("a floating revision revalidates and accepts a 304", async () => {
  const hf = await freshModule();
  await hf.fetchRepoFile("acme/2026-07-29-demo-run", "main", "manifest.json");
  requests = [];
  const again = await hf.fetchRepoFile("acme/2026-07-29-demo-run", "main", "manifest.json");
  assert.equal(again.ok, true);
  assert.ok(requests.length > 0, "a floating revision should revalidate");
  assert.equal(again.cached, true, "a 304 should serve the cached body");
});

test("HF_OFFLINE never touches the network", async () => {
  const hf = await freshModule();
  // Warm the cache first, then go offline.
  await hf.fetchRepoFile("acme/2026-07-29-demo-run", "main", "manifest.json");
  process.env.HF_OFFLINE = "1";
  try {
    requests = [];
    const cached = await hf.fetchRepoFile(
      "acme/2026-07-29-demo-run",
      "main",
      "manifest.json",
    );
    assert.equal(cached.ok, true);
    assert.equal(requests.length, 0);

    const uncached = await hf.fetchRepoFile("acme/never-seen", "main", "manifest.json");
    assert.equal(uncached.ok, false);
    assert.match(uncached.error, /HF_OFFLINE/);
    assert.equal(requests.length, 0);
  } finally {
    delete process.env.HF_OFFLINE;
  }
});

test("redact strips the token from any message", async () => {
  process.env.HF_TOKEN = "hf_thisIsASecretTokenValue123";
  try {
    const hf = await freshModule();
    assert.equal(hf.tokenPresent(), true);
    const message = hf.redact(
      "failed with authorization: Bearer hf_thisIsASecretTokenValue123",
    );
    assert.doesNotMatch(message, /thisIsASecret/);
    assert.match(message, /redacted/);
  } finally {
    delete process.env.HF_TOKEN;
  }
});

test("listing a repo returns file paths and sizes", async () => {
  const hf = await freshModule();
  const listing = await hf.fetchRepoListing("acme/2026-07-29-demo-run", "main");
  assert.equal(listing.ok, true);
  assert.ok(listing.files.some((file) => file.path === "manifest.json"));
});

test("a tree spanning several pages is followed to the end", async () => {
  const hf = await freshModule();
  const listing = await hf.fetchRepoListing(deepTreeRepo, "main");
  assert.equal(listing.ok, true);
  assert.equal(listing.truncated, false);
  assert.equal(listing.pages, 2);
  // Page 1 held only directories. A client that stopped there would report a
  // repo full of files as having none.
  assert.deepEqual(
    listing.files.map((file) => file.path),
    ["group-0/results.jsonl", "group-1/results.jsonl"],
  );
  assert.equal(listing.commit, "deadbeefcafe");
});

test("paging stops at the page cap and reports the listing as truncated", async () => {
  const previous = process.env.HF_TREE_MAX_PAGES;
  process.env.HF_TREE_MAX_PAGES = "1";
  try {
    const hf = await freshModule();
    const listing = await hf.fetchRepoListing(deepTreeRepo, "main");
    assert.equal(listing.ok, true);
    assert.equal(listing.pages, 1);
    assert.equal(listing.truncated, true);
    assert.equal(listing.files.length, 0);
  } finally {
    if (previous === undefined) delete process.env.HF_TREE_MAX_PAGES;
    else process.env.HF_TREE_MAX_PAGES = previous;
  }
});

test("a next link pointing off-endpoint is not followed", async () => {
  const hf = await freshModule();
  const response = {
    headers: {
      get: (name) =>
        name === "link" ? '<http://elsewhere.invalid/api/next>; rel="next"' : null,
    },
  };
  assert.equal(hf.nextPageUrl(response), null);
});

// ---------------------------------------------------------------------------
// The payload guarantee
// ---------------------------------------------------------------------------

test("the baked index carries transcript summaries but no message bodies", async () => {
  const indexUrl = new URL("../lib/generated/content-index.json", import.meta.url);
  const index = JSON.parse(await fs.readFile(indexUrl, "utf8"));
  const runs = index.entries.filter((entry) => entry.type === "petri-runs");
  assert.ok(runs.length > 0, "expected at least one Petri run in the corpus");

  for (const run of runs) {
    assert.ok(run.petri, `${run.slug} has no petri manifest`);
    assert.ok(Array.isArray(run.petri.transcript_index));
    assert.ok(run.petri.transcript_base, "a transcript base URL is required");
    assert.equal(
      run.petri.transcript_index.length,
      run.petri.transcript_count,
      "every transcript needs an index row",
    );
    for (const row of run.petri.transcript_index) {
      assert.ok(row.id && row.file, "each row needs an id and a sidecar file name");
      assert.equal(
        row.messages,
        undefined,
        `${run.slug}/${row.id} leaked message bodies into the baked index`,
      );
      assert.equal(
        row.judge_summary,
        undefined,
        `${run.slug}/${row.id} leaked a judge summary into the baked index`,
      );
    }
  }

  // The whole point of the split: the index must stay small enough to bake.
  const size = Buffer.byteLength(JSON.stringify(index));
  assert.ok(
    size < 300 * 1024,
    `content index is ${(size / 1024).toFixed(1)} KB; the build-time budget is 300 KB`,
  );
});

test("the baked index carries no entry prose, only a sidecar per entry", async () => {
  const indexUrl = new URL("../lib/generated/content-index.json", import.meta.url);
  const index = JSON.parse(await fs.readFile(indexUrl, "utf8"));
  const bodyRoot = new URL("../lib/generated/bodies/", import.meta.url);
  assert.ok(index.entries.length > 0, "expected a non-empty corpus");

  for (const entry of index.entries) {
    // `lib/content.ts` is imported by every page, so a body here is a body on
    // every page. The real write-ups run to hundreds of lines; inlining them
    // took the index to 96% of its budget before this split.
    assert.equal(
      entry.body,
      undefined,
      `${entry.slug} leaked its Markdown body into the baked index`,
    );
    assert.equal(
      typeof entry.body_bytes,
      "number",
      `${entry.slug} must record its body size`,
    );

    // A summary is what listings render, so it must survive the split.
    assert.equal(typeof entry.summary, "string");

    if (entry.body_bytes > 0) {
      const sidecar = new URL(`${entry.slug}.md`, bodyRoot);
      const text = await fs.readFile(sidecar, "utf8");
      assert.equal(
        Buffer.byteLength(text),
        entry.body_bytes,
        `${entry.slug}: sidecar size disagrees with the index`,
      );
    }
  }
});

test("every locally sharded transcript is fetchable and complete", async () => {
  const indexUrl = new URL("../lib/generated/content-index.json", import.meta.url);
  const index = JSON.parse(await fs.readFile(indexUrl, "utf8"));
  const publicRoot = new URL("../public/", import.meta.url);

  for (const run of index.entries.filter((entry) => entry.petri)) {
    if (run.petri.source.kind !== "local") continue;
    for (const row of run.petri.transcript_index) {
      // fileURLToPath, not `.pathname`: the latter stays percent-encoded, so a
      // checkout under a directory with a space in its name resolves to a path
      // containing a literal "%20" and every read fails with ENOENT. It also
      // handles the Windows drive-letter prefix that used to be stripped here
      // by hand.
      const file = path.join(
        fileURLToPath(publicRoot),
        run.petri.transcript_base.replace(/^\//, ""),
        row.file,
      );
      const record = JSON.parse(await fs.readFile(file, "utf8"));
      assert.equal(record.id, row.id);
      assert.equal(record.messages.length, row.message_count);
      assert.ok(typeof record.judge_summary === "string");
    }
  }
});

test("the cache directory stays inside the project and holds no token", async () => {
  const previous = process.env.HF_TOKEN;
  // A distinctive value, so its absence from the cache is a real assertion
  // rather than a coincidence of the fixture's wording.
  process.env.HF_TOKEN = "hf_TESTONLYtoken1234567890";
  try {
    const hf = await freshModule();
    assert.ok(hf.cacheRoot.includes(".hf-cache"));
    assert.ok(!hf.cacheRoot.startsWith(os.tmpdir()));
    await hf.fetchRepoJson("acme/2026-07-29-demo-run", "main", "manifest.json");

    const entries = await fs.readdir(hf.cacheRoot).catch(() => []);
    assert.ok(entries.length > 0, "expected the fetch above to write a cache entry");
    for (const name of entries) {
      const body = await fs.readFile(path.join(hf.cacheRoot, name), "utf8");
      // The token itself, in any form.
      assert.ok(!body.includes(process.env.HF_TOKEN), `${name} contains the token`);
      // Credential SHAPES, not the bare words: cached payloads are research
      // prose that legitimately discusses authorization, so matching the word
      // would fail on content rather than on a leak.
      assert.doesNotMatch(body, /hf_[A-Za-z0-9]{8}/);
      assert.doesNotMatch(body, /"authorization"\s*:/i);
      assert.doesNotMatch(body, /bearer\s+\S/i);
    }
  } finally {
    if (previous === undefined) delete process.env.HF_TOKEN;
    else process.env.HF_TOKEN = previous;
  }
});
