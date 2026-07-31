import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render(pathname = "/") {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  const env = {
      ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) },
  };
  const context = { waitUntil() {}, passThroughOnException() {} };
  let url = new URL(pathname, "http://localhost");

  for (let redirects = 0; redirects < 3; redirects += 1) {
    const response = await worker.fetch(
      new Request(url, { headers: { accept: "text/html" } }),
      env,
      context,
    );
    const location = response.headers.get("location");
    if (response.status < 300 || response.status >= 400 || !location) {
      return response;
    }
    url = new URL(location, url);
  }

  throw new Error(`Too many redirects while rendering ${pathname}`);
}

test("server-renders the research overview", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /Synthetic Finetuning/);
  assert.match(html, /for Constitution/);
  assert.match(html, /Research Log/);
  assert.match(html, /Research surfaces/);
  assert.match(html, /Synthetic datasets/);
  assert.match(html, /Petri audits/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape/);
});

test("server-renders a Markdown research entry", async () => {
  const response = await render("/entry/2026-07-29-msm-ood-vulnerability-findings");
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /does model-spec midtraining introduce/i);
  assert.match(html, /Structured metrics/);
  // Prose from the body sidecar, which is no longer in the baked index: this is
  // what catches a broken body read at prerender time.
  assert.match(html, /Holm-Bonferroni/);
  // A real entry must not be labelled as a fixture.
  assert.doesNotMatch(html, /mock-banner/);
});

test("server-renders the JSONL dialogue inspector", async () => {
  const response = await render("/datasets");
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /Synthetic datasets/);
  assert.match(html, /Approved-constitution SFT corpus/);
  assert.match(html, /Conversation preview/);
  // The page must name a lazily-fetched chunk source, but not care which
  // backend it is: a locally sharded path or a Hugging Face resolve URL.
  assert.match(html, /generated-datasets|huggingface\.co\/datasets\/[^/]+\/[^/]+\/resolve\//);
  // The corpus is real, so the page must not be carrying a fixture warning.
  assert.doesNotMatch(html, /mock-banner/);
});

test("the dialogue corpus is genuinely multi-turn and filterable", async () => {
  // Asserted from the baked manifest rather than by reading records: the corpus
  // lives on the Hub now, and a test must not depend on the network. These are
  // the stats the build computed from the real records at publish time.
  const indexUrl = new URL("../lib/generated/content-index.json", import.meta.url);
  const index = JSON.parse(await readFile(indexUrl, "utf8"));
  const datasets = index.entries.filter((entry) => entry.type === "datasets");
  assert.ok(datasets.length > 0, "expected at least one dataset in the corpus");

  for (const entry of datasets) {
    const { stats, record_count, chunks } = entry.dataset;
    assert.ok(record_count > 0, `${entry.slug} has no records`);
    assert.ok(chunks.length > 0, `${entry.slug} has no chunks to page through`);

    // Genuine dialogue, not prompt/response pairs flattened into two turns.
    assert.ok(
      stats.average_turns > 2,
      `${entry.slug} averages ${stats.average_turns} turns; expected real multi-turn dialogue`,
    );
    assert.ok(stats.role_counts.user > 0, `${entry.slug} has no user turns`);
    assert.ok(stats.role_counts.assistant > 0, `${entry.slug} has no assistant turns`);

    // The browser's filters are only useful if the corpus actually varies along
    // them. An all-"uncategorized" corpus means the field mapping was not set.
    assert.ok(
      Object.keys(stats.categories).length > 1,
      `${entry.slug} has one category; the publisher's category_field is unset`,
    );
  }
});

test("server-renders the Petri audit dossier", async () => {
  const response = await render("/petri");
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /Petri audit/);
  assert.match(html, /Findings by hypothesis/);
  assert.match(html, /Transcript explorer/);
  assert.match(html, /Scenario seeds/);
});

test("server-renders the model lineage index", async () => {
  const response = await render("/models");
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /Model lineages/);
  assert.match(html, /linked records/);
  // The real target checkpoint, not the fixture's placeholder id.
  assert.match(html, /chloeli\/qwen-3-32b-philosophy-spec-msm-aft-cot/);
  // The corpus mixes a real model family with fixture-only ones, so the page
  // must both warn and mark which dossiers are fixtures.
  assert.match(html, /mock-banner/);
  assert.match(html, /mock-badge/);
});
