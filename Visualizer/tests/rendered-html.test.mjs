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
  assert.match(html, /Reasons-rich constitutional dialogue mixture/);
  assert.match(html, /Conversation preview/);
  // The page must name a lazily-fetched chunk source, but not care which
  // backend it is: a locally sharded path or a Hugging Face resolve URL.
  assert.match(html, /generated-datasets|huggingface\.co\/datasets\/[^/]+\/[^/]+\/resolve\//);
});

test("sample JSONL includes genuine multi-turn conversations", async () => {
  // The fixture's source JSONL in `content/`, not a build artifact under
  // `public/`: now that this dataset is served from the Hub, no local chunk is
  // regenerated, and asserting against a stale one would test nothing.
  const sourceUrl = new URL(
    "../content/datasets/reasons-rich-aft-v1/data/dialogues.jsonl",
    import.meta.url,
  );
  const records = (await readFile(sourceUrl, "utf8"))
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
  const multiTurnRecords = records.filter(
    (record) =>
      record.messages.filter((message) => message.role === "user").length >= 2 &&
      record.messages.filter((message) => message.role === "assistant").length >= 2,
  );

  assert.ok(
    multiTurnRecords.length >= 3,
    `expected at least 3 multi-turn records, found ${multiTurnRecords.length}`,
  );
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
