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

test("server-renders the datasets page as a live Hub explorer", async () => {
  // Like /evals, /datasets lists nothing at build time: the corpora are
  // discovered in the browser from the org's `training-data` card tags. The
  // static HTML is the frame and the explorer's listing state, and it must
  // carry no baked corpus - a baked list is exactly the drift this replaced.
  const response = await render("/datasets");
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /Synthetic datasets/);
  assert.match(html, /discovered live from Hugging Face/);
  assert.match(html, /Listing training-data repos on Hugging Face/);
  assert.doesNotMatch(html, /corpus-picker|Conversation preview/);
  // No corpus is on the page yet, so no fixture warning can be either.
  assert.doesNotMatch(html, /mock-banner/);
});

test("a fabricated fixture is flagged as one", async () => {
  // This is the failure the mock banner exists to prevent, and it happened:
  // `2026-07-30-visualizer-mock-dialogues` is eleven hand-written dialogues
  // that exist to exercise the dataset browser, and its entry was regenerated
  // as a stub WITHOUT the flag - so it rendered on /datasets as a real corpus,
  // unbadged, alongside the actual training data.
  //
  // The rule is asserted from the repo id rather than from a list of slugs, so
  // a future fixture cannot be added without either the flag or a deliberate
  // decision to change this test.
  const indexUrl = new URL("../lib/generated/content-index.json", import.meta.url);
  const index = JSON.parse(await readFile(indexUrl, "utf8"));
  const fixtures = index.entries.filter((entry) =>
    /visualizer-mock/i.test(entry.hf_source?.repo_id || entry.slug || ""),
  );
  assert.ok(fixtures.length > 0, "expected the interface fixtures to be in the corpus");
  for (const entry of fixtures) {
    assert.equal(
      entry.mock,
      true,
      `${entry.slug} is an interface fixture but is not flagged \`mock: true\`, so it renders as real research data`,
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
