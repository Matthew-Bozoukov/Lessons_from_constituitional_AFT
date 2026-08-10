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

test("the datasets page leads with what each corpus is made of", async () => {
  const response = await render("/datasets");
  assert.equal(response.status, 200);
  const html = await response.text();

  // The picker, not a 43-option dropdown.
  assert.match(html, /corpus-picker/);
  assert.match(html, /Constitution mixtures/);
  // A real, measured blend on the corpus that opens first.
  assert.match(html, /constitution-grounded/);
  assert.match(html, /composition-table/);
  // Identity is never colour alone: both halves of the bar are named.
  assert.match(html, /General instruction data/);
  // A real corpus must not be carrying a fixture warning because a fixture
  // exists elsewhere in the picker.
  assert.doesNotMatch(html, /mock-banner/);
});

test("every indexed dataset can actually be paged through", async () => {
  // Asserted from the baked manifest rather than by reading records: the corpora
  // live on the Hub, and a test must not depend on the network.
  //
  // There are two paging modes and the guarantee differs between them, so this
  // does not average over both. A CHUNKED corpus was pre-chunked by a publisher
  // that read every record, so the build knows its turn counts and categories
  // and those numbers must be real. A STREAMED corpus is a raw JSONL read by
  // byte range - deliberately never downloaded at build time, so the build
  // knows its size and its published statistics and nothing else. Demanding
  // turn counts from a streamed corpus would only be satisfiable by inventing
  // them or by pulling 300 MB through every build.
  const indexUrl = new URL("../lib/generated/content-index.json", import.meta.url);
  const index = JSON.parse(await readFile(indexUrl, "utf8"));
  const datasets = index.entries.filter((entry) => entry.type === "datasets");
  assert.ok(datasets.length > 0, "expected at least one dataset in the corpus");

  const withData = datasets.filter((entry) => entry.dataset);
  assert.ok(
    withData.length >= 40,
    `only ${withData.length} of ${datasets.length} datasets resolved a reader; ` +
      "the published SFT corpora are meant to be browsable",
  );

  let chunked = 0;
  let streamed = 0;
  for (const entry of withData) {
    const { stats, record_count, chunks, stream } = entry.dataset;
    assert.ok(
      chunks.length > 0 || stream,
      `${entry.slug} has neither chunks nor a stream, so nothing can page it`,
    );

    if (stream) {
      streamed += 1;
      assert.match(
        stream.url,
        /^https:\/\/huggingface\.co\/datasets\/[^/]+\/[^/]+\/resolve\//,
        `${entry.slug} must stream from a public Hub resolve URL`,
      );
      assert.match(stream.path, /\.jsonl$/, `${entry.slug} must stream a JSONL`);
      assert.ok(stream.total_bytes > 0, `${entry.slug} has no file size to page against`);
      assert.ok(stream.window > 0, `${entry.slug} has no window size`);
      // A count is stated only when a published statistics sidecar gives one.
      // Zero means unknown and the viewer says so; a fabricated number here
      // would read as measured on a page whose whole job is provenance.
      assert.ok(record_count >= 0, `${entry.slug} has a negative record count`);
      for (const [name, count] of Object.entries(stats.categories)) {
        assert.ok(count > 0, `${entry.slug} declares source ${name} with ${count} records`);
      }
      if (Object.keys(stats.categories).length > 0) {
        assert.ok(
          stats.categories_source,
          `${entry.slug} has categories but does not say which file they came from`,
        );
      }
      continue;
    }

    chunked += 1;
    assert.ok(record_count > 0, `${entry.slug} has no records`);
    // Genuine dialogue, not prompt/response pairs flattened into two turns.
    assert.ok(
      stats.average_turns > 2,
      `${entry.slug} averages ${stats.average_turns} turns; expected real multi-turn dialogue`,
    );
    assert.ok(stats.role_counts.user > 0, `${entry.slug} has no user turns`);
    assert.ok(stats.role_counts.assistant > 0, `${entry.slug} has no assistant turns`);
    assert.ok(
      Object.keys(stats.categories).length > 1,
      `${entry.slug} has one category; the publisher's category_field is unset`,
    );
  }

  assert.ok(chunked > 0, "expected at least one pre-chunked corpus to still resolve");
  assert.ok(streamed > 0, "expected the byte-range reader to resolve the raw JSONL corpora");
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
