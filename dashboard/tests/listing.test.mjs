// Guarantees about what the listing pages say, asserted from the baked index
// and the rendered HTML.
//
// These are all regressions of the same shape: a page rendering a placeholder,
// a missing value or an incomparable number as though it were a result.

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const ROUTES = ["/", "/logs", "/evals", "/models", "/findings", "/petri", "/datasets", "/glossary"];

async function render(pathname = "/") {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  const env = { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } };
  const context = { waitUntil() {}, passThroughOnException() {} };
  let url = new URL(pathname, "http://localhost");
  for (let redirects = 0; redirects < 3; redirects += 1) {
    const response = await worker.fetch(
      new Request(url, { headers: { accept: "text/html" } }),
      env,
      context,
    );
    const location = response.headers.get("location");
    if (response.status < 300 || response.status >= 400 || !location) return response;
    url = new URL(location, url);
  }
  throw new Error(`Too many redirects while rendering ${pathname}`);
}

/** Visible page text, without scripts, styles or attribute values. */
function bodyText(html) {
  const main = /<main[\s\S]*?<\/main>/.exec(html)?.[0] ?? html;
  return main
    .replace(/<script[\s\S]*?<\/script>/g, "")
    .replace(/<style[\s\S]*?<\/style>/g, "")
    .replace(/<[^>]+>/g, " ")
    .replace(/&[a-z]+;/g, " ");
}

test("no page shows a missing value to the reader", async () => {
  // /evals printed "mmlu · undefined · 1 runs" in its group picker and a bare
  // "undefined" chip beside the suite name, because a run declaring an
  // eval_suite but no eval_version had those slots interpolated anyway.
  for (const route of ROUTES) {
    const text = bodyText(await (await render(route)).text());
    for (const token of ["undefined", "NaN", "[object Object]", "null"]) {
      assert.ok(
        !text.includes(token),
        `${route} renders the literal text "${token}" to the reader`,
      );
    }
  }
});

test("the eval table does not fill itself with em-dashes", async () => {
  // A column was promoted for being reported by more than one run, which on a
  // corpus of six unrelated instruments meant columns 26 of 30 runs could not
  // fill. Half the table was an em-dash, reading as "these runs failed to
  // report" rather than "these runs measure different things".
  const html = await (await render("/evals")).text();
  const cells = html.match(/<td[^>]*>(?:(?!<\/td>)[\s\S])*<\/td>/g) || [];
  assert.ok(cells.length > 0, "expected the eval index to render cells");
  const dashes = cells.filter((cell) => bodyText(cell).trim() === "—").length;
  assert.ok(
    dashes / cells.length < 0.2,
    `${dashes} of ${cells.length} eval table cells are em-dashes; a column no ` +
      "run can fill is not a comparison",
  );
});

test("a one-run group is not offered as a comparison", async () => {
  // The metric explorer's guard fired only on ZERO compatible groups, so a
  // single run declaring an eval_suite produced a "stage comparison" chart of
  // one point with nothing to compare it against.
  const { evalFamily, entryKind, entryMix, byKindThenDate } = await import("../lib/entries.ts");
  assert.equal(typeof evalFamily, "function");

  const indexUrl = new URL("../lib/generated/content-index.json", import.meta.url);
  const index = JSON.parse(await readFile(indexUrl, "utf8"));
  const evals = index.entries.filter((entry) => entry.type === "evals");

  const groups = new Map();
  for (const entry of evals) {
    if (!entry.eval_suite) continue;
    const key = [entry.eval_suite, entry.eval_version ?? "", entry.dataset_version ?? ""].join("|");
    groups.set(key, (groups.get(key) || 0) + 1);
  }
  const comparable = [...groups.values()].filter((count) => count > 1).length;
  const html = await (await render("/evals")).text();
  const plots = html.includes("metric-explorer");
  assert.equal(
    plots,
    comparable > 0,
    plots
      ? "a comparison chart is rendered but no two runs share a suite/version/dataset"
      : "runs share a suite/version/dataset but no comparison is offered",
  );

  // The kind helpers the listings sort and count by.
  assert.equal(entryKind({ status: "stub", tags: [] }), "stub");
  assert.equal(entryKind({ status: "complete", tags: ["auto-indexed"] }), "auto");
  assert.equal(entryKind({ status: "final", tags: ["swe-bench"] }), "written");
  const mix = entryMix([
    { status: "stub", tags: [] },
    { status: "complete", tags: ["auto-indexed"] },
    { status: "final", tags: [] },
  ]);
  assert.deepEqual(mix, { written: 1, auto: 1, stub: 1, total: 3 });
  // Written work sorts ahead of transcribed metrics, which sort ahead of links.
  const sorted = [
    { status: "stub", tags: [], date: "2026-08-09" },
    { status: "final", tags: [], date: "2026-08-01" },
    { status: "complete", tags: ["auto-indexed"], date: "2026-08-05" },
  ].sort(byKindThenDate);
  assert.deepEqual(sorted.map(entryKind), ["written", "auto", "stub"]);
});

test("every eval run is filed under an instrument", async () => {
  // A flat list put an MMLU accuracy directly above a Petri flag count. The
  // grouping is only useful if it actually classifies - a large "Other" bucket
  // is the same flat list with extra steps.
  const { evalFamily } = await import("../lib/entries.ts");
  const indexUrl = new URL("../lib/generated/content-index.json", import.meta.url);
  const index = JSON.parse(await readFile(indexUrl, "utf8"));
  const evals = index.entries.filter((entry) => entry.type === "evals");
  const unclassified = evals.filter((entry) => evalFamily(entry) === "Other");
  assert.ok(
    unclassified.length <= evals.length * 0.15,
    `${unclassified.length} of ${evals.length} runs fall through to "Other": ` +
      unclassified.map((entry) => entry.slug).join(", "),
  );
});

test("each listing says how much of it a human wrote", async () => {
  // "30 evaluation runs" reads as thirty results when 15 are a link and a
  // title. The mix line is the page being honest about its own contents.
  for (const route of ["/evals", "/logs", "/findings"]) {
    const html = await (await render(route)).text();
    assert.match(
      html,
      /corpus-mix/,
      `${route} lists entries without stating how many are written up`,
    );
  }
});
