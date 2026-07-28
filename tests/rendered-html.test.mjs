import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render(pathname = "/") {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request(`http://localhost${pathname}`, {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) },
    },
    { waitUntil() {}, passThroughOnException() {} },
  );
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
  const response = await render("/entry/2026-07-20-sft-reasons-seed-1");
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /Reasons-rich SFT safety battery/);
  assert.match(html, /Structured metrics/);
  assert.match(html, /Evaluation shifts/);
  assert.match(html, /content-assets/);
});

test("server-renders the JSONL dialogue inspector", async () => {
  const response = await render("/datasets");
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /Synthetic datasets/);
  assert.match(html, /Reasons-rich constitutional dialogue mixture/);
  assert.match(html, /Conversation preview/);
  assert.match(html, /generated-datasets/);
});

test("sample JSONL includes genuine multi-turn conversations", async () => {
  const chunkUrl = new URL(
    "../public/generated-datasets/reasons-rich-aft-v1/chunk-000.json",
    import.meta.url,
  );
  const records = JSON.parse(await readFile(chunkUrl, "utf8"));
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
  assert.match(html, /qwen3-32b/);
  assert.match(html, /linked records/);
  assert.match(html, /Sft/);
});
