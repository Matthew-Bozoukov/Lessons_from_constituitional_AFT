import assert from "node:assert/strict";
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
  assert.match(html, /Metric explorer/);
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
