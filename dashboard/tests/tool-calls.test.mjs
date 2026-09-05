// Tool use in the interchange rows the research repo publishes: calls carry the OpenAI
// shape with structured arguments, and the row's `tools` schemas are conversation, not a
// facet. These pin what the transcript viewer shows for both the new and the old shape.

import assert from "node:assert/strict";
import test from "node:test";

import { normalizeRecord, toolCallView } from "../lib/records.ts";

test("an interchange tool call renders as its function name over pretty-printed arguments", () => {
  const view = toolCallView({ type: "function", function: { name: "bash", arguments: { command: "ls -la /app" } } });
  assert.equal(view.name, "bash");
  assert.equal(view.arguments, JSON.stringify({ command: "ls -la /app" }, null, 2));
});

test("a flat legacy call with string arguments still renders", () => {
  assert.deepEqual(toolCallView({ name: "search", arguments: '{"q": "x"}' }), { name: "search", arguments: '{"q": "x"}' });
  assert.deepEqual(toolCallView("garbage"), { name: "tool", arguments: "" });
});

test("a row's tool schemas stay out of the metadata facets", () => {
  const row = normalizeRecord({
    messages: [{ role: "user", content: "go" }, { role: "assistant", content: "done" }],
    tools: [{ type: "function", function: { name: "bash" } }],
    metadata: { trait_id: "t3" },
  }, 0);
  assert.equal(row.metadata.trait_id, "t3");
  assert.equal("tools" in row.metadata, false);
});
