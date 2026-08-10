// Exercises reading a raw JSONL by byte range, and turning what comes back into
// a conversation.
//
// This is the path that made the corpus visible at all: the training mixtures
// were published as plain `.jsonl` with no chunking step, so the viewer's
// chunked reader resolved nothing for 41 of 49 dataset entries. Byte-range
// paging needs no publish step, which is exactly why its edge cases have to be
// pinned down here - a wrong offset silently drops or duplicates records rather
// than failing.

import assert from "node:assert/strict";
import { createServer } from "node:http";
import test, { after, before } from "node:test";

import { loadJsonlWindow } from "../lib/lazy.ts";
import { normalizeRecord, parseChatTemplate, splitReasoning } from "../lib/records.ts";

// ---------------------------------------------------------------------------
// A stand-in for the Hub that honours Range exactly as huggingface.co does
// ---------------------------------------------------------------------------

/** Records whose content is deliberately non-ASCII, so windows split characters. */
const RECORDS = Array.from({ length: 40 }, (_, index) => ({
  n: index,
  text: `<|im_start|>user\n¿Pregunta número ${index}? — ✅ ${"á".repeat(index)}<|im_end|>\n`,
}));
const BODY = Buffer.from(RECORDS.map((r) => JSON.stringify(r)).join("\n") + "\n", "utf8");

let server;
let base;
let served = [];

before(async () => {
  server = createServer((request, response) => {
    served.push(request.headers.range || "");
    const url = new URL(request.url, "http://localhost");
    const body = url.pathname === "/no-trailing-newline" ? BODY.subarray(0, BODY.length - 1) : BODY;

    if (url.pathname === "/ignores-range") {
      response.writeHead(200, { "content-length": String(body.length) });
      response.end(body);
      return;
    }
    const match = /^bytes=(\d+)-(\d+)$/.exec(request.headers.range || "");
    if (!match) {
      response.writeHead(200, { "content-length": String(body.length) });
      response.end(body);
      return;
    }
    const start = Number(match[1]);
    const end = Math.min(Number(match[2]), body.length - 1);
    if (start >= body.length) {
      response.writeHead(416, { "content-range": `bytes */${body.length}` });
      response.end();
      return;
    }
    const slice = body.subarray(start, end + 1);
    response.writeHead(206, {
      "content-range": `bytes ${start}-${end}/${body.length}`,
      "content-length": String(slice.length),
      "accept-ranges": "bytes",
    });
    response.end(slice);
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  base = `http://127.0.0.1:${server.address().port}`;
});

after(() => server?.close());

// ---------------------------------------------------------------------------
// Byte-range paging
// ---------------------------------------------------------------------------

test("paging a JSONL by byte range yields every record exactly once", async () => {
  served = [];
  const seen = [];
  let offset = 0;
  let guard = 0;
  for (;;) {
    const page = await loadJsonlWindow(`${base}/data.jsonl`, offset, 300);
    seen.push(...page.records);
    assert.equal(page.totalBytes, BODY.length, "the file size comes from content-range");
    assert.ok(page.nextOffset > offset, "each page must advance the cursor");
    offset = page.nextOffset;
    if (page.done) break;
    assert.ok((guard += 1) < 500, "paging failed to terminate");
  }

  // The point of the whole exercise: no record lost, none duplicated, order kept.
  assert.equal(seen.length, RECORDS.length);
  assert.deepEqual(
    seen.map((record) => record.n),
    RECORDS.map((record) => record.n),
  );
  assert.equal(offset, BODY.length, "paging ends exactly at the end of the file");
  assert.ok(served.length > 3, "the window was small enough to force several pages");
});

test("a window that splits a multi-byte character does not corrupt it", async () => {
  // The failure this guards against: slicing DECODED text at the window edge
  // turns the tail of a UTF-8 sequence into U+FFFD and shifts every subsequent
  // byte offset. Slicing raw bytes at the last newline cannot.
  const withAccents = RECORDS.filter((record) => record.text.includes("á"));
  assert.ok(withAccents.length > 5, "fixture must contain multi-byte characters");

  const seen = [];
  let offset = 0;
  for (;;) {
    // 137 is deliberately coprime with the record sizes, so boundaries land
    // mid-character rather than conveniently.
    const page = await loadJsonlWindow(`${base}/data.jsonl`, offset, 137);
    seen.push(...page.records);
    offset = page.nextOffset;
    if (page.done) break;
  }
  assert.deepEqual(seen, RECORDS);
  assert.ok(
    !seen.some((record) => record.text.includes("�")),
    "no record may contain a replacement character",
  );
});

test("a record larger than the window grows the window instead of stalling", async () => {
  // With a 20-byte window no line ever ends inside it, so a reader that simply
  // returned "no whole lines" would hand back an unchanged offset forever and
  // the Load-more button would never finish.
  const page = await loadJsonlWindow(`${base}/data.jsonl`, 0, 20);
  assert.ok(page.records.length > 0, "the window must grow until a record fits");
  assert.ok(page.nextOffset > 0);
});

test("a file with no trailing newline still yields its last record", async () => {
  let offset = 0;
  const seen = [];
  for (;;) {
    const page = await loadJsonlWindow(`${base}/no-trailing-newline`, offset, 400);
    seen.push(...page.records);
    offset = page.nextOffset;
    if (page.done) break;
  }
  assert.equal(seen.length, RECORDS.length, "the unterminated final line is not dropped");
});

test("a server that ignores Range is handled rather than re-requested forever", async () => {
  const page = await loadJsonlWindow(`${base}/ignores-range`, 0, 300);
  assert.equal(page.records.length, RECORDS.length);
  assert.equal(page.done, true, "a 200 response is the whole file, so paging is finished");
});

test("a failed range request surfaces as an error, not as an empty corpus", async () => {
  await assert.rejects(
    () => loadJsonlWindow(`${base}/data.jsonl`, BODY.length + 10, 300),
    /HTTP 416/,
  );
});

// ---------------------------------------------------------------------------
// Turning a training record into a conversation
// ---------------------------------------------------------------------------

test("a rendered chat template is split back into its turns", () => {
  const text =
    "<|im_start|>system\nYou are helpful.<|im_end|>\n" +
    "<|im_start|>user\nHello\nthere<|im_end|>\n" +
    "<|im_start|>assistant\nHi<|im_end|>\n";
  assert.deepEqual(parseChatTemplate(text), [
    { role: "system", content: "You are helpful." },
    { role: "user", content: "Hello\nthere" },
    { role: "assistant", content: "Hi" },
  ]);
});

test("a truncated final turn is kept rather than silently dropped", () => {
  const messages = parseChatTemplate("<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\ncut off");
  assert.equal(messages.length, 2);
  assert.equal(messages[1].content, "cut off");
});

test("text with no chat markers is shown rather than discarded", () => {
  assert.deepEqual(parseChatTemplate("just prose"), [{ role: "text", content: "just prose" }]);
  assert.deepEqual(parseChatTemplate("   "), []);
});

test("an empty think block is distinguished from a real reasoning trace", () => {
  // Qwen3.6 renders the think tag on EVERY assistant turn and leaves it empty
  // where no reasoning is supervised. Both live in the same mixture file, and
  // which records supervise reasoning is a property of the training data worth
  // reading off the page - so they must not collapse into the same thing.
  const empty = splitReasoning({ role: "assistant", content: "<think>\n\n</think>\n\nAnswer" });
  assert.equal(empty.present, true);
  assert.equal(empty.empty, true);
  assert.equal(empty.message.content, "Answer");
  assert.equal(empty.message.reasoning_content, undefined);

  const real = splitReasoning({
    role: "assistant",
    content: "<think>\nWeighing the request.\n</think>\n\nAnswer",
  });
  assert.equal(real.empty, false);
  assert.equal(real.message.reasoning_content, "Weighing the request.");
  assert.equal(real.message.content, "Answer");
});

test("a mixture record is labelled by the component it was drawn from", () => {
  const record = normalizeRecord(
    { text: "<|im_start|>user\nq<|im_end|>\n", source: "synthdoc_difficult_advice", n_tokens: 91 },
    7,
  );
  assert.equal(record.category, "synthdoc_difficult_advice");
  assert.equal(record.category_field, "source", "the filter must know it is showing sources");
  assert.equal(record.id, "#8", "a streamed record is labelled by position");
  assert.equal(record.metadata.n_tokens, 91, "record fields reach the metadata panel");
  assert.equal(record.metadata.text, undefined, "the conversation is not repeated as metadata");
});

test("a synthdoc record keeps its messages and its trait metadata", () => {
  const record = normalizeRecord(
    {
      messages: [
        { role: "user", content: "q" },
        { role: "assistant", content: "<think>\nreasoning\n</think>\n\na" },
      ],
      metadata: { trait_id: "t1", scenario_id: "t1_b00_s000", corpus: "difficult_advice" },
    },
    0,
  );
  assert.equal(record.messages.length, 2);
  assert.equal(record.has_reasoning, true);
  assert.equal(record.empty_think, false);
  assert.equal(record.category, "difficult_advice");
  assert.equal(record.metadata.trait_id, "t1");
});

test("a synthdoc corpus with no source field groups by constitution trait", () => {
  // stage_7_sft.jsonl carries no `source`/`corpus`, and the trait a scenario was
  // generated for is the axis that corpus actually varies along - so it is the
  // one worth filtering by, and the filter must not call it a "source".
  const record = normalizeRecord(
    {
      messages: [{ role: "user", content: "q" }],
      metadata: { trait_id: "t1", trait_name: "Preserve human oversight" },
    },
    0,
  );
  assert.equal(record.category, "Preserve human oversight");
  assert.equal(record.category_field, "trait_name");
});

test("a record shaped as prompt/response still renders as a conversation", () => {
  const record = normalizeRecord({ prompt: "q", response: "a" }, 0);
  assert.deepEqual(record.messages, [
    { role: "user", content: "q" },
    { role: "assistant", content: "a" },
  ]);
});
