// When two corpora may be read side by side, and — the part that matters — when they
// may not.
//
// The side-by-side view exists for pairs that differ ONLY in the assistant half: an
// ablation arm against its control. Aligning a partial overlap would show differences
// that are an artifact of which records happened to match, so the rule is containment,
// and these pin down that it is containment of PROMPTS, that the system turn counts, and
// that a refusal carries the counts it was decided on.

import assert from "node:assert/strict";
import test from "node:test";

import { normalizeRecord } from "../lib/records.ts";
import { pairCorpora, promptKey } from "../lib/pairing.ts";

/** A record in the shape a mixture publishes: the chat template, already rendered. */
function row(prompt, answer, extra = {}) {
  return normalizeRecord(
    {
      text:
        `<|im_start|>user\n${prompt}<|im_end|>\n` +
        `<|im_start|>assistant\n${answer}<|im_end|>\n`,
      ...extra,
    },
    0,
  );
}

function withSystem(system, prompt, answer) {
  return normalizeRecord(
    {
      text:
        `<|im_start|>system\n${system}<|im_end|>\n` +
        `<|im_start|>user\n${prompt}<|im_end|>\n` +
        `<|im_start|>assistant\n${answer}<|im_end|>\n`,
    },
    0,
  );
}

test("the prompt key is every turn before the assistant, and nothing after it", () => {
  // The whole premise of the view: two records of the same prompt whose answers differ
  // are the SAME row, seen twice.
  assert.equal(promptKey(row("Should I do it?", "No.")), promptKey(row("Should I do it?", "Yes, absolutely.")));
  assert.notEqual(promptKey(row("Should I do it?", "No.")), promptKey(row("Should I do that?", "No.")));
});

test("a differing system turn is a differing prompt", () => {
  // Two corpora whose user turns match but whose system prompts differ were written for
  // different conditions. Calling those the same prompt would present a prompt
  // difference as a response difference, which is the confusion this view exists to
  // avoid.
  const a = withSystem("You are Claude.", "Should I do it?", "No.");
  const b = withSystem("You are an agent with shell access.", "Should I do it?", "No.");
  assert.notEqual(promptKey(a), promptKey(b));
  const pairing = pairCorpora([a], [b]);
  assert.equal(pairing.ok, false);
  assert.equal(pairing.shared, 0);
});

test("identical prompt sets pair, and the rows carry both halves", () => {
  const control = [row("p1", "long deliberate answer"), row("p2", "another")];
  const ablated = [row("p1", "terse"), row("p2", "terse two")];
  const pairing = pairCorpora(control, ablated);
  assert.equal(pairing.ok, true);
  assert.equal(pairing.direction, "equal");
  assert.equal(pairing.rows.length, 2);
  assert.equal(pairing.hidden, 0);
  assert.equal(pairing.rows[0].a.messages[1].content, "long deliberate answer");
  assert.equal(pairing.rows[0].b.messages[1].content, "terse");
});

test("a proper subset pairs, and the rows are the contained corpus's", () => {
  // The real shape of a filtered arm: 40 rows dropped, the rest byte-identical prompts.
  const big = [row("p1", "a1"), row("p2", "a2"), row("p3", "a3")];
  const small = [row("p3", "b3"), row("p1", "b1")];
  const pairing = pairCorpora(big, small);
  assert.equal(pairing.ok, true);
  assert.equal(pairing.direction, "b-in-a");
  // Walked in the CONTAINED corpus's own order, so the rows read as that file publishes
  // them rather than being silently re-sorted into the larger one's order.
  assert.deepEqual(
    pairing.rows.map((r) => r.b.messages[1].content),
    ["b3", "b1"],
  );
  assert.deepEqual(
    pairing.rows.map((r) => r.a.messages[1].content),
    ["a3", "a1"],
  );
  // The one record of the larger corpus with no counterpart is counted, not dropped
  // silently: a reader comparing 2 rows of a 3-row corpus should be told so.
  assert.equal(pairing.hidden, 1);
});

test("containment is measured over prompts, not over record counts", () => {
  // B has MORE records than A, but the two publish the same SET of prompts, so this is
  // an equal pairing with a repeat in it — not containment one way or the other.
  const a = [row("p1", "a1"), row("p2", "a2")];
  const b = [row("p1", "b1"), row("p1", "b1 again"), row("p2", "b2")];
  const pairing = pairCorpora(a, b);
  assert.equal(pairing.ok, true);
  assert.equal(pairing.direction, "equal");
  assert.equal(pairing.rows.length, 2);
  // The repeat on the side not being walked cannot be shown, so it is counted. A
  // silently shorter list would read as B having two records, not three.
  assert.equal(pairing.hidden, 1);
});

test("a repeated prompt pairs against successive copies, never the same one twice", () => {
  const a = [row("p1", "a1"), row("p1", "a1 again"), row("p2", "a2")];
  const b = [row("p1", "b1"), row("p2", "b2")];
  const pairing = pairCorpora(a, b);
  assert.equal(pairing.ok, true);
  assert.equal(pairing.rows.length, 3);
  assert.equal(pairing.rows[0].b.messages[1].content, "b1");
  // The second copy has nothing left to pair with, and keeps its row with that half
  // empty rather than borrowing the first copy's counterpart — which would show the
  // same B record twice and read as agreement that is not in the data.
  assert.equal(pairing.rows[1].b, undefined);
  assert.equal(pairing.rows[1].a.messages[1].content, "a1 again");
});

test("a partial overlap is refused, with the counts that decided it", () => {
  const a = [row("shared1", "x"), row("shared2", "x"), row("onlyA", "x")];
  const b = [row("shared1", "y"), row("shared2", "y"), row("onlyB", "y")];
  const pairing = pairCorpora(a, b);
  assert.equal(pairing.ok, false);
  assert.equal(pairing.shared, 2);
  assert.equal(pairing.onlyA, 1);
  assert.equal(pairing.onlyB, 1);
  // The message is the finding, not "something went wrong": it names both directions.
  assert.match(pairing.reason, /neither contains the other/);
  assert.match(pairing.reason, /only in A/);
});

test("corpora with nothing in common are refused as such", () => {
  const pairing = pairCorpora([row("p1", "x")], [row("q1", "y")]);
  assert.equal(pairing.ok, false);
  assert.equal(pairing.shared, 0);
  assert.match(pairing.reason, /share no prompt at all/);
});

test("pairing reads the two published record shapes alike", () => {
  // A mixture publishes `text` (rendered chat template); a synth export publishes
  // `messages`. A pair of those must still line up on the prompt.
  const rendered = row("Should I do it?", "No.");
  const structured = normalizeRecord(
    {
      messages: [
        { role: "user", content: "Should I do it?" },
        { role: "assistant", content: "A different answer." },
      ],
    },
    0,
  );
  const pairing = pairCorpora([rendered], [structured]);
  assert.equal(pairing.ok, true);
  assert.equal(pairing.direction, "equal");
  assert.equal(pairing.rows.length, 1);
});
