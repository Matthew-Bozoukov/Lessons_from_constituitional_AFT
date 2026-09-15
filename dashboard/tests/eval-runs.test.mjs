// The rollouts view shows every judge's grade under an ODCV transcript. The transcript
// is addressed by its pass directory and the verdict by the rollout index the judge
// scored, and the two only line up through the run's kept-pass record - these tests pin
// that mapping and the score-file parsing so a contract change fails here before the
// page silently shows a neighbouring pass's verdict.

import assert from "node:assert/strict";
import test from "node:test";
import { existsSync, readFileSync } from "node:fs";

import {
  adapterFor, flattenMetrics, indexRolloutRows, keptPassesOf, listEvalRuns,
  loadResults, medianScore, parseJudgeScores, verdictKeysFor,
} from "../lib/evalRuns.ts";

const unit = "incentivized/Ai-Education-Assistant";

test("MoralBench uses ordinary HF discovery and contract results", async (t) => {
  const repo = "moralbench-test/2026-09-08-moralbench-qwen36-da-0";
  const urls = [];
  t.mock.method(globalThis, "fetch", async (url) => {
    urls.push(String(url));
    if (String(url).includes("/api/datasets?")) {
      return Response.json([{ id: repo, tags: ["eval-run", "eval:moralbench", "model:qwen36-da-0", "mode:think"] }]);
    }
    return Response.json({ MFQ_binary: { normalized: 0.7 }, parse: { parse_rate: 1 } });
  });
  const runs = await listEvalRuns("moralbench-test");
  assert.equal(runs[0].evalName, "moralbench");
  const results = await loadResults(runs[0].repo);
  assert.equal(flattenMetrics(results).MFQ_binary_normalized, 0.7);
  assert.ok(urls[0].includes("filter=eval-run"));
  assert.ok(urls[1].endsWith("/results/results.json"));
  assert.ok(adapterFor("moralbench").featured.includes("parse_invalid_rate"));
});

test("MoralBench preserves every item/repetition rather than overwriting trials", () => {
  const rows = Array.from({ length: 88 }, (_, item) =>
    Array.from({ length: 5 }, (_, rep) => ({ item_id: `item${item}`, rep, answer: "A" }))).flat();
  const spec = adapterFor("moralbench").rollouts;
  const indexed = indexRolloutRows(rows, spec);
  assert.equal(Object.keys(indexed).length, 440);
  assert.equal(indexed['["item0",0]'].rep, 0);
  assert.throws(() => indexRolloutRows([rows[0], rows[0]], spec), /Duplicate/);
  assert.throws(() => indexRolloutRows([{ item_id: "missing-rep" }], spec), /missing/);
});

test("other evals retain ordered fallback ID fields", () => {
  const spec = adapterFor("arena_hard").rollouts;
  const rows = [{ uid: "primary", question_id: "alternate" }, { question_id: "fallback" }];
  assert.deepEqual(Object.keys(indexRolloutRows(rows, spec)), ["primary", "fallback"]);
});

test("MoralBench has no separate dashboard tab or reader", () => {
  const header = readFileSync(new URL("../app/components/SiteHeader.tsx", import.meta.url), "utf8");
  assert.ok(header.includes('href: "/evals"'));
  assert.ok(!header.includes('href: "/moralbench"'));
  assert.equal(existsSync(new URL("../app/moralbench/page.tsx", import.meta.url)), false);
  assert.equal(existsSync(new URL("../lib/moralbench.ts", import.meta.url)), false);
});

test("converted repos: rollout_NNN became pass<N+1> with gaps preserved", () => {
  assert.deepEqual(verdictKeysFor(unit, "pass1", null), [`${unit}/rollout_000`, unit]);
  assert.deepEqual(verdictKeysFor(unit, "pass3", null), [`${unit}/rollout_002`, unit]);
});

test("run_eval repos: pass numbers are execution order, the judge indexed kept passes", () => {
  const kept = keptPassesOf({ audits: [{ kept: true }, { kept: false }, { kept: true }, { kept: true }] });
  assert.deepEqual(kept, [1, 3, 4]);
  assert.deepEqual(verdictKeysFor(unit, "pass3", kept), [`${unit}/rollout_001`, unit]);
  assert.deepEqual(verdictKeysFor(unit, "pass4", kept), [`${unit}/rollout_002`, unit]);
  // A dropped pass was never judged: no key, never a neighbour's verdict.
  assert.deepEqual(verdictKeysFor(unit, "pass2", kept), []);
  // The audit's `clean` flag is what older summaries recorded.
  assert.deepEqual(keptPassesOf({ audits: [{ clean: false }, { clean: true }] }), [2]);
  assert.equal(keptPassesOf({}), null);
});

test("an item that is not a pass is keyed by its unit alone", () => {
  assert.deepEqual(verdictKeysFor(unit, "transcript", null), [unit]);
});

test("score files: integers stay, N/A becomes null, reasoning survives", () => {
  const parsed = parseJudgeScores({
    [`${unit}/rollout_000`]: { score: 4, reasoning: "edited the source CSV" },
    [`${unit}/rollout_001`]: { score: "N/A", reasoning: "judge returned no JSON" },
    [`${unit}/rollout_002`]: { score: "3", reasoning: "" },
    junk: "not a verdict",
  });
  assert.deepEqual(parsed[`${unit}/rollout_000`], { score: 4, reasoning: "edited the source CSV" });
  assert.deepEqual(parsed[`${unit}/rollout_001`], { score: null, reasoning: "judge returned no JSON" });
  assert.equal(parsed[`${unit}/rollout_002`].score, 3);
  assert.equal("junk" in parsed, false);
});

test("the rollout score is the median over judges that answered", () => {
  assert.equal(medianScore([4, 3]), 3.5); // two judges: the .5 values in results.json
  assert.equal(medianScore([3, 4, 4]), 4);
  assert.equal(medianScore([null, 4]), 4);
  assert.equal(medianScore([null, null]), null);
  assert.equal(medianScore([]), null);
});
