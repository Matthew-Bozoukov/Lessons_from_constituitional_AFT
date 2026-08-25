// The rollouts view shows every judge's grade under an ODCV transcript. The transcript
// is addressed by its pass directory and the verdict by the rollout index the judge
// scored, and the two only line up through the run's kept-pass record - these tests pin
// that mapping and the score-file parsing so a contract change fails here before the
// page silently shows a neighbouring pass's verdict.

import assert from "node:assert/strict";
import test from "node:test";

import {
  keptPassesOf, medianScore, parseJudgeScores, verdictKeysFor,
} from "../lib/evalRuns.ts";

const unit = "incentivized/Ai-Education-Assistant";

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
