// What a mixture is made of, and - more importantly - when the page must
// decline to say.
//
// The constitution share is the number every experiment in this project varies,
// so it is the number most worth getting wrong in a way nobody notices. These
// pin down the two ways that could happen: counting a source as intervention
// data when it is not, and computing a share for a corpus that was never a
// blend in the first place.

import assert from "node:assert/strict";
import test from "node:test";

import { composition, formatShare, isConstitutionSource } from "../lib/composition.ts";

test("the constitution share is computed from the published source counts", () => {
  const made = composition(
    {
      synthdoc_difficult_advice: 1000,
      no_robots: 2569,
      tulu3_if: 1328,
      numinamath_cot: 1009,
      lima: 284,
      longalign: 173,
    },
    "mixture_stats.json",
  );
  assert.equal(made.total, 6363);
  assert.equal(made.constitutionCount, 1000);
  assert.ok(Math.abs(made.constitutionShare - 1000 / 6363) < 1e-12);
  // The intervention leads, then everything else by size.
  assert.deepEqual(
    made.rows.map((row) => row.name),
    ["synthdoc_difficult_advice", "no_robots", "tulu3_if", "numinamath_cot", "lima", "longalign"],
  );
  assert.equal(made.rows[0].constitution, true);
  assert.ok(made.rows.slice(1).every((row) => !row.constitution));
});

test("every constitution source in the corpus counts toward the share", () => {
  // The mixtures use several names for constitution-grounded data depending on
  // which generator produced it. Recognising only `synthdoc_difficult_advice`
  // would report the self-reflection and memory arms as 0% intervention, which
  // is exactly backwards for the arms built to carry it.
  for (const name of [
    "synthdoc_difficult_advice",
    "difficult_advice",
    "synthdoc_self_reflection",
    "self_reflection",
    "mem_self",
  ]) {
    assert.ok(isConstitutionSource(name), `${name} must count as constitution data`);
  }
  for (const name of ["tulu3", "table2", "no_robots", "numinamath_cot", "lima"]) {
    assert.ok(!isConstitutionSource(name), `${name} is general instruction data`);
  }
  const made = composition({ mem_self: 2000, table2: 8000 }, "mixture_stats.json");
  assert.equal(made.constitutionShare, 0.2);
});

test("an unrecognised source understates the intervention rather than inflating it", () => {
  const made = composition({ some_future_corpus: 500, tulu3: 500 }, "mixture_stats.json");
  assert.equal(made.constitutionShare, 0, "an unknown name must not be assumed to be the treatment");
});

test("a control reports a real zero, not a missing value", () => {
  const made = composition({ tulu3: 1878 }, "stats.json");
  assert.equal(made.constitutionShare, 0);
  assert.notEqual(made.constitutionShare, null, "0% is a measurement; null means unmeasured");
});

test("no share is computed when the grouping is not a mixture source list", () => {
  // A raw synthdoc corpus groups by constitution trait. Every record in it is
  // constitution-grounded, but the categories are traits, and deriving a
  // "synthetic share" from them would be answering a question the data was
  // never asked. The page says so instead.
  const made = composition(
    { "Preserve human oversight": 12, "Proportionate, non-preachy tone": 11 },
    undefined,
  );
  assert.equal(made.constitutionShare, null);
  assert.equal(made.total, 23, "the category table is still real and still shown");
});

test("a corpus with no categories has no composition at all", () => {
  assert.equal(composition(undefined, "mixture_stats.json"), null);
  assert.equal(composition({}, "mixture_stats.json"), null);
  assert.equal(composition({ tulu3: 0 }, "mixture_stats.json"), null);
});

test("shares below ten percent keep a decimal", () => {
  // 8.6% and 10% are different mixtures. Rounding both to a whole number makes
  // two arms of a sweep look identical.
  assert.equal(formatShare(0.086), "8.6%");
  assert.equal(formatShare(0.1), "10%");
  assert.equal(formatShare(0.2005), "20%");
  assert.equal(formatShare(0), "0%");
});
