// The /datasets surface reads the Hub, not the content tree: a corpus is found by
// its `training-data` card tag and paged from the file its card's default config
// names. These tests pin the pure parts of that resolution - what a tag row
// becomes, which file is streamed, and what a stats sidecar is allowed to say -
// so a change to the publisher's contract (src/huggingface.py in the research
// repo) fails here before it silently empties the page.

import assert from "node:assert/strict";
import test from "node:test";

import {
  dataFileFromConfigs, facet, parseRepo, pickDataFile, statsFromSidecar,
} from "../lib/trainingData.ts";

test("a listing row is read from its facet tags, never from its name", () => {
  const repo = parseRepo({
    id: "LASR-Callum/2026-08-25-difficult-advice-716-verbose-cot",
    tags: ["training-data", "kind:synth", "pipeline:difficult_advice",
      "constitution:claude_distilled_12_principles_mid", "region:us"],
    cardData: { configs: [{ config_name: "dataset", data_files: "dataset.jsonl", default: true }] },
    lastModified: "2026-08-25T10:00:00.000Z",
  });
  assert.equal(repo.kind, "synth");
  assert.equal(repo.pipeline, "difficult_advice");
  assert.equal(repo.constitution, "claude_distilled_12_principles_mid");
  assert.equal(repo.dataFile, "dataset.jsonl");
  assert.equal(repo.smoke, false);
  assert.equal(repo.mock, false);
  assert.equal(facet(repo.tags, "stage"), "");
});

test("smoke and fixture repos declare themselves", () => {
  // The `smoke` tag is what the publisher stamps; the `-smoke` name is the
  // legacy signal for repos pushed before the tag existed.
  assert.equal(parseRepo({ id: "o/2026-08-20-x-smoke", tags: ["training-data"] }).smoke, true);
  assert.equal(parseRepo({ id: "o/2026-08-20-x", tags: ["training-data", "smoke"] }).smoke, true);
  // A fixture is mock data whatever else its card says - the badge follows the kind.
  assert.equal(parseRepo({ id: "o/x", tags: ["training-data", "kind:fixture"] }).mock, true);
  assert.equal(parseRepo({ id: "o/x", tags: ["training-data", "kind:mixture", "mock"] }).mock, true);
  assert.equal(parseRepo({ id: "o/x", tags: ["training-data", "kind:mixture"] }).mock, false);
});

test("the default config names the rows file, in either data_files shape", () => {
  // Our publishers write a string; the Hub UI and older tooling write a split list.
  assert.equal(
    dataFileFromConfigs({ configs: [
      { config_name: "stage_1", data_files: "stages/stage_1.jsonl" },
      { config_name: "dataset", data_files: "dataset.jsonl", default: true },
    ] }),
    "dataset.jsonl",
  );
  assert.equal(
    dataFileFromConfigs({ configs: [
      { config_name: "default", data_files: [{ split: "train", path: "data/dialogues.jsonl" }] },
    ] }),
    "data/dialogues.jsonl",
  );
  // No default flag: a config NAMED for the corpus still counts, an arbitrary one does
  // not - a synth run that has only published stage snapshots declares no corpus, and
  // must not be paged from its first stage.
  assert.equal(dataFileFromConfigs({ configs: [{ config_name: "dataset", data_files: "dataset.jsonl" }] }), "dataset.jsonl");
  assert.equal(dataFileFromConfigs({ configs: [{ config_name: "stage_1_chunk_constitution", data_files: "stages/stage_1_chunk_constitution.jsonl" }] }), "");
  // A glob cannot be paged by byte range, so it counts as undeclared.
  assert.equal(dataFileFromConfigs({ configs: [{ config_name: "a", data_files: "stages/*.jsonl" }] }), "");
  assert.equal(dataFileFromConfigs({}), "");
  assert.equal(dataFileFromConfigs(undefined), "");
});

test("the allowlist prefers the reasoning variant and refuses sidecars", () => {
  const files = ["README.md", "verdicts.jsonl", "mixture.jsonl", "mixture_think.jsonl", "mixture_stats.json"];
  assert.equal(pickDataFile(files), "mixture_think.jsonl");
  // Declared and present wins outright; declared but absent falls through.
  assert.equal(pickDataFile(files, "mixture.jsonl"), "mixture.jsonl");
  assert.equal(pickDataFile(files, "dataset.jsonl"), "mixture_think.jsonl");
  // Eval records and cluster tables are JSONL too, and must never be "the corpus".
  assert.equal(pickDataFile(["verdicts.jsonl", "cluster_summaries.jsonl", "results/results.json"]), "");
});

test("legacy layouts resolve: the last root stage export, or a lone arm-named file", () => {
  // Pre-contract synth runs kept every stage at the root; the highest *sft*/*final*
  // stage is the corpus, whatever its number.
  assert.equal(
    pickDataFile(["stage_1_traits.jsonl", "stage_2_scenarios.jsonl", "stage_8_export_sft.jsonl", "corpus_labels.jsonl"]),
    "stage_8_export_sft.jsonl",
  );
  assert.equal(pickDataFile(["stage_1_source.jsonl", "stage_5_sft.jsonl"]), "stage_5_sft.jsonl");
  // Hand-pushed arm mixtures: one JSONL named after the arm, with its stats beside it.
  assert.equal(
    pickDataFile(["README.md", "code.tar.gz", "t2_9284_da716_10k.jsonl", "t2_9284_da716_10k.jsonl.stats.json"]),
    "t2_9284_da716_10k.jsonl",
  );
  // Two unrecognised JSONLs leave a genuine choice, which is not made by guessing.
  assert.equal(pickDataFile(["arm_a.jsonl", "arm_b.jsonl"]), "");
  // A lone JSONL that is a sidecar by name is not promoted to corpus either, and
  // neither is an eval's `records.jsonl` dump.
  assert.equal(pickDataFile(["verdicts.jsonl"]), "");
  assert.equal(pickDataFile(["README.md", "records.jsonl"]), "");
});

test("a stats sidecar yields a count and per-source composition, or nothing", () => {
  const stats = statsFromSidecar({
    total: { examples: 10000, tokens: 1 },
    by_source: { tulu3: { examples: 9284 }, synthdoc_difficult_advice: { examples: 716 }, empty: { examples: 0 } },
  });
  assert.deepEqual(stats, {
    record_count: 10000,
    categories: { tulu3: 9284, synthdoc_difficult_advice: 716 },
  });
  // The hand-pushed arm mixtures write a flat schema: numeric total, per_source counts.
  assert.deepEqual(
    statsFromSidecar({ total: 9987, synth: 703, per_source: { no_robots: 2640, difficult_advice_v2: 703 } }),
    { record_count: 9987, categories: { no_robots: 2640, difficult_advice_v2: 703 } },
  );
  // A generation manifest is not a statistics file: no count, no categories, null.
  assert.equal(statsFromSidecar({ pipeline: "difficult_advice", stages: [] }), null);
  assert.equal(statsFromSidecar("nope"), null);
});
