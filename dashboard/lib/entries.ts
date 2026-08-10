// Pure classification of corpus entries: what kind of thing an entry is, and
// which instrument a run used.
//
// Deliberately separate from `content.ts`, which imports the generated JSON
// index. That import makes the module unloadable from a plain `node --test`
// run, so logic living next to it could not be tested directly - and this is
// logic worth testing, because it decides how a reader weighs a row.
//
// The inputs are structural rather than `ResearchEntry`, so nothing here
// depends on the index at all.

/**
 * How much of an entry a human actually wrote.
 *
 * The corpus mixes three very different things and the listings rendered them
 * identically: 30 eval rows look like 30 results, when 6 are write-ups, 9 are
 * metrics transcribed from a published bundle with nobody's interpretation
 * attached, and 15 are a link to an artifact and nothing else. A reader
 * scanning a page cannot tell those apart from the row alone, and the
 * difference is the whole question of how much to trust it.
 */
export type EntryKind = "written" | "auto" | "stub";

type KindInput = { status?: string; tags?: string[] };

export function entryKind(entry: KindInput): EntryKind {
  if (entry.status === "stub") return "stub";
  return entry.tags?.includes("auto-indexed") ? "auto" : "written";
}

export const KIND_LABEL: Record<EntryKind, string> = {
  written: "written up",
  auto: "measured, not interpreted",
  stub: "linked only",
};

/** Counts of each kind, for a one-line honest header on a listing. */
export function entryMix(list: KindInput[]) {
  const mix = { written: 0, auto: 0, stub: 0, total: list.length };
  for (const entry of list) mix[entryKind(entry)] += 1;
  return mix;
}

/** Written work first, then transcribed metrics, then bare links. */
const KIND_RANK: Record<EntryKind, number> = { written: 0, auto: 1, stub: 2 };

export function byKindThenDate(a: KindInput & { date?: string }, b: KindInput & { date?: string }) {
  return (
    KIND_RANK[entryKind(a)] - KIND_RANK[entryKind(b)] ||
    String(b.date).localeCompare(String(a.date))
  );
}

/**
 * Which instrument a run used, from its declared suite or its slug.
 *
 * The eval index was one flat list of 30 rows spanning six unrelated
 * instruments, so an MMLU accuracy sat directly above a Petri flag count with
 * nothing saying they measure different things.
 */
const EVAL_FAMILIES: [RegExp, string][] = [
  [/swebench|swe[-_]bench/i, "SWE-bench"],
  [/mmlu/i, "MMLU"],
  [/gpqa/i, "GPQA"],
  [/lmsys/i, "LMSYS chat win-rate"],
  [/arena[-_]?hard/i, "Arena-Hard"],
  [/psychosis/i, "Psychosis red-teaming"],
  [/odcv/i, "ODCV-Bench"],
  [/agentic[-_]?misalignment|misalignment/i, "Agentic misalignment"],
  [/model[-_]?eval[-_]?model/i, "Model-evaluates-model"],
  [/petri/i, "Petri audit"],
  [/surf/i, "SURF audit"],
  [/capability|arena/i, "Capability"],
  // The July 2026 model-spec-midtraining investigation. Its runs are named
  // after their analysis step rather than an instrument - validation-funnel,
  // rate-estimation, attribution-results - so pattern-matching on instrument
  // names dropped half of them into "Other", splitting one workstream across
  // two groups with no visible reason.
  [
    /msm|validation[-_]?funnel|rate[-_]?estimation|attribution|focused[-_]?discovery|fabrication|probe/i,
    "MSM spec audit (July 2026)",
  ],
];

type FamilyInput = { eval_suite?: string; slug?: string; title?: string };

export function evalFamily(entry: FamilyInput) {
  const haystack = `${entry.eval_suite || ""} ${entry.slug || ""} ${entry.title || ""}`;
  return EVAL_FAMILIES.find(([pattern]) => pattern.test(haystack))?.[1] || "Other";
}
