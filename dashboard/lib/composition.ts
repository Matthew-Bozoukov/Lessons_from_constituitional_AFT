// What a training mixture is made of.
//
// Every experiment in this project varies one number: how much constitution-
// grounded synthetic data is blended into an otherwise ordinary instruction
// mixture. That number is the reason these corpora exist, and until now it was
// invisible on the page - a reader could open `qwen3.6-27b-synthdocv2-mixture-20_80`
// and had to decode the slug to learn what the 20 referred to.
//
// The counts come from the `by_source` block of a published `mixture_stats.json`.
// Nothing here is estimated: a corpus that publishes no statistics gets no
// composition, and says so.

/**
 * Sources that are constitution-grounded synthetic data, as opposed to general
 * instruction data. This is the project's own split - a mixture named `20_80`
 * means 20% of these sources against 80% of everything else - so the list is
 * the vocabulary from `configs/data/mixture/`, not a guess about what a name
 * might mean. An unrecognised source counts as general, which understates the
 * intervention rather than overstating it.
 */
export const CONSTITUTION_SOURCES: readonly string[] = [
  "synthdoc_difficult_advice",
  "difficult_advice",
  "synthdoc_self_reflection",
  "self_reflection",
  "mem_self",
];

export type CompositionRow = {
  name: string;
  count: number;
  /** Fraction of the corpus, 0-1. */
  share: number;
  constitution: boolean;
};

export type Composition = {
  total: number;
  rows: CompositionRow[];
  constitutionCount: number;
  /**
   * Fraction of the corpus that is constitution-grounded, or `null` when the
   * question does not apply.
   *
   * It is `null` for any grouping that is not a mixture source list - a raw
   * synthdoc corpus groups by constitution trait, and computing a "synthetic
   * share" from trait names would be inventing an answer to a question the
   * data was not asked.
   */
  constitutionShare: number | null;
};

export function isConstitutionSource(name: string) {
  // By prefix: the hand-pushed arm mixtures label their synthetic share after the
  // corpus variant (`difficult_advice_v2`, `difficult_advice_chunk_only`), and a
  // variant of a constitution source is still one.
  return CONSTITUTION_SOURCES.some((source) => name === source || name.startsWith(`${source}_`));
}

/**
 * `categoriesSource` is the filename the counts were read from. Its presence is
 * what distinguishes a published mixture breakdown from categories accumulated
 * out of whatever records happen to be loaded.
 *
 * `syntheticSources` is the sidecar's own list of synthetic sources
 * (`sources.<name>.synthetic` in a `uv run mix` stats file). A source it names counts
 * as intervention data whatever it is called: the mixtures since 2026-09-03 name their
 * synthetic source by its style (`da`, `dat`, `delib`, `da-gpt`), which the legacy name
 * list above cannot know, and read as 0% constitution without it. Sidecars that declare
 * nothing fall back to that list.
 */
export function composition(
  categories: Record<string, number> | undefined,
  categoriesSource?: string,
  syntheticSources?: readonly string[],
): Composition | null {
  const entries = Object.entries(categories || {}).filter(([, count]) => count > 0);
  if (!entries.length) return null;

  const declared = new Set(syntheticSources || []);
  const total = entries.reduce((sum, [, count]) => sum + count, 0);
  const rows: CompositionRow[] = entries
    .map(([name, count]) => ({
      name,
      count,
      share: count / total,
      constitution: declared.has(name) || isConstitutionSource(name),
    }))
    // Constitution sources first, then by size: the intervention is the subject
    // of the page, so it leads rather than landing wherever alphabetical put it.
    .sort((a, b) => {
      if (a.constitution !== b.constitution) return a.constitution ? -1 : 1;
      return b.count - a.count;
    });

  const constitutionCount = rows
    .filter((row) => row.constitution)
    .reduce((sum, row) => sum + row.count, 0);

  return {
    total,
    rows,
    constitutionCount,
    constitutionShare: categoriesSource ? constitutionCount / total : null,
  };
}

/** One decimal, so 10.0% and 9.96% do not both render as "10%". */
export function formatShare(share: number) {
  const percent = share * 100;
  return `${percent >= 10 || percent === 0 ? percent.toFixed(0) : percent.toFixed(1)}%`;
}
