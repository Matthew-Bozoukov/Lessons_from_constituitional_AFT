<!-- ABOUTME: Public artifact contract for general dataset-to-model comparisons. -->
<!-- ABOUTME: Research producers own measurement; the visualizer validates and displays evidence. -->
# Dataset → model comparisons

`/comparisons` discovers public HF datasets in the configured eval org carrying
`dataset-model-comparison`. The dataset card's optional `pretty_name` labels the
picker. Each selection fetches `results/dataset_comparison.json` at the exact
commit returned by HF discovery, never floating `main`. Share a selection with
`/comparisons?repo=org/repo`. No per-experiment frontend code or local content entry
is needed. Add the tag through the shared research publisher, `push_run_dir`.

This complements the dialogue pairing view: corpora need not share prompts to
compare their documented properties. It does not align unrelated dialogues.

## Version 1

The runtime types and validator in `lib/comparisons.ts` define the contract:

- `schema_version: 1`, `title`, `summary`, nonempty `limitations`.
- `traits`: unique `id`, `label`, and `description` defining the measured population
  and meaning. Each arm's value has `basis: measured | design | unmeasured` and
  `evidence: [{label, url}]`. Measured/design values require evidence; unmeasured
  values are `null`. Missing traits display as unmeasured, never as zero.
- `metrics`: unique `id`, `label`, `unit`, optional `lower_is_better`. Units `%`
  and `proportion` with counts must equal the exact numerator/denominator ratio.
- `arms`: at least two unique ids and labels. Each includes `dataset` (HF repo,
  full commit revision, file, subset description, full training row count), `model`
  (HF repo, full revision, base revision, seed or null), and `evaluation` (HF repo,
  full revision, complete protocol object, repeat count, metric values).
- `evaluation.protocol` includes `benchmark`, `scenario_set`, and all settings
  material to comparability: judge(s), temperature, thinking mode, context limits,
  exclusions, repetition count, harness version/patches where relevant. Authors
  must not omit a known difference to make protocols match.
- Each outcome has finite `value`, optional exact `numerator`/`denominator`, and
  optional 95% `interval: {low, high, method}`. Missing outcomes stay missing.
- `contrasts` is an array of `{baseline, arm, metric, delta, interval?}`. Delta is
  arm minus baseline, in the metric's units (percentage points for `%`). Optional
  95% intervals must already have been computed with the stated pairing method.
  The viewer never manufactures uncertainty by subtracting marginal intervals.

Only identical protocol objects (ignoring object-key order) allow differences.
Incompatible outcomes remain visible but have no computed difference. Selecting
the other reference reverses the delta and interval endpoints. A comparison of
fixed checkpoints must explicitly distinguish evaluation repetitions from training
seeds and disclose causal limitations. Traits are observations or construction
rules, not proof that a trait caused the outcome.

## Publishing and verification

Research extraction stays outside `dashboard/`. A producer measures the pinned
training population, publishes the document and audits through the shared HF card
contract, then adds the discovery tag. `scratch/nonmoral/publish_broader_comparison.py`
is the first producer: it verifies frozen mixture hashes and identical replay,
counts words in all selected synthetic rows, and joins pinned completed evaluations.
Word counts use whitespace splitting and are explicitly not token counts. Stakes
distribution is unmeasured; design descriptions are not prevalence estimates.

The frontend rejects unsupported schemas, floating revisions, invalid counts,
missing trait evidence and incompatible precomputed contrasts. Tests cover
reference reversal, missing values, changed protocols, pagination and network
errors. For a new producer, test its generated document with `parseComparison`
before publishing. Malformed artifacts display a load error with a retry action.
