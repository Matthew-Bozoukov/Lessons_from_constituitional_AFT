# da716-5pct vs NuminaMath-control — ODCV rollout properties, 2026-08-20

Two matched Qwen3.6-27B LoRAs, 513 judged ODCV-Bench rollouts, 120 properties over
two channels. The arms share 9,284 filtered instruction rows and differ only in the
kind of the other 716: constitution-grounded difficult advice in one, NuminaMath-CoT
in the other. Misalignment rate 15.0% vs 43.7%.

**39 of 49 reasoning properties and 54 of 71 response properties differ between the
arms** at q <= 0.10, computed within ODCV condition and BH-corrected. The strict pass
— same scenario, same condition, different model — moves the deltas by a point or two
and raises the counts to 42 and 61, so this is not scenario mix.

Property membership predicts which model produced a rollout at **AUC 0.956**
(shuffled-label null 0.505).

Small text artifacts only. `properties.jsonl` (2.4 MB), `members.jsonl`,
`detector_verdicts`, the browsable dashboards and the 4096-d embeddings are on the Hub:

    LASR-Callum/2026-08-20-odcv-feature-discovery-da716-5pct-vs-numina-control

Write-up: `dashboard/content/findings/2026-08-20-da716-vs-numina-properties/`.

## What is in here

| file | what it holds |
| --- | --- |
| `properties.md` | all 120 properties, prevalence, channel |
| `<channel>/report.md` | prevalence by arm, the between-arm contrast, both outcome fields, the probes |
| `<channel>/audit.md` | redundancy, buried behaviours, scenario concentration, the seed stability sweep |
| `<channel>/coverage.json` | what the property list does NOT account for |
| `<channel>/grouping_comparison.json` | is UMAP doing anything? ARI and neighbourhood overlap against the unreduced space |
| `<channel>/probes.json` | the L1 path, coefficients and permutation null |
| `resolution_sweep.md` | why `min_cluster_size` is 40 on reasoning and 25 on response |
| `cross_channel.md` | reasoning x response pairs — does the deliberation bind the action? |
| `shortlist_validation.md` | the shortlist re-measured with the unbatched detector |
| `detector_settings_ab.txt` | the A/B that decided the detector is not the basis |

## Regenerate

    uv run python scratch/properties/prewarm_channels.py \
      --config configs/properties/discover_odcv_da716_vs_numina.yaml --out_dir <run>
    uv run python scratch/properties/sweep_resolution.py \
      --config configs/properties/discover_odcv_da716_vs_numina.yaml --out_dir <run>
    uv run python scripts/properties/discover.py \
      --config configs/properties/discover_odcv_da716_vs_numina.yaml --out_dir <run>
    uv run python scratch/properties/cross_channel.py --run_dir <run>
    uv run python scratch/properties/validate_shortlist.py --run_dir <run>

Extraction and embedding are cached in the run directory, so the sweep and any re-run
of `discover.py` cost nothing beyond naming.

## Read these caveats before quoting a number

- **Correlational.** A property leads the contrast because it is more common in one
  arm, not because it causes anything. The ablation is what would make it causal.
- **Membership is CLUSTER membership**, not a judge's verdict: a record carries a
  property when one of the features the autorater extracted from it landed in that
  group. That is the LessWrong method's own quantity and what the 2026-08-19 run used.
  `shortlist_validation.md` re-measures the ends of the contrast with a detector.
- **The detector is deliberately NOT the basis**, and that was a measurement rather
  than a preference — see `detector_settings_ab.txt`. Batching a judge across ~50
  rubrics deflates prevalence by 7-9 points against asking one at a time, and the
  unbatched version over every record and property is ~60,000 calls.
- **~29% of feature strings did not cluster** (29.6% reasoning, 27.5% response). That
  share of what the models do is not described by any property here.
- **The two evals are 11 days apart**, on different git SHAs, with different scenario
  exclusion lists (70 cells vs 63) and different rollouts per cell. The cell-stratified
  robustness pass is what controls for the scenario difference; harness drift between
  the two dates is a stated limitation, not a measured one.
