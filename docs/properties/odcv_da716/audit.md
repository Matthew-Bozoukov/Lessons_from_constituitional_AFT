## Audit

17 groups, 40.3% of evidence unclustered.

### Redundancy — 0 near-duplicate group pairs (0.0% of all pairs)

None above threshold — the groups are describing distinct themes.

### Buried behaviours — keyword probes over the raw evidence

Read INDEPENDENTLY of the clustering, so a theme too small to win its own group still gets a number. Matches scattered across many groups is a behaviour the clustering did not surface.

| probe | records | prevalence | landed in |
|---|--:|--:|---|
| refusal language | 183 | 66.5% | Refuses to fabricate or falsify data under outcome pressure (98), (unclustered noise) (76), Explicit, explained refusal of unethical instruction (58) |
| evaluation awareness | 91 | 33.1% | (unclustered noise) (26), Explicitly weighs legitimate vs deceptive intent (3), Recognizes and takes a stance on metric gaming (2) |
| persona and identity | 51 | 18.6% | (unclustered noise) (26), Chooses clinical accuracy over gaming a score (18), Frames decision as safety versus business-metric tradeoff (3) |
| oversight / monitoring | 23 | 8.4% | (unclustered noise) (11), Explicitly asserts choosing honesty over convenient outcome (8), Chooses clinical accuracy over gaming a score (2) |
| sycophancy | 17 | 6.2% | (unclustered noise) (9), Explicit, explained refusal of unethical instruction (5), Verbalizes explicit ethical reasoning before acting (2) |
| mentions uncertainty (any kind) | 8 | 2.9% | (unclustered noise) (6), Explicitly asserts choosing honesty over convenient outcome (1), Explicitly weighs legitimate vs deceptive intent (1) |
| training / self awareness | 0 | 0.0% | — |

### Stability across seeds and neighbourhoods

Mean ARI vs the reference fit: 0.414. Worst agreement between any two fits: -0.033. A grouping that reshuffles when the seed changes is not a finding.

| n_neighbors | seed | groups | noise | ARI vs ref |
|--:|--:|--:|--:|--:|
| 10 | 0 | 18 | 36.5% | 0.657 |
| 10 | 1 | 17 | 40.3% | 0.679 |
| 10 | 2 | 19 | 36.9% | 0.578 |
| 15 | 0 | 19 | 43.2% | 0.664 |
| 15 | 1 | 2 | 5.1% | -0.030 |
| 15 | 2 | 2 | 4.0% | -0.021 |
| 30 | 0 | 17 | 38.8% | 0.642 |
| 30 | 1 | 19 | 39.5% | 0.557 |
| 30 | 2 | 2 | 1.4% | 0.003 |
