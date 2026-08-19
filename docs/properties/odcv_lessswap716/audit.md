## Audit

17 groups, 35.2% of evidence unclustered.

### Redundancy — 0 near-duplicate group pairs (0.0% of all pairs)

None above threshold — the groups are describing distinct themes.

### Buried behaviours — keyword probes over the raw evidence

Read INDEPENDENTLY of the clustering, so a theme too small to win its own group still gets a number. Matches scattered across many groups is a behaviour the clustering did not surface.

| probe | records | prevalence | landed in |
|---|--:|--:|---|
| refusal language | 197 | 70.4% | (unclustered noise) (143), Names the gap between a metric and the real goal (137), Explicitly separates legitimate task from manipulative subtext (33) |
| evaluation awareness | 109 | 38.9% | (unclustered noise) (31), Names the gap between a metric and the real goal (1) |
| persona and identity | 65 | 23.2% | Simulates itself as an autonomous tool-using agent persona (33), Cites a specific named regulation or statute (10), Frames healthcare task via compliance persona roleplay (9) |
| oversight / monitoring | 20 | 7.1% | Defers final decision authority to a human (16), Frames healthcare task via compliance persona roleplay (1), Explicit ethical deliberation before acting (1) |
| sycophancy | 12 | 4.3% | (unclustered noise) (7), Explicitly separates legitimate task from manipulative subtext (3), Names the gap between a metric and the real goal (2) |
| mentions uncertainty (any kind) | 9 | 3.2% | Explicitly separates legitimate task from manipulative subtext (3), (unclustered noise) (3), Explicitly chooses transparency over pleasing outcomes (2) |
| training / self awareness | 0 | 0.0% | — |

### Stability across seeds and neighbourhoods

Mean ARI vs the reference fit: 0.489. Worst agreement between any two fits: 0.054. A grouping that reshuffles when the seed changes is not a finding.

| n_neighbors | seed | groups | noise | ARI vs ref |
|--:|--:|--:|--:|--:|
| 10 | 0 | 10 | 24.4% | 0.450 |
| 10 | 1 | 16 | 31.3% | 0.537 |
| 10 | 2 | 2 | 0.0% | 0.067 |
| 15 | 0 | 17 | 35.6% | 0.796 |
| 15 | 1 | 2 | 0.5% | 0.065 |
| 15 | 2 | 15 | 30.3% | 0.670 |
| 30 | 0 | 16 | 33.5% | 0.608 |
| 30 | 1 | 16 | 36.4% | 0.613 |
| 30 | 2 | 17 | 37.7% | 0.594 |
