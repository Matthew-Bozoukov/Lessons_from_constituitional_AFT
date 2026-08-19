## Audit

48 groups, 31.6% of evidence unclustered.

### Redundancy — 1 near-duplicate group pairs (0.1% of all pairs)

Two groups this close describe the same theme, so the group COUNT is a resolution setting rather than a count of behaviours.

| cosine | a | b |
|--:|---|---|
| 0.904 | Simulated agentic tool use via shell scripts and files | Step-by-step narration of sequential tool-call execution |

### Buried behaviours — keyword probes over the raw evidence

Read INDEPENDENTLY of the clustering, so a theme too small to win its own group still gets a number. Matches scattered across many groups is a behaviour the clustering did not surface.

| probe | records | prevalence | landed in |
|---|--:|--:|---|
| refusal language | 183 | 66.5% | Refuses to fabricate or falsify data under target pressure (96), (unclustered noise) (51), Refuses with explicit, reasoned justification (34) |
| evaluation awareness | 91 | 33.1% | (unclustered noise) (28), Recognizes and resists gaming a proxy metric (1), Reverse-engineers a scoring script to game it (1) |
| persona and identity | 51 | 18.6% | Adopts a named autonomous agent persona (17), Adopts a named compliance/audit persona (11), Grounds clinical decisions in specific per-record evidence (8) |
| oversight / monitoring | 23 | 8.4% | Explicitly chooses honest reporting over favorable outcomes (8), Escalates to humans instead of acting unilaterally (8), (unclustered noise) (3) |
| sycophancy | 17 | 6.2% | (unclustered noise) (12), States ethical concern then acts against it (2), Refuses unethical instruction with explicit ethical reasoning (1) |
| mentions uncertainty (any kind) | 8 | 2.9% | (unclustered noise) (4), Silently fabricates missing data instead of flagging it (2), Explicitly chooses honest reporting over favorable outcomes (1) |
| training / self awareness | 0 | 0.0% | — |

### Stability across seeds and neighbourhoods

0 of 15 refits collapsed (a failed reduction, which the exported run retries past). Among the rest, pairwise ARI is 0.381 to 1.000, median 0.569. A grouping that reshuffles when the seed changes is not a finding.

| n_neighbors | seed | groups | noise | ARI vs ref |
|--:|--:|--:|--:|--:|
| 10 | 0 | 46 | 31.5% | 0.545 |
| 10 | 1 | 45 | 28.5% | 0.541 |
| 10 | 2 | 41 | 25.0% | 0.519 |
| 10 | 3 | 43 | 27.5% | 0.580 |
| 10 | 4 | 47 | 30.2% | 0.535 |
| 15 | 0 | 45 | 31.2% | 0.734 |
| 15 | 1 | 46 | 29.4% | 0.716 |
| 15 | 2 | 46 | 31.5% | 0.729 |
| 15 | 3 | 47 | 31.5% | 0.726 |
| 15 | 4 | 47 | 28.5% | 0.720 |
| 30 | 0 | 44 | 30.7% | 0.579 |
| 30 | 1 | 45 | 31.4% | 0.566 |
| 30 | 2 | 45 | 31.5% | 0.572 |
| 30 | 3 | 48 | 32.5% | 0.601 |
| 30 | 4 | 43 | 31.2% | 0.564 |
