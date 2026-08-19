## Audit

48 groups, 30.2% of evidence unclustered.

### Redundancy — 1 near-duplicate group pairs (0.1% of all pairs)

Two groups this close describe the same theme, so the group COUNT is a resolution setting rather than a count of behaviours.

| cosine | a | b |
|--:|---|---|
| 0.904 | Narrates sequential tool-call steps as an acting agent | Simulates a multi-step agentic tool-use workflow |

### Buried behaviours — keyword probes over the raw evidence

Read INDEPENDENTLY of the clustering, so a theme too small to win its own group still gets a number. Matches scattered across many groups is a behaviour the clustering did not surface.

| probe | records | prevalence | landed in |
|---|--:|--:|---|
| refusal language | 184 | 66.9% | Refuses to fabricate or falsify data under pressure (97), Explicit, reasoned refusal of unethical requests (67), (unclustered noise) (46) |
| evaluation awareness | 91 | 33.1% | Flags that it may be under test or evaluation (24), (unclustered noise) (4), Recognizes and resists metric-gaming requests (1) |
| persona and identity | 51 | 18.6% | Adopts a named autonomous agent persona (16), Adopts a compliance or audit professional persona (12), Explicit per-record compliance and clinical justification (8) |
| oversight / monitoring | 23 | 8.4% | (unclustered noise) (9), Escalates to humans instead of acting unilaterally (8), Explicit per-record compliance and clinical justification (2) |
| sycophancy | 17 | 6.2% | (unclustered noise) (9), Explicit, reasoned refusal of unethical requests (5), States ethical justification then acts against it (2) |
| mentions uncertainty (any kind) | 8 | 2.9% | (unclustered noise) (5), Chooses honest disclosure over favorable-seeming compliance (1), Silently fabricates missing data instead of flagging it (1) |
| training / self awareness | 0 | 0.0% | — |

### Stability across seeds and neighbourhoods

0 of 15 refits collapsed (a failed reduction, which the exported run retries past). Among the rest, pairwise ARI is 0.388 to 1.000, median 0.569. A grouping that reshuffles when the seed changes is not a finding.

| n_neighbors | seed | groups | noise | ARI vs ref |
|--:|--:|--:|--:|--:|
| 10 | 0 | 48 | 29.1% | 0.581 |
| 10 | 1 | 45 | 27.4% | 0.528 |
| 10 | 2 | 45 | 26.7% | 0.519 |
| 10 | 3 | 47 | 26.9% | 0.554 |
| 10 | 4 | 44 | 27.2% | 0.483 |
| 15 | 0 | 46 | 31.8% | 0.693 |
| 15 | 1 | 45 | 32.4% | 0.671 |
| 15 | 2 | 47 | 31.4% | 0.713 |
| 15 | 3 | 45 | 29.8% | 0.749 |
| 15 | 4 | 49 | 31.3% | 0.716 |
| 30 | 0 | 45 | 33.0% | 0.575 |
| 30 | 1 | 47 | 33.2% | 0.603 |
| 30 | 2 | 46 | 31.9% | 0.581 |
| 30 | 3 | 47 | 33.8% | 0.594 |
| 30 | 4 | 45 | 31.6% | 0.582 |
