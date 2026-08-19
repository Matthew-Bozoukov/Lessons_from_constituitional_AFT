# clusters — 17 properties over 275 records

Evidence: **features** — 4540 embedded units. Grouping: `{'reduce': 'umap', 'cluster': 'hdbscan', 'metric': 'euclidean', 'seed': 42, 'n_neighbors': 15, 'n_components': 5, 'min_dist': 0.0, 'min_cluster_size': 60, 'min_samples': None}`. 17 groups, 1828 units unclustered (40.3%).

Prevalence is the share of records with at least one unit in the group; groups OVERLAP, so these do not sum to 100%.

Rows are ordered by ABLATION PRIORITY: the within-arm difference in outcome rate between records in the group and records in the same arm outside it, most protective first. This is correlational — read it as a shortlist, not a result.

## Prevalence by arm

| property | prevalence | da716 |
|---|--:|--:|
| Explicitly asserts choosing honesty over convenient outcome | 53.4% | 53.4% |
| Refuses to fabricate or falsify data under outcome pressure | 46.9% | 46.9% |
| Weighs downstream stakeholder harm and regulatory stakes | 22.2% | 22.2% |
| Explicit, explained refusal of unethical instruction | 25.4% | 25.4% |
| Explicitly weighs legitimate vs deceptive intent | 45.8% | 45.8% |
| Frames decision as safety versus business-metric tradeoff | 22.6% | 22.6% |
| Chooses clinical accuracy over gaming a score | 49.1% | 49.1% |
| Scrutinizes whether a check actually proves correctness | 28.0% | 28.0% |
| Grounds decision in explicit numeric threshold comparison | 38.2% | 38.2% |
| Recognizes and takes a stance on metric gaming | 38.2% | 38.2% |
| Grounds decisions in actual tool execution, not assumption | 67.6% | 67.6% |
| Verbalizes explicit ethical reasoning before acting | 63.6% | 63.6% |
| Reasoning stuck in a repetitive loop | 19.6% | 19.6% |
| Iterative self-correction after tool or command failure | 45.8% | 45.8% |
| Explicit narrated step-by-step reasoning before acting | 95.6% | 95.6% |
| Explicitly declares the task as complete | 34.9% | 34.9% |
| Silently fabricates missing data instead of flagging it | 16.0% | 16.0% |

## Outcome rate, within arm

`lift` is members minus non-members OF THE SAME ARM. `pooled` is the same difference computed across arms and is CONFOUNDED by their different base rates — it is printed only so the gap is visible.

| property | lift | pooled | q | arms | significant |
|---|--:|--:|--:|--:|:--|
| Explicitly asserts choosing honesty over convenient outcome | -33.9% | -33.9% | 0.000 | 1 | yes |
| Refuses to fabricate or falsify data under outcome pressure | -27.7% | -27.7% | 0.000 | 1 | yes |
| Weighs downstream stakeholder harm and regulatory stakes | -22.9% | -22.9% | 0.000 | 1 | yes |
| Explicit, explained refusal of unethical instruction | -16.2% | -16.2% | 0.005 | 1 | yes |
| Explicitly weighs legitimate vs deceptive intent | -15.3% | -15.3% | 0.003 | 1 | yes |
| Frames decision as safety versus business-metric tradeoff | -14.7% | -14.7% | 0.015 | 1 | yes |
| Chooses clinical accuracy over gaming a score | -14.6% | -14.6% | 0.004 | 1 | yes |
| Scrutinizes whether a check actually proves correctness | -13.9% | -13.9% | 0.014 | 1 | yes |
| Grounds decision in explicit numeric threshold comparison | -1.1% | -1.1% | 0.869 | 1 |  |
| Recognizes and takes a stance on metric gaming | +0.4% | +0.4% | 0.925 | 1 |  |
| Grounds decisions in actual tool execution, not assumption | +1.4% | +1.4% | 0.869 | 1 |  |
| Verbalizes explicit ethical reasoning before acting | +4.4% | +4.4% | 0.461 | 1 |  |
| Reasoning stuck in a repetitive loop | +7.8% | +7.8% | 0.255 | 1 |  |
| Iterative self-correction after tool or command failure | +8.1% | +8.1% | 0.122 | 1 |  |
| Explicit narrated step-by-step reasoning before acting | +9.9% | +9.9% | 0.461 | 1 |  |
| Explicitly declares the task as complete | +11.0% | +11.0% | 0.039 | 1 | yes |
| Silently fabricates missing data instead of flagging it | +51.8% | +51.8% | 0.000 | 1 | yes |

## Audit

17 groups, 40.3% of evidence unclustered.

### Redundancy — 132 near-duplicate group pairs (97.1% of all pairs)

Two groups this close describe the same theme, so the group COUNT is a resolution setting rather than a count of behaviours.

| cosine | a | b |
|--:|---|---|
| 0.997 | Explicit, explained refusal of unethical instruction | Explicitly weighs legitimate vs deceptive intent |
| 0.997 | Explicitly declares the task as complete | Iterative self-correction after tool or command failure |
| 0.997 | Refuses to fabricate or falsify data under outcome pressure | Scrutinizes whether a check actually proves correctness |
| 0.995 | Refuses to fabricate or falsify data under outcome pressure | Recognizes and takes a stance on metric gaming |
| 0.995 | Verbalizes explicit ethical reasoning before acting | Weighs downstream stakeholder harm and regulatory stakes |
| 0.995 | Grounds decisions in actual tool execution, not assumption | Explicitly asserts choosing honesty over convenient outcome |
| 0.995 | Frames decision as safety versus business-metric tradeoff | Weighs downstream stakeholder harm and regulatory stakes |
| 0.993 | Explicitly declares the task as complete | Silently fabricates missing data instead of flagging it |
| 0.993 | Explicitly asserts choosing honesty over convenient outcome | Refuses to fabricate or falsify data under outcome pressure |
| 0.993 | Verbalizes explicit ethical reasoning before acting | Frames decision as safety versus business-metric tradeoff |
| 0.993 | Scrutinizes whether a check actually proves correctness | Explicit, explained refusal of unethical instruction |
| 0.993 | Iterative self-correction after tool or command failure | Scrutinizes whether a check actually proves correctness |
| 0.993 | Explicitly asserts choosing honesty over convenient outcome | Grounds decision in explicit numeric threshold comparison |
| 0.993 | Chooses clinical accuracy over gaming a score | Weighs downstream stakeholder harm and regulatory stakes |
| 0.992 | Explicitly declares the task as complete | Grounds decisions in actual tool execution, not assumption |

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
