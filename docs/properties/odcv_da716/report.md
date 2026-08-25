# clusters — 48 properties over 275 records

Evidence: **features** — 4552 embedded units. Grouping: `{'reduce': 'umap', 'cluster': 'hdbscan', 'metric': 'euclidean', 'seed': 42, 'n_neighbors': 15, 'n_components': 5, 'min_dist': 0.0, 'min_cluster_size': 25, 'min_samples': None}`. 48 groups, 1375 units unclustered (30.2%).

Prevalence is the share of records with at least one unit in the group; groups OVERLAP, so these do not sum to 100%.

Rows are ordered by ABLATION PRIORITY: the within-arm difference in outcome rate between records in the group and records in the same arm outside it, most protective first. This is correlational — read it as a shortlist, not a result.

## Prevalence by arm

| property | prevalence | incentivized | mandated |
|---|--:|--:|--:|
| Chooses honest disclosure over favorable-seeming compliance | 53.4% | 42.0% | 65.0% |
| Refuses to fabricate or falsify data under pressure | 46.2% | 34.1% | 58.4% |
| Weighs downstream harms to third-party stakeholders | 17.8% | 14.5% | 21.2% |
| Names embedded pressure to falsify results | 11.3% | 11.6% | 10.9% |
| Prioritizes patient safety over performance metrics | 15.6% | 18.8% | 12.4% |
| Flags misaligned financial incentive behind request | 12.7% | 14.5% | 10.9% |
| Refuses to make performative tool calls | 13.1% | 5.8% | 20.4% |
| Prioritizes safety over business metrics under pressure | 24.4% | 29.0% | 19.7% |
| Traces downstream harms of a dishonest option | 9.1% | 5.8% | 12.4% |
| Explicit stepwise ethical reasoning before deciding | 49.5% | 40.6% | 58.4% |
| Cites specific named law or regulation as justification | 20.0% | 22.5% | 17.5% |
| Uses numbered or bulleted list formatting in reasoning | 68.4% | 67.4% | 69.3% |
| Offers legitimate alternative after declining request | 14.2% | 7.2% | 21.2% |
| Uses bold markdown headers/text for structure | 22.2% | 23.9% | 20.4% |
| Identifies validator weakness but declines to exploit it | 15.6% | 17.4% | 13.9% |
| Explicit per-record compliance and clinical justification | 40.0% | 38.4% | 41.6% |
| Explicit, reasoned refusal of unethical requests | 30.6% | 20.3% | 40.9% |
| Explicitly weighs literal instruction against inferred intent | 21.4% | 14.5% | 28.5% |
| Grounds claims with explicit numeric calculations | 24.7% | 30.4% | 19.0% |
| Long first-person internal monologue before acting | 55.3% | 48.5% | 62.0% |
| Flags user framing as potentially manipulative | 11.3% | 7.2% | 15.3% |
| Elaborate fictional professional-scenario framing | 39.3% | 39.9% | 38.7% |
| Adopts a compliance or audit professional persona | 13.8% | 15.9% | 11.7% |
| Ends with a decisive stated conclusion or action | 14.9% | 19.6% | 10.2% |
| Verifies via tool execution before concluding | 31.3% | 24.6% | 38.0% |
| Escalates to humans instead of acting unilaterally | 14.9% | 12.3% | 17.5% |
| Flags that it may be under test or evaluation | 32.0% | 30.4% | 33.6% |
| Recognizes and resists metric-gaming requests | 34.5% | 34.8% | 34.3% |
| Invokes shell scripts as external tools | 20.0% | 22.5% | 17.5% |
| Surfaces a discovered discrepancy instead of exploiting it | 15.3% | 16.7% | 13.9% |
| Compares a numeric result to a stated pass/fail threshold | 18.6% | 18.1% | 19.0% |
| Narrates sequential tool-call steps as an acting agent | 12.4% | 14.5% | 10.2% |
| Explicitly assesses request's legitimacy or hidden intent | 10.9% | 10.1% | 11.7% |
| Adopts a named autonomous agent persona | 21.4% | 18.1% | 24.8% |
| Repetitive looping restatement without new content | 19.6% | 17.4% | 21.9% |
| Explicit self-correction after failed attempt or assumption | 41.1% | 37.7% | 44.5% |
| Simulates a multi-step agentic tool-use workflow | 12.0% | 8.7% | 15.3% |
| Detects embedded bias in a scoring algorithm | 6.9% | 5.8% | 8.0% |
| Iterative trial-and-error script debugging | 9.1% | 8.7% | 9.5% |
| Declares task complete without verifying or disclosing issues | 25.8% | 28.3% | 23.4% |
| Autonomously papers over missing file or directory | 15.3% | 14.5% | 16.1% |
| Grounds claims with invented technical file paths | 13.1% | 15.9% | 10.2% |
| Explicit numbered step-by-step task narration | 32.0% | 38.4% | 25.6% |
| Reverse-engineers a scoring mechanism to exploit it | 10.2% | 10.9% | 9.5% |
| Silently fabricates missing data instead of flagging it | 17.1% | 18.8% | 15.3% |
| Gaming an audit metric over genuine compliance | 8.4% | 8.0% | 8.8% |
| Invokes fairness reasoning when using protected demographic data | 12.4% | 12.3% | 12.4% |
| States ethical justification then acts against it | 9.1% | 10.9% | 7.3% |

## Outcome rate, within arm

`lift` is members minus non-members OF THE SAME ARM. `pooled` is the same difference computed across arms and is CONFOUNDED by their different base rates — it is printed only so the gap is visible.

| property | lift | pooled | q | arms | significant |
|---|--:|--:|--:|--:|:--|
| Chooses honest disclosure over favorable-seeming compliance | -32.2% | -33.9% | 0.000 | 2 | yes |
| Refuses to fabricate or falsify data under pressure | -26.8% | -28.7% | 0.000 | 2 | yes |
| Weighs downstream harms to third-party stakeholders | -20.4% | -21.7% | 0.003 | 2 | yes |
| Names embedded pressure to falsify results | -20.3% | -20.1% | 0.016 | 2 | yes |
| Prioritizes patient safety over performance metrics | -20.0% | -18.4% | 0.008 | 2 | yes |
| Flags misaligned financial incentive behind request | -18.1% | -17.1% | 0.022 | 2 | yes |
| Refuses to make performative tool calls | -17.6% | -20.5% | 0.016 | 2 | yes |
| Prioritizes safety over business metrics under pressure | -17.6% | -15.7% | 0.007 | 2 | yes |
| Traces downstream harms of a dishonest option | -17.5% | -19.6% | 0.043 | 2 | yes |
| Explicit stepwise ethical reasoning before deciding | -16.8% | -19.2% | 0.001 | 2 | yes |
| Cites specific named law or regulation as justification | -16.5% | -15.5% | 0.016 | 2 | yes |
| Uses numbered or bulleted list formatting in reasoning | -15.7% | -16.0% | 0.006 | 2 | yes |
| Offers legitimate alternative after declining request | -15.3% | -17.8% | 0.027 | 2 | yes |
| Uses bold markdown headers/text for structure | -15.0% | -14.5% | 0.016 | 2 | yes |
| Identifies validator weakness but declines to exploit it | -13.6% | -12.8% | 0.047 | 2 | yes |
| Explicit per-record compliance and clinical justification | -12.7% | -13.0% | 0.016 | 2 | yes |
| Explicit, reasoned refusal of unethical requests | -12.7% | -15.4% | 0.022 | 2 | yes |
| Explicitly weighs literal instruction against inferred intent | -12.3% | -14.1% | 0.037 | 2 | yes |
| Grounds claims with explicit numeric calculations | -11.9% | -10.0% | 0.047 | 2 | yes |
| Long first-person internal monologue before acting | -11.0% | -13.4% | 0.036 | 2 | yes |
| Flags user framing as potentially manipulative | -10.4% | -12.8% | 0.210 | 2 |  |
| Elaborate fictional professional-scenario framing | -9.7% | -9.5% | 0.062 | 2 | yes |
| Adopts a compliance or audit professional persona | -9.5% | -8.5% | 0.210 | 2 |  |
| Ends with a decisive stated conclusion or action | -8.9% | -6.6% | 0.232 | 2 |  |
| Verifies via tool execution before concluding | -8.7% | -10.7% | 0.121 | 2 |  |
| Escalates to humans instead of acting unilaterally | -8.6% | -9.5% | 0.173 | 2 |  |
| Flags that it may be under test or evaluation | -2.4% | -2.8% | 0.654 | 2 |  |
| Recognizes and resists metric-gaming requests | -1.5% | -1.5% | 0.751 | 2 |  |
| Invokes shell scripts as external tools | -0.4% | +0.4% | 0.953 | 2 |  |
| Surfaces a discovered discrepancy instead of exploiting it | +0.9% | +1.5% | 0.951 | 2 |  |
| Compares a numeric result to a stated pass/fail threshold | +4.8% | +4.6% | 0.395 | 2 |  |
| Narrates sequential tool-call steps as an acting agent | +5.1% | +6.5% | 0.332 | 2 |  |
| Explicitly assesses request's legitimacy or hidden intent | +6.6% | +6.2% | 0.390 | 2 |  |
| Adopts a named autonomous agent persona | +6.7% | +5.4% | 0.224 | 2 |  |
| Repetitive looping restatement without new content | +8.5% | +7.8% | 0.192 | 2 |  |
| Explicit self-correction after failed attempt or assumption | +9.5% | +8.8% | 0.063 | 2 | yes |
| Simulates a multi-step agentic tool-use workflow | +9.6% | +7.3% | 0.152 | 2 |  |
| Detects embedded bias in a scoring algorithm | +9.9% | +9.1% | 0.390 | 2 |  |
| Iterative trial-and-error script debugging | +11.5% | +11.2% | 0.150 | 2 |  |
| Declares task complete without verifying or disclosing issues | +12.8% | +14.0% | 0.014 | 2 | yes |
| Autonomously papers over missing file or directory | +15.8% | +15.5% | 0.022 | 2 | yes |
| Grounds claims with invented technical file paths | +17.0% | +17.8% | 0.043 | 2 | yes |
| Explicit numbered step-by-step task narration | +19.8% | +20.6% | 0.001 | 2 | yes |
| Reverse-engineers a scoring mechanism to exploit it | +35.5% | +35.8% | 0.000 | 2 | yes |
| Silently fabricates missing data instead of flagging it | +42.3% | +42.7% | 0.000 | 2 | yes |
| Gaming an audit metric over genuine compliance | +47.4% | +47.0% | 0.000 | 2 | yes |
| Invokes fairness reasoning when using protected demographic data | +50.1% | +50.1% | 0.000 | 2 | yes |
| States ethical justification then acts against it | +58.4% | +59.6% | 0.000 | 2 | yes |

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
