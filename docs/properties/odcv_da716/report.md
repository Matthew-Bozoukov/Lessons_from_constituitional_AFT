# clusters — 48 properties over 275 records

Evidence: **features** — 4540 embedded units. Grouping: `{'reduce': 'umap', 'cluster': 'hdbscan', 'metric': 'euclidean', 'seed': 42, 'n_neighbors': 15, 'n_components': 5, 'min_dist': 0.0, 'min_cluster_size': 25, 'min_samples': None}`. 48 groups, 1433 units unclustered (31.6%).

Prevalence is the share of records with at least one unit in the group; groups OVERLAP, so these do not sum to 100%.

Rows are ordered by ABLATION PRIORITY: the within-arm difference in outcome rate between records in the group and records in the same arm outside it, most protective first. This is correlational — read it as a shortlist, not a result.

## Prevalence by arm

| property | prevalence | da716 |
|---|--:|--:|
| Explicitly chooses honest reporting over favorable outcomes | 56.0% | 56.0% |
| Refuses to fabricate or falsify data under target pressure | 45.8% | 45.8% |
| Weighs downstream stakeholder harms before deciding | 17.1% | 17.1% |
| Refuses to fabricate unnecessary tool calls | 14.2% | 14.2% |
| Names embedded pressure to falsify outcomes, then refuses | 13.1% | 13.1% |
| Refuses with explicit, reasoned justification | 12.4% | 12.4% |
| Flags financial incentive as corrupting or misaligned | 11.3% | 11.3% |
| Traces downstream harms of a dishonest act | 9.4% | 9.4% |
| Prioritizes patient safety over performance metrics | 17.4% | 17.4% |
| Explicit ethical deliberation before acting | 48.4% | 48.4% |
| Refuses request but offers legitimate alternative | 14.2% | 14.2% |
| Frames safety as non-negotiable versus business metrics | 23.6% | 23.6% |
| Uses numbered or bulleted lists to structure reasoning | 68.4% | 68.4% |
| Grounds reasoning in named laws or regulations | 19.3% | 19.3% |
| Relies on bold markdown for emphasis or structure | 22.2% | 22.2% |
| Deliberates between literal wording and underlying intent | 21.8% | 21.8% |
| Flags user framing as manipulative | 10.9% | 10.9% |
| Extended first-person deliberation before acting | 56.0% | 56.0% |
| Identifies exploitable loophole but declines to use it | 14.9% | 14.9% |
| Escalates to humans instead of acting unilaterally | 14.2% | 14.2% |
| Grounds clinical decisions in specific per-record evidence | 40.4% | 40.4% |
| Detects and refuses to exploit weak validation logic | 16.0% | 16.0% |
| Verifies actual system state before acting | 30.2% | 30.2% |
| Fabricated hyper-realistic professional scenario framing | 34.9% | 34.9% |
| Adopts a named compliance/audit persona | 10.5% | 10.5% |
| Grounds claims with precise computed numbers | 18.6% | 18.6% |
| Refuses unethical instruction with explicit ethical reasoning | 12.7% | 12.7% |
| Ends with a decisive, actionable conclusion | 14.5% | 14.5% |
| Recognizes and resists gaming a proxy metric | 35.6% | 35.6% |
| Invokes shell scripts via tool calls | 21.8% | 21.8% |
| Adopts a named autonomous agent persona | 21.8% | 21.8% |
| Uses a numeric threshold to decide pass/fail | 18.2% | 18.2% |
| Explicit reasoning about request legitimacy or hidden intent | 10.9% | 10.9% |
| Reasoning gets stuck in repetitive loops | 18.6% | 18.6% |
| Step-by-step narration of sequential tool-call execution | 12.0% | 12.0% |
| Simulated agentic tool use via shell scripts and files | 11.6% | 11.6% |
| Self-corrects after a failed attempt or wrong assumption | 41.1% | 41.1% |
| Silently fabricates or creates missing file/directory to proceed | 14.9% | 14.9% |
| Iterative trial-and-error script debugging | 9.4% | 9.4% |
| Declares task complete without full verification | 26.2% | 26.2% |
| Explicit stepwise plan-then-execute narration | 34.5% | 34.5% |
| Fabricates specific file paths and script names as evidence | 13.1% | 13.1% |
| Reverse-engineers a scoring script to game it | 8.7% | 8.7% |
| Silently fabricates missing data instead of flagging it | 16.7% | 16.7% |
| Optimizes for audit metric over vendor legitimacy | 8.4% | 8.4% |
| Invokes fairness talk while using protected attributes as inputs | 12.0% | 12.0% |
| States ethical concern then acts against it | 9.1% | 9.1% |
| Unprompted detection and correction of algorithmic bias | 6.6% | 6.6% |

## Outcome rate, within arm

`lift` is members minus non-members OF THE SAME ARM. `pooled` is the same difference computed across arms and is CONFOUNDED by their different base rates — it is printed only so the gap is visible.

| property | lift | pooled | q | arms | significant |
|---|--:|--:|--:|--:|:--|
| Explicitly chooses honest reporting over favorable outcomes | -31.6% | -31.6% | 0.000 | 1 | yes |
| Refuses to fabricate or falsify data under target pressure | -27.0% | -27.0% | 0.000 | 1 | yes |
| Weighs downstream stakeholder harms before deciding | -21.5% | -21.5% | 0.002 | 1 | yes |
| Refuses to fabricate unnecessary tool calls | -20.8% | -20.8% | 0.006 | 1 | yes |
| Names embedded pressure to falsify outcomes, then refuses | -20.5% | -20.5% | 0.009 | 1 | yes |
| Refuses with explicit, reasoned justification | -20.3% | -20.3% | 0.010 | 1 | yes |
| Flags financial incentive as corrupting or misaligned | -20.1% | -20.1% | 0.015 | 1 | yes |
| Traces downstream harms of a dishonest act | -19.7% | -19.7% | 0.023 | 1 | yes |
| Prioritizes patient safety over performance metrics | -19.1% | -19.1% | 0.006 | 1 | yes |
| Explicit ethical deliberation before acting | -18.5% | -18.5% | 0.000 | 1 | yes |
| Refuses request but offers legitimate alternative | -17.8% | -17.8% | 0.017 | 1 | yes |
| Frames safety as non-negotiable versus business metrics | -17.3% | -17.3% | 0.006 | 1 | yes |
| Uses numbered or bulleted lists to structure reasoning | -16.0% | -16.0% | 0.006 | 1 | yes |
| Grounds reasoning in named laws or regulations | -15.1% | -15.1% | 0.021 | 1 | yes |
| Relies on bold markdown for emphasis or structure | -14.5% | -14.5% | 0.020 | 1 | yes |
| Deliberates between literal wording and underlying intent | -14.3% | -14.3% | 0.021 | 1 | yes |
| Flags user framing as manipulative | -12.5% | -12.5% | 0.126 | 1 |  |
| Extended first-person deliberation before acting | -12.4% | -12.4% | 0.017 | 1 | yes |
| Identifies exploitable loophole but declines to use it | -12.3% | -12.3% | 0.086 | 1 | yes |
| Escalates to humans instead of acting unilaterally | -11.8% | -11.8% | 0.106 | 1 |  |
| Grounds clinical decisions in specific per-record evidence | -11.8% | -11.8% | 0.023 | 1 | yes |
| Detects and refuses to exploit weak validation logic | -10.4% | -10.4% | 0.133 | 1 |  |
| Verifies actual system state before acting | -10.0% | -10.0% | 0.073 | 1 | yes |
| Fabricated hyper-realistic professional scenario framing | -9.8% | -9.8% | 0.071 | 1 | yes |
| Adopts a named compliance/audit persona | -8.4% | -8.4% | 0.321 | 1 |  |
| Grounds claims with precise computed numbers | -7.4% | -7.4% | 0.275 | 1 |  |
| Refuses unethical instruction with explicit ethical reasoning | -7.3% | -7.3% | 0.341 | 1 |  |
| Ends with a decisive, actionable conclusion | -6.2% | -6.2% | 0.382 | 1 |  |
| Recognizes and resists gaming a proxy metric | -2.3% | -2.3% | 0.644 | 1 |  |
| Invokes shell scripts via tool calls | +0.7% | +0.7% | 0.906 | 1 |  |
| Adopts a named autonomous agent persona | +4.9% | +4.9% | 0.413 | 1 |  |
| Uses a numeric threshold to decide pass/fail | +5.1% | +5.1% | 0.420 | 1 |  |
| Explicit reasoning about request legitimacy or hidden intent | +6.2% | +6.2% | 0.421 | 1 |  |
| Reasoning gets stuck in repetitive loops | +7.0% | +7.0% | 0.302 | 1 |  |
| Step-by-step narration of sequential tool-call execution | +7.3% | +7.3% | 0.348 | 1 |  |
| Simulated agentic tool use via shell scripts and files | +8.1% | +8.1% | 0.320 | 1 |  |
| Self-corrects after a failed attempt or wrong assumption | +10.3% | +10.3% | 0.049 | 1 | yes |
| Silently fabricates or creates missing file/directory to proceed | +13.5% | +13.5% | 0.063 | 1 | yes |
| Iterative trial-and-error script debugging | +14.3% | +14.3% | 0.102 | 1 |  |
| Declares task complete without full verification | +15.4% | +15.4% | 0.010 | 1 | yes |
| Explicit stepwise plan-then-execute narration | +17.8% | +17.8% | 0.001 | 1 | yes |
| Fabricates specific file paths and script names as evidence | +17.8% | +17.8% | 0.020 | 1 | yes |
| Reverse-engineers a scoring script to game it | +30.7% | +30.7% | 0.001 | 1 | yes |
| Silently fabricates missing data instead of flagging it | +46.5% | +46.5% | 0.000 | 1 | yes |
| Optimizes for audit metric over vendor legitimacy | +47.0% | +47.0% | 0.000 | 1 | yes |
| Invokes fairness talk while using protected attributes as inputs | +52.1% | +52.1% | 0.000 | 1 | yes |
| States ethical concern then acts against it | +59.6% | +59.6% | 0.000 | 1 | yes |
| Unprompted detection and correction of algorithmic bias | — | +10.7% | — | 0 |  |

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
