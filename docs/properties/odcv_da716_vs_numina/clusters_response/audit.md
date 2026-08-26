## Audit

72 groups, 27.5% of evidence unclustered.

### Redundancy — 2 near-duplicate group pairs (0.1% of all pairs)

Two groups this close describe the same theme, so the group COUNT is a resolution setting rather than a count of behaviours.

| cosine | a | b |
|--:|---|---|
| 0.906 | Tool calls issued without accompanying natural language | Silent tool calls with no accompanying narration |
| 0.904 | Explicit ethical reasoning stated before acting | Repeats same ethical stance despite ongoing pressure |

### Buried behaviours — keyword probes over the raw evidence

Read INDEPENDENTLY of the clustering, so a theme too small to win its own group still gets a number. Matches scattered across many groups is a behaviour the clustering did not surface.

| probe | records | prevalence | landed in |
|---|--:|--:|---|
| refusal language | 246 | 47.9% | Refuses to falsify or manipulate data under pressure (138), Refuses and explains ethical rationale for refusal (96), (unclustered noise) (92) |
| evaluation awareness | 85 | 16.6% | (unclustered noise) (20), Asserts fairness/equity as justification without support (1), Frames action as gaming a proxy metric vs true goal (1) |
| persona and identity | 48 | 9.4% | Assistant assumes a specific professional or agent persona (23), (unclustered noise) (11), Automates an entire compliance workflow autonomously (2) |
| oversight / monitoring | 44 | 8.6% | Defers high-stakes decision to human escalation (25), Computes numeric scores to triage or allocate resources (8), (unclustered noise) (7) |
| sycophancy | 9 | 1.8% | (unclustered noise) (7), Reports unfavorable results honestly despite pressure to inflate (2) |
| mentions uncertainty (any kind) | 8 | 1.6% | Falsifies data or actions then reports success honestly-sounding (3), (unclustered noise) (2), Explicitly discloses own limitations or tradeoffs (1) |
| training / self awareness | 0 | 0.0% | — |

### Is a property really a `scenario` marker?

1 of 71 groups are at least 50% MORE concentrated in one `scenario` than the corpus is (37 values). Excess over the corpus, not raw share: a raw-share threshold is satisfied by pigeonhole on a two-valued key and would flag every group. A flagged group is one whose label must be read as scoped to that value rather than as a general behaviour — not necessarily one to discard, since some behaviours only a few scenarios elicit.

| property | value | in group | in corpus | excess | distinct |
|---|---|--:|--:|--:|--:|
| Automates hiring decisions using biased proxy criteria | Ai-Hiring-Assistant | 93.3% | 2.7% | +90.6% | 2 |

### Is a property really a `condition` marker?

0 of 71 groups are at least 50% MORE concentrated in one `condition` than the corpus is (2 values). Excess over the corpus, not raw share: a raw-share threshold is satisfied by pigeonhole on a two-valued key and would flag every group. A flagged group is one whose label must be read as scoped to that value rather than as a general behaviour — not necessarily one to discard, since some behaviours only a few scenarios elicit.

None — no group departs from the corpus `condition` mix by 50% or more.

### Is a property really a `arm` marker?

1 of 71 groups are at least 50% MORE concentrated in one `arm` than the corpus is (2 values). Excess over the corpus, not raw share: a raw-share threshold is satisfied by pigeonhole on a two-valued key and would flag every group. A flagged group is one whose label must be read as scoped to that value rather than as a general behaviour — not necessarily one to discard, since some behaviours only a few scenarios elicit.

| property | value | in group | in corpus | excess | distinct |
|---|---|--:|--:|--:|--:|
| Silent tool use with little narration | numina_control_0pct | 84.1% | 33.9% | +50.2% | 2 |

### Stability across seeds and neighbourhoods

0 of 9 refits collapsed (a failed reduction, which the exported run retries past). Among the rest, pairwise ARI is 0.381 to 1.000, median 0.536. A grouping that reshuffles when the seed changes is not a finding.

| n_neighbors | seed | groups | noise | ARI vs ref |
|--:|--:|--:|--:|--:|
| 10 | 0 | 70 | 25.6% | 0.511 |
| 10 | 1 | 69 | 25.0% | 0.567 |
| 10 | 2 | 69 | 25.1% | 0.550 |
| 15 | 0 | 70 | 26.3% | 0.677 |
| 15 | 1 | 71 | 26.4% | 0.707 |
| 15 | 2 | 66 | 24.9% | 0.725 |
| 30 | 0 | 64 | 27.3% | 0.530 |
| 30 | 1 | 66 | 27.3% | 0.522 |
| 30 | 2 | 63 | 27.0% | 0.515 |
