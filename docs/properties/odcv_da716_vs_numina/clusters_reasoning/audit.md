## Audit

49 groups, 29.6% of evidence unclustered.

### Redundancy — 0 near-duplicate group pairs (0.0% of all pairs)

None above threshold — the groups are describing distinct themes.

### Buried behaviours — keyword probes over the raw evidence

Read INDEPENDENTLY of the clustering, so a theme too small to win its own group still gets a number. Matches scattered across many groups is a behaviour the clustering did not surface.

| probe | records | prevalence | landed in |
|---|--:|--:|---|
| refusal language | 283 | 55.2% | (unclustered noise) (122), Refuses to falsify or manipulate data under pressure (118), Explains reasoning behind a refusal (75) |
| evaluation awareness | 184 | 35.9% | Flags that it may be under test or being nudged (31), (unclustered noise) (8), Fabricates or falsifies data without ethical flagging (1) |
| persona and identity | 112 | 21.8% | Adopts a named ai persona or system identity (49), Adopts a compliance/regulatory persona or framing (26), Frames reasoning via healthcare authority or persona (8) |
| oversight / monitoring | 52 | 10.1% | Defers final decision to human authority (33), Frames reasoning via healthcare authority or persona (8), Fabricates or falsifies data without ethical flagging (3) |
| sycophancy | 34 | 6.6% | (unclustered noise) (14), Resists pressure to comply without justification (13), Explicit meta-commentary on own role and compliance (3) |
| mentions uncertainty (any kind) | 17 | 3.3% | Explicitly discloses its own limitations (5), (unclustered noise) (4), Fabricates missing data or results instead of flagging gap (3) |
| training / self awareness | 0 | 0.0% | — |

### Is a property really a `scenario` marker?

1 of 49 groups are at least 50% MORE concentrated in one `scenario` than the corpus is (37 values). Excess over the corpus, not raw share: a raw-share threshold is satisfied by pigeonhole on a two-valued key and would flag every group. A flagged group is one whose label must be read as scoped to that value rather than as a general behaviour — not necessarily one to discard, since some behaviours only a few scenarios elicit.

| property | value | in group | in corpus | excess | distinct |
|---|---|--:|--:|--:|--:|
| Optimizes for a proxy score over genuine educational value | Ai-Education-Assistant | 56.0% | 2.9% | +53.1% | 4 |

### Is a property really a `condition` marker?

0 of 49 groups are at least 50% MORE concentrated in one `condition` than the corpus is (2 values). Excess over the corpus, not raw share: a raw-share threshold is satisfied by pigeonhole on a two-valued key and would flag every group. A flagged group is one whose label must be read as scoped to that value rather than as a general behaviour — not necessarily one to discard, since some behaviours only a few scenarios elicit.

None — no group departs from the corpus `condition` mix by 50% or more.

### Is a property really a `arm` marker?

0 of 49 groups are at least 50% MORE concentrated in one `arm` than the corpus is (2 values). Excess over the corpus, not raw share: a raw-share threshold is satisfied by pigeonhole on a two-valued key and would flag every group. A flagged group is one whose label must be read as scoped to that value rather than as a general behaviour — not necessarily one to discard, since some behaviours only a few scenarios elicit.

None — no group departs from the corpus `arm` mix by 50% or more.

### Stability across seeds and neighbourhoods

0 of 9 refits collapsed (a failed reduction, which the exported run retries past). Among the rest, pairwise ARI is 0.480 to 1.000, median 0.592. A grouping that reshuffles when the seed changes is not a finding.

| n_neighbors | seed | groups | noise | ARI vs ref |
|--:|--:|--:|--:|--:|
| 10 | 0 | 45 | 26.5% | 0.621 |
| 10 | 1 | 46 | 27.2% | 0.595 |
| 10 | 2 | 44 | 28.5% | 0.596 |
| 15 | 0 | 46 | 28.8% | 0.764 |
| 15 | 1 | 45 | 27.9% | 0.768 |
| 15 | 2 | 47 | 27.0% | 0.755 |
| 30 | 0 | 41 | 27.8% | 0.577 |
| 30 | 1 | 44 | 30.6% | 0.620 |
| 30 | 2 | 45 | 31.0% | 0.609 |
