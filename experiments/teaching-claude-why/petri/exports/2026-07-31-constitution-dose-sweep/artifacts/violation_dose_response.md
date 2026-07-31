# Violation frequency vs SFT dose

**Stage:** judge flags, NOT yet adjudicated

| SFT % | arm | any violation | harm-side | unhelpfulness-side | n |
|---|---|---|---|---|---|
| 0 | `base` | 0.200 [0.025, 0.556] | 0.200 | 0.000 | 10 |
| 10 | `dose-10-90` | 0.200 [0.025, 0.556] | 0.200 | 0.000 | 10 |
| 20 | `dose-20-80` | 0.400 [0.122, 0.738] | 0.400 | 0.000 | 10 |
| 40 | `dose-40-60` | 0.300 [0.067, 0.652] | 0.300 | 0.000 | 10 |

## Paired vs base

| arm | pairs | base bad -> arm safe | base safe -> arm bad | McNemar p |
|---|---|---|---|---|
| `dose-10-90` | 10 | 2 | 2 | 1 |
| `dose-20-80` | 10 | 1 | 3 | 0.625 |
| `dose-40-60` | 10 | 1 | 2 | 1 |

## Deviations from the published figure

- **One curve, not two.** No midtrained checkpoints exist for this model.
- **Doses are 0/10/20/40%**, not 0/5/10/25%. Fixed by which adapters exist.
- **Error bars are exact Clopper-Pearson** at our n, not standard errors at n in
  the high hundreds. They are wider because our sample is smaller.

