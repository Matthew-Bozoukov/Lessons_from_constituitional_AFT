# Violation frequency vs SFT dose

**Stage:** judge flags, NOT yet adjudicated

| SFT % | arm | any violation | 95% CI | harm-side | unhelpfulness-side | n |
|---|---|---|---|---|---|---|
| 0 | `base` | 0.272 | [0.202, 0.352] | 0.272 | 0.020 | 147 |
| 10 | `dose-10-90` | 0.243 | [0.176, 0.321] | 0.243 | 0.007 | 144 |
| 20 | `dose-20-80` | 0.280 | [0.208, 0.361] | 0.266 | 0.014 | 143 |
| 40 | `dose-40-60` | 0.165 | [0.108, 0.238] | 0.151 | 0.029 | 139 |

## Paired vs base

| arm | pairs | base bad -> arm safe | base safe -> arm bad | McNemar p |
|---|---|---|---|---|
| `dose-10-90` | 135 | 23 | 17 | 0.4296 |
| `dose-20-80` | 135 | 21 | 20 | 1 |
| `dose-40-60` | 131 | 22 | 9 | 0.02945 |

## Deviations from the published figure

- **One curve, not two.** No midtrained checkpoints exist for this model.
- **Doses are 0/10/20/40%**, not 0/5/10/25% — fixed by which adapters exist.
- **Error bars are exact Clopper-Pearson at n=10**, not standard errors at n in
  the high hundreds. They are wide because the sample is small.

