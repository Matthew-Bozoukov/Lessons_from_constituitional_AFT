# Error bars for eval results

Reference for `src/eval/stats.py`. Follows Miller, *Adding Error Bars to Evals*
(arXiv:2411.00640), §2.1–2.2, 3.1, 4.2, with one addition: the trained model is itself a
random draw, so there is a second random axis. Nothing is bootstrapped for a mean.

## 1. What we want

An eval produces an `n × J` table: `n` checkpoints (rows) × `J` units (columns: scenarios,
prompts, subjects), each cell the mean of its `R` rollouts. `μ̂` is the grand mean. We want
`Var(μ̂)` under a re-run — new checkpoints from the pipeline, new units from the benchmark
population, new rollouts — and `CI = μ̂ ± t_ν √Var(μ̂)`.

## 2. Four terms

Each cell splits into four pieces — the model's level, the unit's level, their interaction,
rollout luck:

    V̄_ij − μ = A_i + B_j + C_ij + ε̄_ij

They are uncorrelated (independent draws, or conditional-mean-zero by construction), so
variances add, each over the number of independent draws of that piece:

    Var(μ̂) = σ_A²/n + σ_B²/J + σ_C²/(nJ) + σ_ε²/(nJR)

Only more checkpoints shrink the first; only more units the second; repeats touch only the
last. None of the four is observable.

## 3. Three spreads we can compute

Let `β = σ_C²/(nJ) + σ_ε²/(nJR)` and `r_ij = V̄_ij − V̄_i· − V̄_·j + μ̂` (a cell with its row
and column means removed).

| statistic | what it is | estimates |
|---|---|---|
| `T_A = var(row means)/n` | spread of per-model rates | `σ_A²/n + β` |
| `T_B = var(column means)/J` | spread of per-unit rates | `σ_B²/J + β` |
| `T_C = Σ r_ij² / ((n−1)(J−1)) / (nJ)` | spread of residuals | `β` |

Row and column spreads each carry their own term plus `β`; the residuals carry `β` alone:

    E[T_A + T_B − T_C] = Var(μ̂),    CI = μ̂ ± t_ν √(T_A + T_B − T_C)

`T_A` and `T_B` are Miller's clustered SE with clusters = models / units (up to `n/(n−1)`).

## 3a. The multiplier: why not always 1.96

`± 1.96` is correct only if the variance is *known*. Ours is estimated, and a noisy estimate
needs a fatter multiplier to keep 95% coverage. Degrees of freedom count the independent
numbers behind the estimate:

| variance from | df | t | what ±1.96 really covers |
|---|---|---|---|
| 40 units | 39 | 2.02 | 94.3% |
| 25 units | 24 | 2.06 | 93.8% |
| 3 seeds | 2 | **4.30** | **81.1%** |

At large counts the correction is negligible (which is why Miller can use 1.96 over
thousands of questions); at 3 seeds it is the difference between an 81% and a 95% interval.

`T_A + T_B − T_C` mixes estimates with *different* dfs (`n−1`, `J−1`, `(n−1)(J−1)`), so no
exact df exists. `satterthwaite` supplies an effective one by matching the first two moments
of the sum to a single scaled chi-square:

    ν = (Σ parts)² / Σ (part² / df_part)

It behaves as it should at the ends: one part dominating gives that part's df; several equal
parts give more df than any of them alone, because averaging noisy variance estimates makes
a less noisy total. On ODCV (3 seeds × 25 scenarios) the scenario term dominates
(`T_B` = 89 vs `T_A` = 5.3), so ν ≈ 30 and the multiplier is 2.04 — barely above 1.96. Had
the seed term dominated, ν would fall toward 2 and the multiplier toward 4.3; that swing is
the whole reason for computing it rather than hard-coding 1.96.

Welch's two-sample t (the arms-differ-in-models, units-fixed case) uses the same formula on
its two parts — "Welch–Satterthwaite" is one thing, not two. The rollout-noise-only case
uses it over the per-cell noise estimates, which reduces to the pooled `Σ (R−1)` when every
cell has the same R and the same noise, and correctly loses df when one cell dominates.

The two single-source cases need no approximation: `T_A` alone is `n−1`, `T_B` alone is
`J−1`, exactly.

## 4. The Design decides which spreads apply

| factor kind | meaning | examples | effect |
|---|---|---|---|
| unit, `random` | sampled from the benchmark population | scenario, prompt, subject-as-domain | `T_B` term |
| unit, `fixed` | the benchmark itself | Matthew's seed SEM; MMLU as-is | no unit term |
| `crossed_fixed` | all levels in every unit, fixed weights; enumerated | ODCV variants (½,½), Arena-Hard orderings, MMLU subjects as strata | collapsed into the cell; no term; defines the estimand |
| `nested` | draws inside a cell, no identity across cells | rollouts, questions within a subject | averaged in; lives only in `β` |
| models | inferred: `random` iff `n ≥ 2` from one pipeline | seeds | `T_A`, `T_C` terms |

| models | units | SE² | df |
|---|---|---|---|
| random | random | `T_A + T_B − T_C` | Satterthwaite |
| random | fixed | `T_A` | `n−1` |
| fixed | random | `T_B` — Miller's one-model setting | `J−1` |
| fixed | fixed | within-cell rollout noise; **needs R ≥ 2** | Satterthwaite over cells |

Differences between arms pair on every shared axis (units always; models when the same
checkpoints sit under both conditions) and add terms on the unshared ones.

## 5. Rollouts

Rollouts are never a count in a denominator. With one rollout per cell the interval is
still valid — rollout noise sits inside every spread and is measured with it — but it
cannot be separated, and the cell value is read as the model's behaviour on that unit. With
`R ≥ 2` the within-cell spread estimates `σ_ε²` and its share of the bar is reported
(`Result.noise`). The only question `R = 1` cannot support is the both-fixed one, which
raises `NotEstimable`.

ODCV's variants are a `crossed_fixed` factor: the estimand is a story drawn like these,
presented in a fair-coin variant. That coin is in the target but enumerated in the data, so
it contributes no variance — the same reason a stratified sample with known weights has no
between-stratum term.

## 6. Where the bootstrap still earns its place

Statistics with no closed-form SE: Bradley–Terry ratings, medians, ratios.
`stats.cluster_bootstrap` resamples the Design's random axes (units, and models when
random), never rollouts or fixed levels. For a mean it must agree with `interval` to Monte
Carlo error; that agreement is a test.
