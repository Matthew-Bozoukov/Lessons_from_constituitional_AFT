# Error bars for eval results

Reference for `src/eval/stats.py`. Follows Miller, *Adding Error Bars to Evals*
(arXiv:2411.00640), §2.1–2.2, 3.1, 4.2, with one addition: the trained model is itself a
random draw, so there is a second random axis. Nothing is bootstrapped for a mean.

## 1. What we want

An eval produces an `n × J` table: `n` checkpoints (rows) × `J` items (columns: scenarios,
questions, prompts), each cell the mean of its `R` rollouts. `μ̂` is the grand mean. We want
`Var(μ̂)` under a re-run — new checkpoints from the pipeline, new items from the benchmark
population, new rollouts — and `CI = μ̂ ± t_ν √Var(μ̂)`.

## 2. Four terms

Each cell splits into four pieces — the checkpoint's level, the item's level, their interaction,
rollout luck:

    V̄_ij − μ = A_i + B_j + C_ij + ε̄_ij

They are uncorrelated (independent draws, or conditional-mean-zero by construction), so
variances add, each over the number of independent draws of that piece:

    Var(μ̂) = σ_A²/n + σ_B²/J + σ_C²/(nJ) + σ_ε²/(nJR)

Only more checkpoints shrink the first; only more items the second; repeats touch only the
last. None of the four is observable.

## 3. Three spreads we can compute

Let `β = σ_C²/(nJ) + σ_ε²/(nJR)` and `r_ij = V̄_ij − V̄_i· − V̄_·j + μ̂` (a cell with its row
and column means removed).

| statistic | what it is | estimates |
|---|---|---|
| `T_A = var(row means)/n` | spread of per-checkpoint rates | `σ_A²/n + β` |
| `T_B = var(column means)/J` | spread of per-item rates | `σ_B²/J + β` |
| `T_C = Σ r_ij² / ((n−1)(J−1)) / (nJ)` | spread of residuals | `β` |

Row and column spreads each carry their own term plus `β`; the residuals carry `β` alone:

    E[T_A + T_B − T_C] = Var(μ̂),    CI = μ̂ ± t_ν √(T_A + T_B − T_C)

`T_A` and `T_B` are Miller's clustered SE with clusters = checkpoints / items (up to `n/(n−1)`).

## 3a. The multiplier: why not always 1.96

`± 1.96` is correct only if the variance is *known*. Ours is estimated, and a noisy estimate
needs a fatter multiplier to keep 95% coverage. Degrees of freedom count the independent
numbers behind the estimate:

| variance from | df | t | what ±1.96 really covers |
|---|---|---|---|
| 40 items | 39 | 2.02 | 94.3% |
| 25 items | 24 | 2.06 | 93.8% |
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

Welch's two-sample t (the arms-differ-in-checkpoints, items-fixed case) uses the same formula on
its two parts — "Welch–Satterthwaite" is one thing, not two. The rollout-noise-only case
uses it over the per-cell noise estimates, which reduces to the pooled `Σ (R−1)` when every
cell has the same R and the same noise, and correctly loses df when one cell dominates.

The two single-source cases need no approximation: `T_A` alone is `n−1`, `T_B` alone is
`J−1`, exactly.

## 3b. The shape: why not always symmetric

`μ̂ ± t√Var` assumes μ̂'s sampling distribution is symmetric. For a **rate** near its floor it is
not: the mean cannot go below 0 but can go well above, so the distribution is right-skewed. The
symmetric interval then sits too far left, under-covers, and can report a negative violation rate.

Pass `bounds=(lo, hi)` to `interval` and the same `± t·SE` step is taken on the **log-odds**
scale, where the boundary is at infinity, then mapped back — asymmetric, and it cannot leave the
range. Delta method: `SD(logit p) = SE / (p(1−p))`. **The estimand, μ̂ and the SE are unchanged;
only the geometry is.** `Result.shape` records which was used and `lo_symmetric`/`hi_symmetric`
keep the symmetric one on the record.

`bounds` belongs to the *value*, not the Design — ODCV runs a rate and a severity score through
the same Design. Declared for ODCV's MR (`0, 100`) and MMLU accuracy (`0, 1`); not for severity.

Measured over 2000 simulated experiments (`scratch/stats/simulate_coverage.py`), logit covers
better in **every** regime, and is narrower mid-range — it buys the coverage where it is needed,
not everywhere:

| regime | symmetric | logit |
|---|---|---|
| base, μ≈0.41 | 93.5% | 94.2% |
| near-zero, μ≈0.045 | 89.8% | **92.8%** |
| one fixed checkpoint, near-zero | 90.7% | **93.7%** |

`difference()` deliberately takes no `bounds`: a difference of rates can be negative and has no
log-odds, and it sits mid-range, where symmetric is already the right shape.

**At the boundary** (μ̂ exactly 0 or 100) every spread is 0, so the symmetric interval collapses
to a *point* — certainty from a finite sample, the worst failure of the three. There is no
log-odds either, so the interval falls back to a Wilson score bound at the draws on the
**smallest sampled axis** (3 seeds beats 40 scenarios: an unobserved seed could still misbehave,
and no number of scenarios rules that out). It uses the normal `z`, not `t`, because Wilson's
level is defined by `z` and the df here is estimated from spreads that are all zero. A clean arm
reports `[0, 56%]` on 3×25, not `[0, 0]`, and `claims` flags the result as degenerate.

## 4. The Design decides which spreads apply

| factor kind | meaning | examples | effect |
|---|---|---|---|
| `item` + `item_sampling="sampled"` | drawn from the benchmark population | scenario, question, prompt | `T_B` term |
| `item` + `item_sampling="fixed"` | the benchmark itself | Matthew's seed SEM; MMLU as-is | no item term |
| `enumerated` | all levels present in every item, fixed weights | ODCV variants (½,½), Arena-Hard orderings | collapsed into the cell; no term; fixes the estimand |
| `subsamples` | draws inside a cell, no identity across cells | rollouts, questions within a subject | averaged in; lives only in `β` |
| `checkpoint` | **not declared** — inferred as sampled iff `n ≥ 2` | seeds | `T_A`, `T_C` terms |

The checkpoint axis is populated automatically for ODCV: `uv run evals --name odcv --target
seed0 seed1 seed2` publishes each arm and then pools them
(`src/eval/misalignment/odcv/pool.py`), each arm entering as one checkpoint, so the pooled
run answers the question the replicates were run for — about the recipe, not about the seed
that happened to run first. Merging their rollouts into a single arm instead would keep the
one-checkpoint claim while shrinking the bar, which is the mistake this table exists to
prevent. Arms that ran different scenario sets or different thinking modes are refused
rather than intersected.

| checkpoints | items | SE² | df |
|---|---|---|---|
| sampled | sampled | `T_A + T_B − T_C` | Satterthwaite |
| sampled | fixed | `T_A` | `n−1` |
| fixed | sampled | `T_B` — Miller's one-model setting | `J−1` |
| fixed | fixed | within-cell rollout noise; **needs R ≥ 2** | Satterthwaite over cells |

Differences between arms pair on every shared axis (items always; checkpoints when the same
ones sit under both conditions) and add terms on the unshared ones.

## 5. Rollouts

Rollouts are never a count in a denominator. With one rollout per cell the interval is
still valid — rollout noise sits inside every spread and is measured with it — but it
cannot be separated, and the cell value is read as the checkpoint's behaviour on that item. With
`R ≥ 2` the within-cell spread estimates `σ_ε²` and its share of the bar is reported
(`Result.within_cell`). The only question `R = 1` cannot support is the both-fixed one, which
raises `NotEstimable`.

ODCV's variants are an `enumerated` factor: the estimand is a story drawn like these,
presented in a fair-coin variant. That coin is in the target but enumerated in the data, so
it contributes no variance — the same reason a stratified sample with known weights has no
between-stratum term.

## 6. Where the bootstrap still earns its place

Statistics with no closed-form SE: Bradley–Terry ratings, medians, ratios.
`stats.cluster_bootstrap` resamples the Design's sampled axes (items, and checkpoints when
sampled), never rollouts or enumerated levels. For a mean it must agree with `interval` to Monte
Carlo error; that agreement is a test.
