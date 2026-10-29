# Error bars for eval results

Reference for `src/eval/stats.py`. Replaced the Design / clustered-spread machinery on
2026-10-29; results published before that date carry the older scenario-level intervals.

## 1. The rule

One run of an eval is **one number**: ODCV's misalignment rate over its 80 cells, MASK's
honesty score over its rows, the agentic-misalignment rate over its 18 conditions, Petri's
`concerning >= 4` rate over its 90 briefs. The cells inside a run are the fixed benchmark and
are not independent draws of anything — the mandated and incentivized variants of one ODCV
scenario co-move, the 18 agentic conditions share three scenarios — so **nothing puts a cell
count in a denominator**.

The thing that is replicated is the run. `passes: 3` runs the whole eval three times and
gives three numbers; those are the sample.

| passes | report |
|---|---|
| 1 | the number, **no interval** (nothing to measure the spread with) |
| n ≥ 2 | mean of the n numbers, and a t-interval on n−1 df over them |

At n = 3 the multiplier is t₀.₉₇₅,₂ = 4.30, not 1.96. That is the honest price of three
draws: ±1.96 over three numbers covers 81%, not 95%.

## 2. Shape

A bounded score (a rate in [0, 100], a 0–5 severity, a fraction) gets the t-step on the
**log-odds scale** — `logit_t(values, n_items, scale)`:

    q = value / scale
    z = ln((q + c) / (1 − q + c)),   c = 0.5 / n_items     (so 0 and 100 stay finite)
    z̄ ± t₀.₉₇₅,ₙ₋₁ · s_z / √n,  mapped back through the logistic and rescaled

The point estimate is the plain mean of the values. The interval is asymmetric and cannot
leave the range: three ODCV passes at 26.2, 16.2, 20.0 report 20.8 with 11.2 to 35.6, not
20.8 ± 12.4. `n_items` only sets the continuity correction; it never narrows the bar.

An unbounded value gets the plain step — `t_mean(values)`.

## 3. Per eval

| eval | the number | passes | interval |
|---|---|---|---|
| ODCV | MR over the 40×2 cells (50/50 variant mixture); mean severity | `passes:` in `configs/eval/odcv/*.yaml` | `logit_t` over per-pass MR (scale 100) and severity (scale 5) |
| MASK | honesty score over the rows | `passes:` | `logit_t` over per-pass scores |
| Agentic misalignment | harmful fraction over the 18 conditions | trial k of every condition = sweep k | `logit_t` over per-sweep fractions (scale 1) |
| Petri tenet audit | `concerning >= 4` rate over the 90 briefs | `--passes` (Inspect epochs) | `logit_t` over per-pass rates |
| Colosseum | each measure, per cell | episodes ARE independent draws | `wilson` over episodes for a yes/no measure, `t_mean` for a continuous one; arm contrasts paired on seed |

Per-condition, per-scenario and per-family rates are reported as plain numbers without an
interval: they are not used as estimates of anything on their own.

Every results artifact carries `<metric>`, `<metric>_ci95` (null for one pass),
`<metric>_ci95_lo/hi`, `<metric>_by_pass`, `n_passes` and `ci_method`, so the per-pass
numbers travel with the run and nothing has to be recomputed from rollouts.

## 4. Several training seeds

An artifact is one model. Its interval says how far the eval's number would move on a fresh
**pass** of that model; it says nothing about a fresh **seed**. For a recipe trained three
times, the three artifacts' means are three draws of the recipe's number, and the interval is
a t over those three means (df 2). Pooling the nine passes as if independent understates the
seed term four-fold (the dice part of the variance is shared within a seed). That
cross-artifact step is deliberately not in any eval's own summary; it is done over published
artifacts when a recipe-level number is wanted.

## 5. Standing rules for comparing arms

1. **Three seeds or it cannot be ranked.** Between-seed spread on ODCV has been 1.2–9.4 pp,
   so a single-run gap under ~10 pp is not a difference.
2. **Never rebuild a mixture to reproduce an arm**: copy the published bytes. The builder's
   shuffle depends on the corpus it reads.
