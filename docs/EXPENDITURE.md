<!-- ABOUTME: Running ledger of real money spent on this project - API credit and GPU rental. -->
<!-- ABOUTME: Append a new dated section per work item; never rewrite history, correct with a follow-up line. -->

# Expenditure ledger

Every entry records **what was bought, what it cost, and what it produced**. Costs that bought
nothing (failed runs, wasted spend) are recorded too — those are the entries with the most value
to a future reader.

**Conventions**
- One dated section per work item, most recent first.
- Always give a unit cost (`$/1k tokens`, `$/document`, `$/GPU-hour`) so future estimates have a base.
- OpenRouter's `/credits` endpoint lags several minutes; wait ~30 s before a final reading or it
  under-reports. Read it before *and* after a run — per-run manifests include cached replays and
  therefore overstate cash spent.
- Record failures and their cost explicitly.

## Running total

| category | spent to date |
|---|---|
| OpenRouter (data generation) | **$255.96** |
| OpenRouter (eval judging) | $16.54 |
| OpenRouter (eval scaffolding/smoke) | $0.14 |
| GPU rental | $19.78 (+ ~4 RunPod A100-h on 2026-08-03, $ TBD from dashboard) |
| **total** | **$292.42** (+ GPU TBD) |

<!-- Total recomputed once at the jamie/write-all-evals-to-hf <- main merge (2026-08-06),
     not carried over from either side: both branches advanced the same running total
     independently from $256.15 at the fork. This branch added $16.52 judging + $19.61 GPU
     (psychosis, table2 arms); main added the $0.13 scaffolding/smoke category (SWE-bench
     wiring). 255.96 + 16.54 + 0.13 + 19.78 = 292.41. Both source sections are kept below. -->

---

## 2026-08-06 — Mixture pipeline smoke, spec-filter judge calls (OpenRouter, gpt-5.6-terra)

**What was bought:** two `--smoke` runs of the integrated staged mixture build
(`configs/data/mixture/qwen36_msm_table2.yaml`) — 12 judge calls on the first (the smoke cap
at the time), 3 on the second (cap lowered on request; the rerun verified the
BatchEncoding token-count fix). 15 calls total, each ~4.4k-token constitution system prompt
(cached after call 1) + one sample, ≤900 output tokens at `reasoning_effort: low`.
**Cost:** ~**$0.01** (unit cost from the 2026-08-04 full filter pass: ~$0.0004/call with
prompt caching; 15 × $0.0004 ≈ $0.006, rounded up). Booked under eval scaffolding/smoke.
**What it produced:** a working end-to-end rehearsal and a caught bug — the first run's
stats showed 2 "tokens" per row, exposing that `apply_chat_template(tokenize=True)` returns
a BatchEncoding whose `len()` is its key count, so the `max_seq_len` cap never fired.
**Lesson:** a smoke run that prints its stats pays for itself; also state the exact call
budget before running anything that spends.
**Note:** the full filter pass this rehearses costs ~10,000 calls ≈ $4–5 — flag before
launching.

---

## 2026-08-05 — SWE-bench baseline wiring smoke (OpenRouter, Gemini 3 Flash)

**What was bought:** end-to-end validation of the new `swebench_mini` eval on 2 SWE-bench
Verified instances, with `google/gemini-3-flash-preview` standing in for a served target (no
GPU). Two runs: the first bought two findings and no rollouts (every instance died on
mini-SWE-agent's 120s container-start timeout, which cannot cover a cold multi-GB image pull —
now fixed by a pre-pull step; and grading died on a repo-relative `--project` path resolved
against a changed cwd). The second produced real rollouts: 2 trajectories, 25 steps each,
**tool-call rate 100%**, no patches because the smoke's reduced step limit cut them off.

**Cost:** $0.13 by `/credits` delta (601.161 → 601.292 of $900). Waited before the final read.

**Unit cost:** ~$0.066 per instance at 25 steps, ~$0.0026 per step, for a Flash-class model on
a django SWE-bench task. Useful only for scaffolding work — the real baseline runs against our
own vLLM endpoint and costs GPU-hours, not API credit.

**Also learned (free, but it shapes the budget):** SWE-bench images run ~1.15 GB each for
django instances (2.31 GB for two). And the official grading harness **cannot run on Windows
at all** — it imports the Unix-only `resource` module at import time — so grading must happen
on Linux regardless of Docker Desktop. That makes a cheap vast.ai CPU box (~$0.01/hr) the
grading host, which is a rounding error against GPU time.

---

## 2026-08-05 — psychosis eval, table2 arms (2× RunPod H100 + Grok via OpenRouter)

First live psychosis runs: `table2-synthdoc-r64` (20% DA / 80% benign) and `table2-only-9284-r64`
(benign-only control, 0% DA).

- **GPU: ≈$17.11.** Two H100 80GB SECURE pods, ~2.47h + ~2.73h @ $3.29/hr (rate read off pod
  B's listing; pod A assumed same class — confirm on the RunPod dashboard). ≈**$3.29/GPU-hour**.
  Roughly half the wall-clock was consumed by two aborted rounds (below), not the science.
- **OpenRouter: $16.47** (credits 614.32 → 630.79; final read taken >30s after the run). Of
  this, ≈$10 bought the two clean runs — 13 persona arcs + 156 graded cells ≈ **$0.60 per
  persona-arc** (grok-4.5 red-teamer + grok-4.3 judge, ~12+12 calls per arc) — and ≈$6.5 was
  burned by aborted rounds: upstream's pinned grok-3/grok-4 discovered dead only at run time,
  a context overflow at turn ~7 (16384 serving window vs preserved-reasoning turns), and a
  trace-parse leak that invalidated one completed arm (rerun). Lessons: probe config-pinned
  API models for liveness AND task-compliance before booting GPUs; smoke one persona
  end-to-end before the full fan-out.
- **Follow-up, same day: reasoning-parser verification pod: ≈$2.60.** One more H100
  (~45 min @ $3.29/hr ≈ $2.50) + $0.05 OpenRouter (1-persona smoke) to verify vLLM 0.26's
  `qwen3` parser splits Qwen3.6's prefilled think stream server-side (it does — trace
  arrives out-of-band in `reasoning`, content clean). Included in the running totals
  (GPU $17.28 → $19.78, A100 TBD unchanged; judging $16.49 → $16.54; total → $292.28).

## Running total

| category | spent to date |
|---|---|
| OpenRouter (data generation) | **$172.76** |
| OpenRouter (eval judging) | $0.02 |
| GPU rental | $0.00 (+ ~4 RunPod A100-h on 2026-08-03, $ TBD from dashboard) |
| **total** | **$172.78** (+ GPU TBD) |

---

---

## 2026-08-04 — MEM pipeline smoke validation (OpenRouter, Sonnet 5)

**What was bought:** end-to-end validation of the new `synthdoc mem` pipeline — one MEM smoke
(2 control + 2 m4 documents, $0.22), one `synthdoc check` pass with real judge calls (~$0.06),
and two failed `synthdoc run --smoke` attempts (~$0.17, bought a finding: trait t1 generates
CBRN-adjacent scenarios that Bedrock content-filters at stage 4, so the 2-item smoke cannot pass —
see LOG 2026-08-04).

**Cost:** $0.45 by `/credits` delta (578.457 → 578.907 of $600). Waited 30 s before the final read.

**Unit costs (measured, the numbers that matter):** MEM control $0.046/doc (11,968 in / 2,184 out
tokens per call), MEM critique $0.064/doc (12,062 in / 3,958 out) at Sonnet 5 $2/$10 per 1M.
Prompts are ~12k tokens because the full constitution + a whole transcript are injected — 70%
above the pre-smoke assumption. Pilot at 300+300 = **$32.84** (`synthdoc estimate --measured`);
**remaining credit $21.09 cannot cover it** — flag raised, top-up needed before the pilot.

**Follow-up, same day (self-reflection pass):** five-cell smoke (10 docs incl. 4 perturbations,
$0.82) + full checks with flaw-ID judge (~$0.06): **$0.85** by `/credits` delta (578.907 → 579.755).
New measured unit costs: reflect $0.059/doc (12,282 in / 3,470 out), critique re-measured
$0.074/doc (12,104 in / 4,937 out — m3 critiques run longer than m4's), perturb $0.018/doc
(2,571 in / 1,325 out). **Full 5×300 matrix = $104.84** ($0.070/doc all-in). Account topped up to
$800 → **$220.25 remaining**; the matrix is affordable but >$20, flagged for sign-off before
running.

---

## 2026-08-03 — Eval-framework pod validation (RunPod A100-80GB + OpenRouter)

**What was bought:** the full validation matrix for the new eval framework — Option A
(pod-driven) and Option B (Mac-driven, `--server`) internalization smokes against
`LASR-Callum/qwen3.6-27b-synthdocv2-lora-20_80`, a 2-step training smoke (TRL under
transformers 5.14), Qwen3.6-27B serving proven on vLLM 0.26, plus four bugs caught live
(inline-nohup SSH hang, smoke-slice thinking validation, remote `.env` sourcing, RunPod
docker verdict).

- **GPU:** ~4 A100-80GB pod-hours across three pod incarnations on Jamie's RunPod account
  (two pods died/restarted mid-session). Rate not visible from this machine — **estimate
  ~$2/GPU-h ≈ $8; correct this line from the RunPod dashboard.** Unit cost worth recording:
  cold engine boot for Qwen3.6 on vLLM 0.26 is ~14 min of billed GPU doing no eval work
  (mostly kernel compile + CUDA-graph capture); warm boots are several times faster —
  batch evals per pod session, don't boot per eval.
- **OpenRouter (judging):** $0.01 + $0.00 self-reported by the two 4-item internalization
  smokes (gemini-flash judge). The before/after `/credits` discipline was skipped for these
  micro-runs; resume it for real runs.
- **Wasted spend:** one engine boot (~14 min GPU) lost to the inline-nohup SSH hang, and
  one to the first training-smoke validation misfire — both bought the bug fixes above.

---

---

## 2026-07-29 — Approved-constitution SFT corpus (synthdoc)

**Bought:** 1,443 synthetic training documents, **1,531,369 Qwen3 tokens**, generated from
`docs/claude_approved_constitution.md` via the `synthdoc` pipeline. Sized to match the v1 corpus
(1.52M tokens) so the two are directly comparable.

**Total: $171.46** (OpenRouter, `anthropic/claude-sonnet-4.5` throughout).

**Artifact:** `LASR-Callum/synthdoc-approved-constitution-sft` on HuggingFace (private), plus
`data/sft_approved_constitution.jsonl` locally. All uploads SHA-verified against local copies.

The repo also carries the **full lineage** (`runs/<name>/`, per-stage snapshots with the 607
dropped documents retained and their verdicts) and the **LLM call cache**
(`cache/synthdoc_cache.tar.gz`, 9,470 entries / 13 MB compressed). The cache is this $171.46
made replayable: extract it to `output/synthdoc_cache/` and re-filtering, re-exporting, or
diffing stages costs **$0**. Only genuinely new work hits the API. Push the cache alongside any
future corpus for the same reason.

### By phase

| phase | cost | what it bought |
|---|---|---|
| calibration & tuning | $10.88 | ~10 small runs; found four real bugs before they scaled |
| production generation | $92.39 | plan + `draft_then_align` on 1,660 documents |
| recovery after API key disabled mid-run | $40.75 | `values_deliberation` + rating on ~1,350 docs |
| top-up to 1.5M tokens | $27.44 | 288 extra documents |

### By corpus (final run manifests)

Manifest figures **include cached replays**, so they exceed cash spent — the sum below is $160.75
against $171.46 actually charged, and the difference is calibration runs whose outputs were
overwritten.

| run | manifest | docs kept | generated |
|---|---|---|---|
| `approved_difficult_advice` | $48.59 | 528 | 700 |
| `approved_embodied` | $48.17 | 476 | 700 |
| `approved_agentic` | $35.20 | 151 | 260 |
| `approved_difficult_advice_top` | $13.84 | 141 | 190 |
| `approved_embodied_top` | $14.49 | 147 | 200 |

### Unit economics

| metric | this run | v1 (`difficult_advice_gen.py`) |
|---|---|---|
| $ / 1k tokens | **$0.112** | $0.049 |
| $ / kept document | **$0.119** | ~$0.035 |
| model calls / document | 5 (plan, draft, align, deliberate, rate) | ~3 |

**Why 2.3× v1:** each document is written **three times** — draft → align →
values-deliberation rewrite. Output tokens at $15/M are **79% of spend** (measured: 1.99M input /
1.48M output on corpus A, i.e. $5.96 input vs $22.16 output). Generating fewer times is the only
large lever; prompt caching touches only the 21% input share.

Per-stage split (corpus C, per document): planning $0.011 · generation $0.023 ·
values_deliberation $0.017 · rating $0.008.

### What went wrong, and what it cost

| incident | cost | lesson |
|---|---|---|
| **API key disabled mid-run** | ~$40 to recover | Generation completed but `values_deliberation` and rating returned 401s → `autorater_overall = 0.0` → below `min_score` → 1,266 documents dropped. Snapshots survived, so recovery only re-paid the missing stages. |
| `generation.max_tokens: 4096` too low for agentic docs | ~$2 | Truncated JSON mid-object → documents arrived **empty**, not shorter. Raised to 9000. Capping it *lower* (2600) made it far worse. |
| `export.mix: {}` | $0 | Deep-merges with the parent, so `pretrain_shard: 0.4` stayed in force and 40% of documents never reached the SFT file. Only **`recipe`** mixtures replace. Must write `{pretrain_shard: 0.0}`. |
| Haiku 4.5 swap for planning + rating | ~$3 | **Net loss.** Rating on Haiku cut corpus C from 11/12 to 5/12; Haiku *planning* cut it to 7/12 with Sonnet scoring the results 2.0 where it had scored 5.0. Cheaper model degraded the scenarios, not just the grades. Reverted. |
| `n_raters: 3` | ~$8 across runs | `autorater_std = 0.0` on **every** document — three raters returned identical scores. Dropped to 1. |

### Things that saved money

- **Call cache** (`output/synthdoc_cache`, 8k entries / 61 MB): corpus B's full re-run cost **$0.00**
  and corpus A's export rebuild cost **$0.38**. Without it the recovery would have been ~$90 more.
- **Calibrating at n=12 before scaling** — every bug above was found for a few dollars.

### Traps worth knowing

- **`budget_usd` counts cumulative cost including cached replays**, not new spend. A $6 cap aborted
  instantly against $31 of work it was merely replaying. It is not a guard on incremental spend;
  track the credits endpoint instead.
- **Keep rates measured at n=12 do not extrapolate.** A and C were 92% at n=12 and ~75% at n=700 —
  dedup and the autorater have more to reject as a corpus grows.

### Not yet spent

GPU rental for QLoRA (~$10–20) and the honeypot eval judge (~$15–25) are still ahead.

---

---

## Prior work (pre-ledger, from README/LOG)

Recorded for comparison; these predate this ledger and are taken from `README.md` and `LOG.md`.

| item | cost |
|---|---|
| v1 difficult-advice generation (1.52M tokens) | ~$46 |
| v1 think-trace augmentation | ~$28 |
| ODCV-Bench replication (80 runs, 4 judges) | $17.84 |
| ODCV-Bench on the DPO adapter | $11.32 |

---

## 2026-08-04 — synthdoc self_reflection corpus: 592 records, 1.56M tokens (~$83.20)

**Bought:** the `self_reflection` SFT corpus — **592 records, 1,555,017 Qwen3.6 tokens** — on HF
`LASR-Callum/2026-08-03-synthdoc-self-reflection`. Unit cost **$0.141 per record**, **$0.0535 per 1k
tokens**.

| leg | outcome | cost |
|---|---|---|
| smoke runs (4) | pipeline validation + measured per-stage tokens | ~$1.80 |
| base run, attempt 1 | stages 2-3 complete, stage 4 aborted on the failure guard | ~$27.54 (reconstructed) |
| base run, resume 1 | aborted immediately — guard bug, below | ~$0.94 (reconstructed) |
| base run, resume 2 | completed: 451 records | **$35.25** (manifest) |
| top-up (`total_scenarios=144,id_prefix=b`) | completed: 141 records | **$19.47** (manifest) |

$54.72 is exact from manifests; ~$28.48 is reconstructed from measured per-call costs for the leg
whose manifest was overwritten on resume. Against the pre-run estimate of $0.1226/record the
completed legs landed within 1% — the estimator is trustworthy; the aborts were the cost.

**Wasted: ~$28.50.** Both aborts were the failure guard, not the pipeline. Sonnet 5 on Bedrock
returns `finish_reason=content_filter` on a small fraction of these environments outright — a
per-scenario refusal, not a fault — which tripped the 2.0% default at 2.8%. Raising it exposed a
worse bug: on resume the rate was measured against the *remaining* items, precisely the ones that
had already failed, so 12/13 read as 92.3% and aborted again. Per-item checkpointing meant the spend
still bought records, so the genuine waste is only the ~$1.95 of re-attempted refusals.

**Biggest saving, reusable anywhere:** extended-thinking tokens bill as completion tokens. With
thinking on, the refine stage burned ~8,800 completion tokens to emit a ~1,200-token environment —
7x its visible output. Setting `reasoning: {enabled: false}` on the two stages that assemble text
rather than judge it cut the projected full run from $198.73 to $117.67 at the original sizing: an
**$81 saving for a two-line config change**. Check this on any pipeline that puts a reasoning model
on a mechanical stage.

**Balance note:** OpenRouter `/credits` read `total_usage` $410.92 of $600 at the time. That figure
covers the whole shared account, not this repository, so it cannot be used to derive per-task spend
— it is recorded only as a runway check. Take a `/credits` reading **before and after** a run when a
true incremental figure is wanted; none was taken here, which is why one leg had to be reconstructed.

---

---

## 2026-08-04 — Qwen3.6-27B think-token probe (RunPod A100-80GB)

**What was bought:** ground truth on Qwen3.6-27B's token-level think-tag behavior, for eval
metric design: greedy token-by-token dump on a trivial question, thinking on and off
(`scratch/qwen3_empty_think_tokens.py` launched by `scratch/launch_qwen36_probe.sh`; output in
`output/logs/qwen36_think_probe_20260804_154008.txt`, findings in LOG.md 2026-08-04).

- **GPU:** A100-80GB PCIe secure-cloud pod `80noxz67x08net`, 15:32–15:40 (~7 min) at
  $1.39/GPU-h ≈ **$0.17**. Unit cost: one single-question 27B transformers probe (55GB Xet
  download ~3 min + shard load + 2 greedy generations) ≈ **$0.20/probe-pod**; the marginal
  question is nearly free, so batch questions into one pod.
