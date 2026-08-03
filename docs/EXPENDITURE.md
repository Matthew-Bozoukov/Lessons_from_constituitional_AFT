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

---

## Running total

| category | spent to date |
|---|---|
| OpenRouter (data generation) | **$171.46** |
| GPU rental | **$5.21** |
| **total** | **$176.67** |

Caveat on the GPU line: it was $0.00 before 2026-08-03 even though `LOG.md` records many earlier GPU
runs, so this figure is *not* the project's true GPU spend to date — it is only what has been
recorded here. Earlier runs went unlogged. Treat it as a floor.

---

## 2026-08-03 — Tool-calling 80/20 arm, corrected retrain (RunPod H100)

**Bought:** one 1×H100 80GB pod (RunPod `ft5p3ydj4z8202`, SECURE, US-MO-1) at **$2.99/GPU-hour**,
run **1.74 h** end to end. **Total: $5.21.**

| phase | wall clock | cost |
|---|---:|---:|
| provision + SSH up | 0.07 h | $0.20 |
| install stack, pull base model (52 GB) + agentic corpus | 0.13 h | $0.39 |
| build mixture (TULU3 streaming, 1.19 M replay tokens) | 0.04 h | $0.12 |
| mask gate (two passes: one caught a bug in the gate itself) | 0.03 h | $0.09 |
| **training** — 112 steps, 1 epoch, 4096 ctx | **1.38 h** | **$4.13** |
| publish to HF + teardown | 0.09 h | $0.28 |

**Unit costs for future estimates:**
- **$3.00 per 1M training tokens** at 1 epoch / 4096 ctx / batch 1×16 on one H100 (1.497 M tokens,
  $4.13 of GPU time). LoRA r=32 on a 27B bf16 base.
- **~$0.037 per training step** (~40.6 s/step) at 4096 ctx. The sibling 2048-ctx arm ran ~47 s/step,
  so **doubling the context did not cost more per step** — most sequences are far shorter than the
  cap and batch size is 1, so there is no padding waste. Budget by token count, not by context.
- **$0.59 of unavoidable setup** before any useful work (provision + 52 GB model pull). Any run
  shorter than ~15 min of real compute is mostly setup.

**Against budget:** ~$6–8 approved, **$5.21 actual**. Came in under because setup was much faster
than the reference run (3.6 Gbps box, 12 min from create to training-ready vs the ~25 min assumed).

**Wasted spend: ~$0.05.** One training launch crashed immediately — `report_to: ["wandb"]` with
wandb not installed; `WANDB_MODE=disabled` does not help, because the callback needs the *package*
present regardless. Caught in under a minute by the log monitor. **Lesson: install `wandb` on any
box running `train_lora.py` even when you intend to disable it**, or the run dies at trainer
construction after the model is already loaded.

**Produced:** [`LASR-Callum/nika-sft-tulu-toolcall-80-20`](https://huggingface.co/LASR-Callum/nika-sft-tulu-toolcall-80-20)
(adapter) and [`LASR-Callum/2026-08-03-tulu-toolcall-80-20-mixture`](https://huggingface.co/datasets/LASR-Callum/2026-08-03-tulu-toolcall-80-20-mixture)
(training mixture), both private, both carded. Pod terminated and verified gone (404).

**Not yet bought:** evaluation. This arm has no misalignment or capability numbers; budget a separate
eval run.

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

## Prior work (pre-ledger, from README/LOG)

Recorded for comparison; these predate this ledger and are taken from `README.md` and `LOG.md`.

| item | cost |
|---|---|
| v1 difficult-advice generation (1.52M tokens) | ~$46 |
| v1 think-trace augmentation | ~$28 |
| ODCV-Bench replication (80 runs, 4 judges) | $17.84 |
| ODCV-Bench on the DPO adapter | $11.32 |
