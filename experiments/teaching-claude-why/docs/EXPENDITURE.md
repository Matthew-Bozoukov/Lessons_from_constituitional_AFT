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
| GPU rental | **$14.26** |
| Anthropic API (Petri realism + judge) | **$8.65** |
| **total** | **$194.37** |

---

## 2026-07-31 — Petri constitution dose sweep: 48 audits across 4 arms

**Bought:** an adaptive Petri audit of four Qwen3.6-27B arms against the v1 constitution —
48 audits, 44 retained, a three-panel dose-response figure, and a published-shaped export.
**Result was null** (no dose-response; see `LOG.md`), which is what the money bought and is
worth recording as such.

**Total: $22.91.**

| item | cost | notes |
|---|---|---|
| GPU — A100-SXM4-80GB, 7.56 h @ $1.49/h | $11.26 | the successful pod |
| GPU — two failed pods | ~$3.00 | see incidents |
| Anthropic API — Haiku realism (grid) | $4.47 | 7.15M tokens, measured |
| Anthropic API — Sonnet uniform re-judge | ~$4.18 | 45 judge calls, ~22k in / 1.8k out each |
| Claude subscription (auditor + judge in-run) | **$0.00** | notional $44.26 — this is the saving |

### Unit economics

| metric | value |
|---|---|
| $ / audit (all-in) | **$0.48** |
| $ / audit (API only) | $0.18 |
| GPU-hours / audit | 0.157 |
| wall-clock / audit | 26.5 min at concurrency 1; **~7.3 min at concurrency 4** |
| subscription tokens / audit | ~212k (notional $0.92) |

**The subscription is the dominant lever.** Running auditor+judge on the API would have
cost ~$44 more for the same 48 audits — roughly tripling the bill. Conversely, moving the
*realism* role off the subscription to Haiku roughly halved wall-clock for $4.47, which
was worth it: GPU time is billed by the hour and the run is latency-bound, so $4.47 of API
bought back more than that in GPU.

### What went wrong, and what it cost

| incident | cost | lesson |
|---|---|---|
| **Watchdog reaped a healthy pod** mid-pilot | ~$2.30 + 90 min | `New-AuditPod` issues a fixed 30-min lease; nothing renewed it. `Start-HeartbeatKeeper` existed for exactly this and was never wired in. Worse, I misdiagnosed the symptom as a slow harness for an hour before checking whether the machine was alive. **Check the resource before theorising about the software.** |
| **vLLM 0.26 on a CUDA 12.8 pod** | ~$0.75 | Its only published wheel targets CUDA 13; no cu128 build exists. RunPod's machine allocation was silently deciding whether the run worked. Fixed with an `-AllowedCudaVersions` filter — and verified the driver *before* paying for a 55GB download. |
| **Anthropic API at zero balance** | 26 min + 12 wasted audits | `Test-Credentials` hits a read-only endpoint and passes on an empty account. The realism role then failed at its first paid call and the whole arm produced complete-looking transcripts with no target participation. The runner now makes a real paid call as a preflight. |
| **A 200GB volume that would not attach** | $0.25 | Pod sat `RUNNING` with `runtime: null` for 16 min. 120GB attached immediately. Ample for 56GB of weights. |
| **Missing `openai` / `anthropic` packages** | $0 | Both caught in <10 s by the runner's stdout capture, before any GPU work. Cheap because the preflights fail loudly. |

### Things that saved money

- **`check_arm.py`** stopped the grid the moment an arm produced empty transcripts. Without
  it, arms 3 and 4 would each have burned ~an hour against a dead credential — ~$3 of GPU
  and, far worse, a published chart built on nothing.
- **Serving all four arms from one vLLM process** via `--enable-lora`: one 55GB load instead
  of four, saving ~75 min of GPU (~$1.90) *and* removing serving-stack variance between arms.
- **Concurrency 4**: 26.5 min/audit -> ~7.3 min. The GPU was idle at 0% between calls, so
  this was free.

### Traps worth knowing

- **A token cap is not a performance knob when the target thinks before answering.** At
  `max_tokens=700` the base arm spent its whole budget reasoning and returned **zero
  content**, while tuned arms answered fine. Scoring that would have produced a clean
  dose-response curve made entirely of truncation. Measured 4096 as sufficient (peak 1493).
- **A valid credential is not a funded one**, and **a completed sample is not a valid audit**.

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
