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
| GPU rental | **$63.20** |
| Anthropic API (Petri realism + judge) | **$68.89** |
| **total** | **$303.55** |

---

## 2026-07-31 (pm) — Tool-calling 20/80 LoRA: one H100 epoch

> **Correction, 2026-08-03: this $5.73 bought nothing usable.** The run trained
> full-sequence instead of masking the loss to assistant tokens, and its mixture
> did not guarantee a thinking block on every assistant turn (empty
> `<think></think>` where there was no CoT). Both are documented gotchas in
> `CLAUDE.md` — #4 and #2 — and both invalidate the resulting adapter. The arm is
> withdrawn. The spend stays in the running total because it was really spent;
> it is recorded here as waste so the next mixture run budgets for the rerun.
>
> **Lesson:** verify the label mask on a real batch before committing GPU hours —
> count non-`-100` positions and confirm they fall only on assistant spans. A
> mask defect is invisible in the loss curve, which fell 2.753 → 1.057 and looked
> entirely healthy.

**Bought:** the pure-tool-calling arm of the constitution mixture family — a trained LoRA
adapter, its training mixture, and the run record, all published. The missing cell of a
sweep whose other cells already exist.

**Total: $5.73.** No API spend: both source corpora were already generated and published,
so nothing was sampled from any model.

| item | cost | notes |
|---|---|---|
| GPU — NVIDIA H100 80GB HBM3, 1.918 h @ $2.99/h | $5.73 | RunPod Secure Cloud, one pod, no failures |
| OpenRouter / Anthropic | $0.00 | mixture assembled from published data only |

### Unit economics

| metric | value |
|---|---|
| $/GPU-hour | **$2.99** (H100 SXM 80GB, Secure Cloud, High stock) |
| training throughput | **254 tok/s** (1,492,498 tok in 5,889 s) |
| $/1M training tokens | **$3.84** |
| $/optimizer step | $0.045 (126 steps) |
| overhead vs training | 0.28 h of 1.92 h (15%) — bootstrap, 52 GB model download, smoke test |

Useful for the next estimate: **a 1.5M-token, 1-epoch bf16 LoRA on this 27B model costs
about $6 and takes about 2 hours end to end on one H100**, of which 1h38m is training.
The earlier Qwen3.6 run recorded 1h38m for the same token count at seq 2048 on an H100 at
$3.13/h ($8.10) — so the rate is reproducible and the saving here is the cheaper card.

### What was estimated vs what it cost

Estimated **~$10.50** for ~3.5 h; actual **$5.73** for 1.92 h. The estimate padded
bootstrap at 0.75–1.0 h and it took ~0.4 h, and padded 25% contingency that was not
needed. Bootstrap is faster than assumed because the 52 GB download saturates RunPod's
network, not because anything was skipped.

### Notes

- No idle time was bought: the pod was torn down 7 minutes after the adapter was saved,
  once artifacts were verified locally. Publishing happened after teardown, off the clock.
- Teardown verified absence twice (direct 404 + absence from the account listing) and
  reported the two unrelated pods on the shared account as untouched.

---

## 2026-08-01 — Petri constitution dose sweep v2: 672 audits across 4 arms

**Bought:** the powered rerun of the 2026-07-31 null — 28 seeds x 6 epochs x 4 arms,
619 transcripts retained, uniformly re-judged, published to HF with a Visualizer entry.
**Result is a lead, not a result** (dose-40-60 at 16.5% vs base 27.2%, McNemar p = 0.029,
but the pre-specified severity test crosses zero and control false positives rise with
dose). See `LOG.md`.

**Total: $103.45.**

| item | cost | notes |
|---|---|---|
| GPU — A100-SXM4-80GB, ~29 h @ $1.49/h | ~$43.21 | **reconstructed, not metered** — see below |
| Anthropic API — Haiku 4.5 realism grader | $30.95 | **exact**: 0.13M in / 3.66M out / 38.5M cache-read / 6.9M cache-write, read from the eval logs |
| Anthropic API — Sonnet 4.5 uniform re-judge | ~$29.29 | estimated: 619 calls x ~12.3k in / ~700 out, transcript sizes measured |
| Claude subscription — auditor (Sonnet 4.5) | **$0.00** | 45.1M cache-read + 36.6M cache-write + 4.6M output; quota only |

### Unit economics

| metric | value | vs v1 |
|---|---|---|
| $ / audit (all-in) | **$0.154** | $0.48 — 3.1x cheaper |
| $ / audit (API only) | $0.090 | $0.18 |
| Haiku realism $ / audit | $0.046 | $0.093 |
| Sonnet re-judge $ / call | $0.047 | $0.093 |
| GPU-hours / audit | 0.043 | 0.157 |
| compute wall-clock | 568 min for 672 audits | 454 min for 48 |

Both API unit costs halved against v1 for the same models and rubric. That is not a
discount — v2's transcripts are **shorter** (mean 2.0-2.8 target turns vs 3.2), so every
per-transcript call had less to read. Cheaper here is a symptom of *less adversarial
pressure applied*, which is a measurement problem, not a saving. Recorded as such.

### Two caveats on these numbers

1. **The GPU line is reconstructed.** No pod-lifecycle record was written, and RunPod's
   GraphQL API exposes no per-pod historical billing (`billing`, `transactions` and
   `pods(input:)` were all tried; none returns spend for a terminated pod). 29 h is inferred
   from the audit window (2026-07-31 11:58 -> 2026-08-01 16:22 = 28.4 h) at v1's measured
   $1.49/h. **Lesson: write the pod's create/terminate timestamps and rate to disk at
   provision time.** Everything else in this table is measured; this one is arithmetic.
2. **~19 of the 29 GPU hours were idle**, including a ~14 h overnight gap, while the run
   waited on subscription quota windows. Compute was 9.5 h. Idle cost **~$28 — more than
   the compute itself**, and more than the whole v1 run.

### What it bought, and the one call worth revisiting

Keeping the pod warm across quota windows was a deliberate choice (reprovisioning means
re-downloading 55 GB and re-clearing the CUDA-13 machine filter that bit us on 2026-07-30).
For the 1.6 h and 3.3 h gaps that was right. For the **14 h overnight gap it was not**:
~$21 of idle against ~25 min of rebuild (~$0.62) plus the risk of a machine that fails the
CUDA filter. **Rule for next time: tear down above a ~2 h expected gap; hold below it.**

**The subscription remains the dominant lever.** The auditor consumed 4.6M output tokens
against 45.1M cache-reads. On the API that is roughly $70-200 depending on cache behaviour,
against $60.24 of API spend actually incurred — so the subscription carried more than half
the run's notional cost, at 672 audits versus v1's 48.

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
