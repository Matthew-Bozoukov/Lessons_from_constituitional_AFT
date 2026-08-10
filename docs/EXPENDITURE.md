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
| OpenRouter (eval judging) | $0.02 |
| OpenRouter (eval scaffolding/smoke) | $0.13 |
| GPU rental | $0.17 (+ ~4 RunPod A100-h on 2026-08-03, $ TBD from dashboard) |
| GCP (CPU VM) | ~$0.29 (closed — all instances destroyed 2026-08-06, nothing accruing) |
| vast.ai (CPU/VM verification) | ~$0.06 (closed — box destroyed 2026-08-06) |
| vast.ai (swebench full-sweep attempt) | ~$107 (closed 2026-08-07; bought ~10 instances) |
| vast.ai (driver + grader, successful run) | ~$4 (closed — 0 instances) |
| RunPod (7x H100 NVL rollouts) | **~$100** (closed — 0 pods, burn $0.00/hr) |
| **total** | **~$468** |

---

## 2026-08-07 (2) — swebench_mini COMPLETED: 372 rollouts, 250/500 instances, both LoRAs

**What was bought:** the actual result. 369 of 372 rollouts (99.2%), 218 patches, all four
cells graded by the pinned harness, published to HF. See `docs/LOG.md` 2026-08-07.

**Cost**, by provider, from balance deltas (the authority; per-run manifests overstate):

| provider | what | spend |
|---|---|---|
| RunPod | 7x H100 NVL rollouts (peak $22.33/hr), ~5.5 h + setup | **~$100** |
| vast.ai | CPU driver ($0.20/hr) + regrade box ($0.325/hr) | **~$4** |
| | **total this run** | **~$104** |

Final balances: RunPod $233.50 (burn $0.00/hr), vast $104.86. **Zero instances running on
RunPod, vast or GCP** — verified by API on all three.

**Unit costs worth keeping:**

- **~$0.28 per graded SWE-bench instance** all-in (372 rollouts + grading for ~$104).
- **H100 NVL on RunPod Secure: $3.19/hr** actual (vs $2.59 advertised "lowest"). vast.ai listed
  $2.55-2.64/hr but its supply was thin — only 1-2 available at a time.
- **The CPU side is ~4% of cost.** All the money is GPU-hours. The driver that hosts every
  Docker container, all 400GB of images and every result costs **$0.20/hr**.
- **vast.ai CPU-only offers remain unrentable** (`no_such_ask`), re-confirmed today. The cheapest
  Docker-capable box therefore carries a GPU you do not use — ~$0.33/hr instead of the
  advertised ~$0.01/hr. Budget accordingly.

**Waste in this run: small and bounded.** Two GPUs idled ~20 min after their arm finished before
I noticed the supervisor was relaunching completed arms (~$2). One RTX PRO 6000 Blackwell rented
to benchmark and killed within minutes when vLLM 0.26 refused it (`FlashInfer requires sm75+`)
(~$1). Compare with the previous day's ~$46 of never-bootstrapped GPUs: renting one box,
bootstrapping it, verifying it serves, *then* renting the next is what changed.

**Lesson that cost a re-grade:** the driver died after grading, taking the harness reports with
it. Backups covered `preds.json` but not `grading/`. Re-grading from the preserved predictions
cost ~$0.70 and 25 min — cheap, but avoidable. **Back up the artifact you cannot regenerate
cheaply, not just the one you thought of first.**

---

## 2026-08-07 — swebench_mini full sweep: ABANDONED after ~$107 for ~10 instances

**What was bought:** three genuine engineering results and a very expensive lesson in
marketplace reliability. It did **not** buy the benchmark. ~10 instances completed out of 372
attempted, over ~14 h. Full analysis in `docs/swebench_run_postmortem.md`.

**Cost:** vast credit **$163.39 -> $62.83 = ~$100.56** over the run window, plus ~$6 spent before
that reading (verification box, first GPU, early driver hours) — call it **~$107**. Authoritative
figure is the credit delta, not a per-instance manifest. Peak burn reached **$17.30/hr** across
6 GPUs + driver; final teardown verified 0 instances on vast and 0 on GCP.

| what | rate | note |
|---|---|---|
| H100 NVL (vast, on-demand) | $2.55–2.69/hr | the working card; 93GB is what the config's `max_model_len: 65536` assumes |
| H200 (vast) | $3.97/hr | rented, never usefully used |
| VM-rental driver (docker host) | $0.12/hr | cheap, and the single point of failure — see below |

**Wasted spend, itemised — these are the entries worth reading:**

- **~$46: three GPUs rented up front and never bootstrapped.** They idled ~5 h because the
  driver died before I set them up; one never left `loading` while billing normally. **Rent one,
  bootstrap it, verify it serves, then rent the next.** Never pre-rent a fleet.
- **~$45: GPU time burned against a broken pipeline.** Two separate KV-cache death spirals and a
  hardcoded-`localhost` health probe meant the GPUs were up and idle-or-thrashing rather than
  producing rollouts. A monitor on `Prefix cache hit rate` and `Waiting:` would have caught both
  within minutes (spec in the post-mortem).
- **~$3: a GPU host that died mid-run**, and a duplicate instance rented because a success-check
  misparsed the API response (destroyed within minutes, ~$0.30).

**Lesson for future estimates:** the CPU/driver side of this workload is genuinely ~$0.12/hr and
irrelevant to cost. **100% of the money is GPU-hours, and the dominant risk is paying for GPUs
that are not producing rollouts** — through misconfiguration, an unrecoverable request queue, or
simply being rented before they are needed. Budget by *verified* throughput, not by instance
count: quote a rate only from a clean steady-state window, and treat any unmeasured rate as
unknown.

---

## 2026-08-06 (2) — vast.ai VM rental: Docker + gold-check verification

**What was bought:** proof that a vast.ai **VM rental** passes all three host gates — Docker
capability 5/5, `docker_preflight()`, and the gold-patch grading check (1/1 resolved, harness
4.1.0). First rentable host ever blessed for grading; see `docs/LOG.md` 2026-08-06 (2).

**Cost:** one `nika-vast-vm-verify` box (VM-capable GTX 1660 S offer, 5 vCPU / 49GB / 243GB) at
**$0.087/hr** for ~40 min ≈ **$0.06**. Destroyed; only the pre-existing `nika-swebench-arm2`
H100 remains, untouched.

**Unit-cost correction that matters for future estimates:** vast's advertised **CPU-only offers
at ~$0.0102/hr are NOT rentable** (`no_such_ask` on every attempt — see LOG for the full test
matrix). The cheapest *rentable* VM-capable box is **~$0.067–0.087/hr**, because it must carry a
GPU we do not use. Any estimate built on $0.0102/hr for the CPU host is wrong by ~7×. In
absolute terms this is still noise next to the GPU bill (~$2.55/hr), so it does not change the
provider decision — but do not quote the CPU-only figure again.

**Note on concurrent spend:** the `nika-swebench-arm2` H100 PCIE ($1.900/hr) from a separate
workstream was running throughout, 3.1 h and ~$5.94 at first observation and still running at
99.99% GPU utilisation. Not provisioned by this task and deliberately not touched; recorded here
only so the ledger reflects that account-level burn was concurrent.

---

## 2026-08-06 — GCP small CPU VM: Docker viability check for swebench_mini/ODCV

**What was bought:** proof that GCP VMs can run the Docker workloads RunPod cannot — bridge
network creation, container DNS, bind mounts, concurrent containers (5/5 passed; see
`docs/LOG.md` 2026-08-06). Also bought the measurement that `us-central1` quota is 32 vCPU / 8
instances / 2048 GB with 0 in use, so the earlier large-instance refusal was not a zero quota.

**Cost:** ~$0.01 for the first ~30 minutes. **This is a list-price estimate, not a billed
figure** — the GCP billing console is the authority and lags. Unit costs used
(`us-central1`, on-demand list):

| component | $/hr | note |
|---|---|---|
| `e2-small` (2 shared vCPU, 2GB) | ~$0.0168 | |
| 10 GB pd-balanced | ~$0.0014 | $0.10/GB-month |
| ephemeral external IPv4, in use | ~$0.005 | charged whenever attached to a running VM |
| **total** | **~$0.023/hr** | ≈ $0.55/day ≈ $17/month |

Image pulls are inbound and free; egress was negligible.

**Still running.** The instance was deliberately left up to accrue the account usage history
GCP wants before approving a larger CPU instance. It bills continuously at the rate above.
Tear down with `instances.delete` on `nika-healthcheck-01` in `us-central1-a`, then confirm the
aggregated instance list is empty.

**Correction (same day, 13:01 UTC): torn down.** `nika-healthcheck-01` was destroyed after a
total lifetime of **1.33 h ≈ $0.031** at the rate above. Verified clean via the aggregated
lists: 0 instances, 0 disks, 0 addresses, and region quota usage back to 0 CPUS / 0
DISKS_TOTAL_GB / 0 IN_USE_ADDRESSES. The boot disk auto-deleted with the instance
(`autoDelete: true`) and the external IP was ephemeral, so no orphaned billable resources
remain. Nothing is accruing. Total GCP spend for this work item is closed at ~$0.03.

**Lesson:** an `e2-small` is enough to prove Docker *capability* for ~$0.02, which is a cheap
way to qualify a provider before committing to a large box. It proves nothing about *capacity*
— SWE-bench images run to ~300GB, so the real grading host is a separate purchase.

**Second instance, same day.** `nika-swebench-host-01` — `c3d-standard-8` (8 vCPU, 32GB, 200GB
pd-balanced) in `us-central1-c`, stood up as a candidate Docker/grading host. Ran ~0.6 h before
teardown at ~$0.43/hr (c3d-standard-8 ~$0.398 + 200GB pd-balanced ~$0.027 + external IP
~$0.005) ≈ **$0.26**, again a list-price estimate. It reached "docker + uv installed, repo
synced" but the gold-patch check **was not run** — teardown was requested first, so this spend
bought provisioning knowledge, not a blessed grading host.

**Both instances destroyed 2026-08-06.** Verified clean across every zone: 0 instances, 0
disks, 0 addresses, 0 snapshots, 0 images, 0 instance templates, 0 managed instance groups
(nothing that could recreate a VM), and `CPUS_ALL_REGIONS` usage back to 0/12. GCP spend for
this work item is closed at **~$0.29**.

**Capacity lesson worth the money:** `e2-standard-8` and `n2-standard-8` were
`ZONE_RESOURCE_POOL_EXHAUSTED` in **all four** `us-central1` zones; placement succeeded only on
the 6th attempt (`c3d-standard-8` in `us-central1-c`). Budget provisioning *attempts*, not just
instance-hours, and prefer a zone/shape walk over a single hardcoded target.

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
