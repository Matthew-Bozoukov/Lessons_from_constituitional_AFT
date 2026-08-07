<!-- ABOUTME: Post-mortem of the failed 2026-08-06/07 swebench_mini full-sweep attempt. -->
<!-- ABOUTME: Failure modes, the signal that detects each one, and the monitor to build next time. -->

# Post-mortem: the swebench_mini full sweep that never finished (2026-08-06 → 2026-08-07)

**Outcome: ~10 instances genuinely completed out of 372 attempted, in ~14 hours, for ~$121.**
Torn down at the user's instruction with 0 instances left running on vast or GCP.

This document exists so the next run does not repeat it. The technical problems are all
**solved and committed**; what killed the run was a combination of (a) three real bugs that each
took hours to isolate, and (b) marketplace infrastructure that failed three separate times in a
way the pipeline could not survive.

Read the **Monitor spec** section at the end before starting another long run. Nearly every hour
lost here would have been caught in minutes by a watcher on four numbers.

---

## What we were trying to do

Full SWE-bench Verified sweep on two LoRAs of Qwen3.6-27B, via the pinned `swebench_mini`
baseline (mini-SWE-agent v2 + official harness):

| Arm | Target | Selection | n |
| --- | --- | --- | --- |
| A | `qwen3.6-27b-lora-table2-only-9284-r64` | `fraction=0.5, seed=0, shard 1 of 2` | 122 |
| B | `qwen3.6-27b-lora-table2-synthdoc-r64` | `fraction=0.5, seed=0, shard 1 of 2` | 122 |
| C | `qwen3.6-27b-lora-table2-synthdoc-r64` | `fraction=0.5, seed=0, shard 0 of 2` | 128 |

Architecture: model served on rented GPUs; rollouts + grading driven from a separate CPU host
running Docker (the harness spawns one container per instance).

---

## The three bugs (all fixed and committed)

### 1. `--server` health probe used `localhost`, but the tunnel binds the Docker bridge

**Symptom.** `run_eval --server` hung for the full 30-minute `_HEALTH_TIMEOUT_S` and then died
reporting a timeout that read exactly like a slow model load. vLLM was healthy the entire time,
had loaded the LoRA, and was answering `/v1/models`.

**Root cause.** `run_eval` binds the SSH tunnel to `172.17.0.1` for every `needs_docker` eval on
linux, so scenario containers can reach the model. The tunnel then listens on that address and
**only** that address — `127.0.0.1` stays closed. Both `_wait_healthy()` and the `base_url`
handed to the eval hardcoded `localhost`.

**Why it survived until now.** Local serving and remote serving for non-docker evals both bind
loopback. This was the first remote-served `swebench_mini` run, so the combination had never
been exercised.

**Fix.** Commit `c204d92`. Executors expose `endpoint_host`; `LocalExec` pins loopback, `SshExec`
follows its `bind`. Five regression tests in `tests/test_vllm_endpoint_host.py`.

**Cost:** ~45 min of idle GPU.

### 2. Prefix caching was blamed on the vLLM version; it was actually KV starvation

**Symptom.** Prefix cache hit rate 0.6%, KV usage 96%, generation throughput collapsed to
10–24 tok/s, requests timing out at litellm's 600s.

**The wrong inference.** vLLM 0.26 warns, for `Qwen3_5ForConditionalGeneration`:
`Mamba cache mode is set to 'align' ... prefix caching ... is currently experimental`. Combined
with a 0.6% hit rate this looks conclusively like "prefix caching is unsupported for this hybrid
Mamba architecture." **It is not.**

**Root cause.** The 0.6% was measured at **96% KV usage**. At that pressure nothing can be
*retained* between agent steps, so every entry is evicted before reuse. Because SWE-bench agents
re-send their whole history each step, a 0% hit rate means re-prefilling the entire 30–80k
context every step — which is itself what pins KV at 96%. Self-sustaining.

**Measured A/B** (same model, same commit, same vLLM 0.26.0, two H100 NVLs concurrently):

| | caching ON, `workers=3` | caching OFF, `workers=2` |
| --- | --- | --- |
| Prefix cache hit rate | **77–94%** | 0.0% |
| Generation throughput | **194–400 tok/s** | 57.7 tok/s |
| GPU KV cache usage | 10–20% | 16.9% |

**Fix.** Commit `c973e19`. **Keep `vllm==0.26.0` pinned — no upgrade is warranted.** The rule is
*give the cache room*: size concurrency so KV stays well under ~50%, and verify the hit rate in
the server log rather than assuming it.

**Cost:** ~3 hours chasing a version problem that did not exist.

### 3. High `workers` is self-defeating on long-context agent workloads

`configs/eval/swebench_mini_verified.yaml` sets `workers: 12`, reasoned from KV headroom. The
reasoning is sound but its arithmetic assumed prefix caching would hold. It does not hold at
high concurrency, and the failure is a **death spiral**, not a gradual slowdown:

```
many workers -> KV saturates -> prefix cache evicted -> every step re-prefills 40k tokens
    -> requests exceed litellm's 600s timeout -> litellm retries
    -> the retry ALSO queues (the abandoned request is never cancelled)
    -> queue grows ~1 request/worker/minute, forever
```

Observed: **8 running + 49 waiting** against `workers=8`; zero completions in 482 seconds.
Once the queue exceeds what is servable inside the timeout, **it never recovers** — only a
server restart clears it.

Measured healthy: `workers=3` → 78% hits; `workers=10` → 90% hits *on a fresh server*;
`workers=8–12` on a loaded server → <1%. **`workers` alone cannot bound concurrency**, because
retries stack on top of it. Use `serving.max_num_seqs` as a hard server-side cap as well.

---

## The three infrastructure failures (not fixable in code — must be monitored)

### A. GPU host died mid-run

The first H100 NVL vanished from the account entirely, ~1h in. On-demand vast instances are not
supposed to be preempted, but hosts go offline. Every in-flight instance failed.

### B. Driver IP was remapped by vast

The CPU driver's public IP changed (`184.145.198.181` → `184.145.198.49`, port unchanged) ~14h
in. This killed every process on it and every in-flight rollout. Note `vastai ssh-url` returned
the **stale** address afterwards while the API's `public_ipaddr` field had the new one — neither
alone is trustworthy, try both.

### C. A rented instance never left `loading`

`nika-gpu-B2` sat in `loading/stopped` for 5 hours while billing normally.

### The amplifier that turned each blip into hours of loss

`mini-SWE-agent` **skips any instance already present in `preds.json`** — including ones recorded
with an empty patch. So when the endpoint died, every in-flight instance was written as
done-with-no-patch and then permanently skipped. Both synthdoc arms burned through their entire
instance list at ~97% infrastructure failure:

```
synthdoc shard 1: 119 InternalServerError,   3 Submitted
synthdoc shard 0: 122 InternalServerError,   6 Submitted
```

**This is recoverable but only if you know to look.** `scratch/`-level tooling written during
this run (`gapfill.py`) classifies exit statuses and drops **only** transport failures
(`InternalServerError`, `Timeout`, `APIConnectionError`, …) from `preds.json` so they re-run.

> **Do not blanket-retry empty patches.** An instance whose agent genuinely ran and produced no
> patch **is a measurement**. Re-rolling it until it succeeds turns pass@1 into best-of-N. The
> filter must be: transport-failure status **AND** no patch.

---

## Process mistakes (mine)

1. **Rented 3 GPUs up front and bootstrapped none.** They idled 5h at $9.34/hr — **~$46 wasted**
   — because the driver died before I set them up. *Rent one, bootstrap it, verify it serves,
   then rent the next.*
2. **Quoted ETAs from arithmetic instead of measurement**, three times, each wrong. Every
   throughput window was contaminated by an unresolved fault. *State a rate only from a clean
   steady-state window.*
3. **Nearly crossed two arms' shards.** Resume directories were selected by `ls -t` timestamp;
   both synthdoc arms sort adjacently, so shard 0 and shard 1 got swapped. Caught by checking
   `selection.json:shard_index` against the launch flag before damage. *Always resolve a resume
   target by reading its `run_meta.json`, never by mtime.*
4. **`pkill -f <pattern>` matched the invoking SSH command** and killed the session mid-cleanup.
   *Put kill patterns in a script file on the remote host.*

---

## Monitor spec — build this before the next long run

A watcher polling every 60s and alerting on any of the following would have caught **every hour
lost above** within minutes. All four signals are cheap to read.

| # | Signal | Where | Alert when | Catches |
| --- | --- | --- | --- | --- |
| 1 | `Prefix cache hit rate` | GPU: `output/serve/vllm.log`, `Engine 000` lines | **< 50%** for 3 consecutive samples | KV starvation / death spiral (bug 2, 3) |
| 2 | `Waiting:` request count | same line | **> 2 × workers** | retry pile-up before it becomes unrecoverable |
| 3 | Completion delta | driver: count `exit_status == Submitted` across trajectories | **0 new in 15 min** while GPUs are up | stalls of any cause |
| 4 | Infra-failure ratio | driver: `InternalServerError`+`Timeout` ÷ total trajectories | **> 20%** | endpoint death, IP remap (failures A, B) |

Plus two liveness checks:

| # | Check | Alert when |
| --- | --- | --- |
| 5 | `vastai show instances` label set | any expected label missing, or status ≠ `running` for >5 min |
| 6 | Driver reachable at its **API-reported** `public_ipaddr` | SSH fails but the instance shows `running` → **IP was remapped**, re-resolve |

**Recommended automatic responses**

- Signal 1 or 2 trips → restart the vLLM server (clears the queue; nothing else does) and drop
  `workers`. Do **not** just wait it out.
- Signal 4 trips → stop the arm, run the gapfill classifier, resume. Otherwise every affected
  instance is silently scored unresolved.
- Signal 5/6 trips → re-resolve addresses, restart arms, gapfill.

---

## Settings to start from next time

```bash
uv run scripts/run_eval.py --target <hf> --name swebench_mini --server <alias> --port <p> \
  subset.fraction=0.5 shard.index=<i> shard.count=2 \
  workers=8 serving.max_num_seqs=12 serving.enable_prefix_caching=true
```

- **Do not change** `max_model_len` (65536) or the litellm timeout. Both are outcome-determining:
  truncation aborts instances and scores them unresolved.
- **Verify before trusting any pass@1**: `scripts/eval/swebench_mini_check_env.py --n 2`
  (gold-patch check) on every fresh grading host.
- **Splitting an arm across two GPUs is safe and proven**: `shard.count=4` indices `{1,3}` is
  bit-identical to `count=2` index 1 (disjoint, balanced 63/61). Verified against
  `subset.shard()`. Merge the two `preds.json` by instance id before grading.
- **Prefer a stable provider for the driver.** The driver is the single point of failure: it
  holds every run dir, all preds, and ~384GB of pulled images. Losing it loses everything
  in flight. It is cheap (~$0.12/hr) — put it somewhere that does not remap IPs.

---

## What was actually achieved (worth keeping)

- **vast.ai VM rental is a blessed grading host** — Docker capability 5/5 and the gold-patch
  check passed 1/1 on harness 4.1.0 (first rentable host to do so). See `docs/LOG.md` 2026-08-06 (2).
- **Prefix caching confirmed working for Qwen3.6-27B on pinned vLLM 0.26** at 90–94% hits — a
  result that matters every time we serve this model. See `docs/LOG.md` 2026-08-07.
- Two real bugs fixed with regression tests (`c204d92`, `c973e19`).
- **GCP characterised**: hard-capped at 12 vCPU project-wide, all 396 compute quotas
  `NOT_ENOUGH_USAGE_HISTORY`, GPU quota 0. See `docs/LOG.md` 2026-08-06.
- **RunPod ruled out for the CPU role**: its June 2026 CPU-pod Docker runtime explicitly removed
  Docker-in-Docker, which the harness requires.
