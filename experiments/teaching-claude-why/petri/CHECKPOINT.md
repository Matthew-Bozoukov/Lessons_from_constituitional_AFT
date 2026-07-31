<!-- ABOUTME: Resume point for the constitution dose sweep - state, exact commands, costs, and the power problem. -->
<!-- ABOUTME: Written 2026-07-31 after epoch 1 completed. Read this before spending anything on epoch 2+. -->

# CHECKPOINT — constitution dose sweep, 2026-07-31

Epoch 1 is **complete, analysed and published**. This file is what a future
session needs to resume without rediscovering anything.

---

## 1. Where things stand

| item | state |
|---|---|
| Grid | 4 arms x 12 seeds x **1 epoch** = 48 audits, 44 retained |
| Result | **Null.** 20% / 20% / 40% / 30% violation frequency at 0/10/20/40% SFT |
| Paired vs base | McNemar exact p = 1, 0.625, 1 |
| Published | [`LASR-Callum/2026-07-31-petri-constitution-dose-sweep`](https://huggingface.co/datasets/LASR-Callum/2026-07-31-petri-constitution-dose-sweep) @ `258608f5` |
| Visualizer | `content/petri-runs/2026-07-31-constitution-dose-sweep`, loads from HF |
| GPU | terminated, verified absent |
| Spend | $22.91 ($14.26 GPU + $8.65 API) |
| Branch | `petri-constitution-audit` |

Local logs (gitignored, **not** reproducible without a re-run):
`logs/grid-e1/<arm>/*.eval`, `output/rejudged/*.json`, `output/analysis/`.

---

## 2. THE POWER PROBLEM — read before buying more epochs

More epochs is a **bad buy**, and the reason is not sample size.

### Measured power, paired McNemar, base 20%

| epochs | n/arm | power vs 8 pp effect | vs 15 pp |
|---|---|---|---|
| 1 (ran) | 10 | ~0% | ~1% |
| 9 | 90 | **10.6%** | 47% |
| 25 | 250 | — | **93%** |
| 90 | 900 | **89%** | — |

Eight more epochs costs roughly **$104-139** and reaches about **11% power**
against an ODCV-scale effect. It would most likely produce a second null that
still cannot be interpreted.

### Why the pairing is weak

The paired standard deviations are enormous relative to the effect:

| arm | n pairs | mean diff in severity | sd | effect size d |
|---|---|---|---|---|
| dose-10-90 | 10 | -0.20 | 4.04 | -0.05 |
| dose-20-80 | 10 | +0.80 | 3.19 | +0.25 |
| dose-40-60 | 10 | +0.00 | 2.93 | 0.00 |

**Because the auditor authors a new scenario for every (arm, seed) pair, the
arms are not actually matched.** "Paired on seed" pairs on the *archetype*, not
on the situation - so scenario variance, the dominant noise source, stays in the
comparison. Buying epochs averages that noise down at the slow 1/sqrt(n) rate
instead of removing it.

The sibling investigation reached the same conclusion independently: Petri
"cannot produce valid controls - because the auditor authors the scenario and
the judge is never told which arm it is scoring", and a fixed, non-adaptive
design "is the right instrument for attribution, which adaptive audits cannot
support" (`vulnerabilities/docs/22-petri-run-mechanics.md`).

### The cheaper, better instrument

A **fixed-probe eval** - byte-identical prompts across all four arms - removes
scenario variance entirely instead of averaging it. The same money buys far more
power, and the sibling experiment already has the pattern in
`vulnerabilities/seeds/fixed-eval/probes.json`, where the same judge reached
**kappa 1.000** on a fixed task versus a 57% false-positive rate on adaptive
audits. Judging is not unreliable; task shape is what differs.

**Recommendation: do not buy 8 adaptive epochs. Spend a fraction of it on a
fixed-probe eval across the same four arms, and keep the adaptive run as the
qualitative, hypothesis-generating half.**

---

## 3. Costs, measured not estimated

Per-audit, from the epoch-1 grid (48 audits, concurrency 4):

| quantity | measured |
|---|---|
| wall clock | **6.9 min/audit** (26.5 min at concurrency 1) |
| GPU | 0.115 h/audit = $0.171 |
| Haiku realism | 148,970 tok/audit = **$0.093** |
| Sonnet judge | ~22k in / 1.8k out = **$0.093** |
| subscription | 211,772 tok/audit (**$0 cash**, plan quota) |

### 8 more epochs (384 audits, n -> 90/arm)

| line | concurrency 4 (proven) | concurrency 8 (unvalidated) |
|---|---|---|
| GPU wall clock | **45.6 h** | ~23.6 h |
| GPU cost | **$67.95** | $35.09 |
| API (realism + judge) | **$71.46** | $71.46 |
| **total** | **~$139** | **~$107** |
| subscription quota | **81M tokens** | 81M tokens |

Both exceed the current caps: `MAX_GPU_SPEND_USD=40` and the $50 API ceiling.
45.6 h also exceeds `MAX_WALL_CLOCK_HOURS=36`, so a single run cannot legally
finish under the watchdog - it would need splitting or the cap raised.

**81M subscription tokens is the largest unknown.** Epoch 1 used 10M and did not
trip a limit; 81M over ~2 days may.

---

## 4. Exact resume steps

Prerequisites: `claude setup-token` done (User-scope `CLAUDE_CODE_OAUTH_TOKEN`),
Anthropic API funded, RunPod funded, `.env` HF token (the one with repo-create
scope - the `~/.config/msm-audit/infra.env` token does **not** have it).

```powershell
# 0. Repo
cd C:\Users\nikak\source\repos\LASR\teaching_claude_why_replication
git checkout petri-constitution-audit

# 1. Watchdog FIRST, before any paid resource
cd experiments\vulnerabilities
.\scripts\provider\Start-Monitoring.ps1 -Gpu 'NVIDIA A100-SXM4-80GB' -HourlyUsd 1.49

# 2. Provision. -AllowedCudaVersions is NOT optional: vLLM 0.26's only wheel
#    targets CUDA 13, and a driver-570 pod dies at engine init.
.\scripts\provider\New-AuditPod.ps1 -Name nika-petri-constitution `
    -VolumeInGb 120 -AllowedCudaVersions @('13.0')

# 3. Umbrella keeper for the whole run (per-arm keepers start automatically)
.\scripts\provider\Start-HeartbeatKeeper.ps1 -Activity 'petri-grid-umbrella' `
    -MaxHours 12 -IntervalSeconds 180 -LeaseMinutes 20

# 4. Bootstrap + serve (each ~15 and ~25 min; run detached, they outlive the call)
.\scripts\remote\Invoke-Remote.ps1 -ScriptFile '..\teaching-claude-why\petri\scripts\bootstrap_arms.sh' -RemoteName 'bootstrap_arms.sh' -Detach
.\scripts\remote\Invoke-Remote.ps1 -ScriptFile '..\teaching-claude-why\petri\scripts\serve_arms.sh' -RemoteName 'serve_arms.sh' -Detach
#    wait for "all four arms answer tool-bearing requests" in
#    /workspace/logs/serve_arms.sh.out before continuing

# 5. Tunnel
cd ..\teaching-claude-why\petri
.\scripts\Start-PetriTunnel.ps1 -Verify

# 6. Grid. USE A NEW TAG per epoch batch so logs never collide.
.\scripts\Run-Grid.ps1 -Epochs 8 -MaxConnections 4 -Tag grid-e2

# 7. Judge uniformly on the API (epoch 1 was judged this way; do not mix)
.\..\..\vulnerabilities\scripts\secrets\Invoke-WithPetriSecrets.ps1 `
  -FilePath .venv\Scripts\python.exe `
  -ArgumentList @('scripts\rejudge.py','--logs','logs\grid-e2','--out','output\rejudged-e2')

# 8. Teardown IMMEDIATELY after the grid - do not analyse on a live GPU
cd ..\..\vulnerabilities
.\scripts\secrets\Invoke-WithInfraSecrets.ps1 -ScriptBlock { & .\scripts\provider\Stop-AuditRun.ps1 }
#    then VERIFY BY HAND - the account-sweep step has a property-shape bug and
#    reports FAIL even on a clean account.

# 9. Analyse across BOTH epoch batches, plot, export, publish
cd ..\teaching-claude-why\petri
.venv\Scripts\python.exe scripts\analyse.py --rejudged output\rejudged-combined --out output\analysis
.venv\Scripts\python.exe scripts\plot_violation_curve.py --results output\analysis\results.json --out output\analysis
.venv\Scripts\python.exe scripts\build_export.py --logs logs\grid-all --rejudged output\rejudged-combined --analysis output\analysis --out exports\<date>-<slug>
.venv\Scripts\python.exe scripts\build_manifest.py --export exports\<date>-<slug>
.venv\Scripts\python.exe scripts\publish_hf.py LASR-Callum/<repo>
```

**Combining epochs:** `analyse.py` pairs on `(sample_id, epoch)`. Epoch numbers
restart at 1 in each run, so merging `output/rejudged` and `output/rejudged-e2`
requires renumbering the second batch's `epoch` field (offset by 1) or the pairs
will collide. **This is not yet implemented** - it is the one code change a
top-up run needs.

---

## 5. Gotchas that cost time or money (all already fixed in the scripts)

| trap | what happens | fix in place |
|---|---|---|
| No heartbeat keeper | watchdog reaps a healthy pod mid-run | keeper wired into the runner with `finally` stand-down |
| Driver 570 | vLLM 0.26 dies: "driver too old (found version 12080)" | `-AllowedCudaVersions @('13.0')`, verified before downloading |
| Zero API balance | `Test-Credentials` passes; realism fails at first paid call; a whole arm produces empty transcripts | real paid call as a preflight |
| Sample count != valid audit | 12 transcripts, target never participated | `check_arm.py` gates every arm on target model events |
| `max_tokens` too low | base arm returns `finish=length` with ZERO content, faking a dose-response | 4096, measured (peak 1493) |
| 200GB volume | pod sits `RUNNING` with `runtime: null` | 120GB |
| Stale tunnel | guard saw a tunnel for a dead pod and refused to start a live one | compares recorded port against the current pod |
| CLI turn limit | ~23% of in-run judge calls die on long transcripts, unevenly across arms | judge on the API, or `rejudge.py` afterwards |
| Orphaned `claude.exe` | ~16 accumulate per run | harmless; reboot clears them |

---

## 6. Things a future session should NOT redo

Already verified, no need to re-establish:

- vLLM LoRA works on this hybrid Mamba/linear-attention arch: 512 adapter
  tensors -> 256 modules (MLPs on all 64 layers, attn on the 16 full-attention
  layers), matching the model cards.
- All three adapters' `chat_template.jinja` are byte-identical to base
  (sha `e84f32a2...`, 6 tool refs) - the HTTP-400 tool trap does not apply here.
- Adapter tokenizers differ from base only by merges re-serialisation and 7
  unused audio tokens; tokenization verified byte-identical, so one tokenizer
  serves all arms.
- The judge's 24,205-char structured schema round-trips through the subscription
  CLI, and the rubric discriminates (planted P2 violation scored 7/10 while
  seven other dimensions stayed at 1).
- Serving all four arms from one vLLM process is both cheaper and cleaner than
  merging.
