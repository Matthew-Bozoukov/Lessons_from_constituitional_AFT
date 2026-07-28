---
title: "Live provider monitor and cleanup watchdog"
date: 2026-07-28
summary: "Monitoring and cleanup enforcement built and drill-tested before any paid GPU exists; all three watchdog triggers fire and terminate verifiably."
status: complete
---

# Live provider monitor and cleanup watchdog

Both were built and tested **before** provisioning any paid resource.

## Design: two independent processes

| | Provider monitor | Cleanup watchdog |
| --- | --- | --- |
| Script | `runtime/provider-monitor/Monitor-Loop.ps1` | `runtime/watchdog/Watchdog-Loop.ps1` |
| Interval | 240 s (inside the 5-minute ceiling) | 60 s |
| Job | observe and publish | enforce and terminate |
| Terminates anything? | **no** | **yes** |

They are separate processes deliberately: a monitor crash must not leave a GPU
running, and a watchdog crash must not blind the operator. Each receives its own
copy of `RUNPOD_API_KEY` directly in its environment block via
`Start-DetachedProcessWithSecretEnv`, so the key never becomes a variable in the
launching shell and never appears on a command line.

## Status artifacts

All under `runtime/provider-monitor/`:

| Artifact | Purpose |
| --- | --- |
| `status.json` | machine-readable current status, every figure labelled |
| `status.md` | human-readable current status |
| `status-history.jsonl` | append-only, one object per refresh |
| `run-state.json` | run identity, instance, hourly price, caps, deadline |
| `monitor.pid` | monitor process state |

Refreshes happen every 240 s and at every event boundary through
`scripts/provider/Update-ProviderStatus.ps1 -Reason <event>`: `post-provision`,
`post-model-download`, `post-server-start`, `post-audit`, `pre-cleanup`,
`post-cleanup`.

### Every number carries a basis

Required fields are published with one of four labels, and nothing is invented:

- `exact-provider-reported` - read from the provider's authenticated API
- `locally-calculated` - derived from provider figures plus local clocks
- `estimated` - derived from an assumption that may not hold
- `unavailable` - not exposed; reported as unavailable rather than guessed

If the balance API fails, the balance is published as `unavailable`, the compact
line prints `provider balance unavailable`, and the last *successful* refresh
timestamp is retained rather than overwritten. The locally-calculated experiment
GPU budget keeps being displayed regardless, since it does not depend on the
provider.

Compact line format:

```text
[GPU ACTIVE] runpod | NVIDIA A100 80GB PCIe | $1.19/h | provider balance $108.11 | experiment GPU budget $34.87 remaining | elapsed 01:12 | vLLM healthy
```

## Watchdog triggers

| # | Trigger | Condition |
| --- | --- | --- |
| 1 | `hard-wall-clock-deadline` | `now > run-state.hard_deadline_utc` (created_at + 36 h) |
| 2 | `gpu-budget-exhausted` | locally-calculated cost >= `MAX_GPU_SPEND_USD` ($40) |
| 3 | `idle-timeout` | no declared activity for `IDLE_SHUTDOWN_MINUTES` (30) |

### Why trigger 3 does not kill live work

Slow work must never be mistaken for inactivity. Activity is *declared* through
`runtime/watchdog/heartbeat.json`, written by
`scripts/provider/Send-AuditHeartbeat.ps1`. A heartbeat carries a timestamp and
an optional **lease**:

```powershell
# A long operation takes a lease covering its expected duration.
.\Send-AuditHeartbeat.ps1 -Activity 'model-download' -BusyMinutes 90
```

Idle time is counted only when **every lease has expired AND the heartbeat has
gone stale**. Model downloads, model loading, Petri calls, inference and report
generation all hold leases, so none of them can be killed as idle.

Fail-safe direction: if `heartbeat.json` is missing or unreadable, the idle rule
is *suspended* rather than triggered, because a missing file means the
orchestration never started reporting rather than that it went idle. Triggers 1
and 2 are never suspended.

## Drill results

Run against a deliberately non-existent pod ID, so the drills cost nothing and
could not affect real infrastructure. Evidence retained under
`evidence/cleanup/DRILL-watchdog-termination-*.json`.

| Drill | Setup | Expected | Result |
| --- | --- | --- | --- |
| Hard deadline | deadline 5 min in the past | terminate | **fired**; `verified_absent: true`, account sweep `active_pod_count: 0`, final balance recorded |
| Idle, lease held | heartbeat 55 min stale, lease active 60 min | do **not** terminate | **held**; logged `activity lease held until ... by 'model-download'` |
| Idle, no lease | heartbeat 55 min stale, no lease, limit 30 min | terminate | **fired**; `run-state.instance_state` became `terminated-by-watchdog` |

The hard-deadline drill produced this evidence, confirming the full path
terminate -> verify -> sweep account -> record balance:

```json
"termination_result": { "action": "terminated", "verified_absent": true,
                        "detail": ["attempt 1: REST DELETE returned 404, pod already absent"] },
"account_sweep":      { "active_pod_count": 0, "clear": true },
"final_balance_usd":  108.1129952401, "final_balance_basis": "exact-provider-reported"
```

### One observed behaviour worth recording

A drill run **without** the API key present stalled at status `terminating` for
about 25 s before reporting. This is correct rather than a defect: with no
credential each of the 5 termination attempts fails, and `Get-RunPodPod` returns
`unknown` rather than `null`, so the pod cannot be *verified* absent. The
watchdog then escalates to status `TERMINATION-UNVERIFIED` with instructions to
check the provider console, which is the right outcome when cleanup genuinely
cannot be confirmed. Re-run with the credential present, the same drill
completed and wrote full evidence.

## Idempotent by design

`Stop-RunPodPodHard` treats an already-absent pod as success (HTTP 404 ->
`verified_absent: true`). The watchdog and the normal shutdown path can therefore
both call it without racing. It tries REST `DELETE` first and falls back to the
GraphQL `podTerminate` mutation, retrying up to 5 times with verification between
attempts, and only reports success once a follow-up read confirms the pod is
gone.

## Raw artifacts

- [status.json](../runtime/provider-monitor/status.json)
- [status.md](../runtime/provider-monitor/status.md)
- [run-state.json](../runtime/provider-monitor/run-state.json)
- [Drill evidence](../evidence/cleanup/)
