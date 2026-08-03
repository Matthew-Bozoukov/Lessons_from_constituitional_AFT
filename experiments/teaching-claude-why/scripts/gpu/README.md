<!-- ABOUTME: How the tool-calling SFT run rents, uses and releases a GPU without touching anyone else's. -->
<!-- ABOUTME: Read this before running anything in this directory - the isolation is the point. -->

# GPU scripts for the tool-calling SFT arm

These deliberately duplicate rather than reuse the sibling experiment's provider
machinery (`experiments/vulnerabilities/scripts/provider/`). The duplication buys one
property that matters more than the shared code:

> **Nothing here ever *terminates* a pod other than the one it created, and nothing here
> writes to state another task owns.**

The RunPod account is shared. On 2026-07-31 it held five to six pods belonging to other
people, burning ~$15/h between them.

**The hazard is shared state, not termination.** `New-AuditPod.ps1` and
`Start-Monitoring.ps1` write a fixed
`experiments/vulnerabilities/runtime/provider-monitor/run-state.json`. Running either
while another task holds that file repoints that task's watchdog at this pod. That is a
real, observed failure: on 2026-07-31 starting monitoring twice in one session left two
orphaned monitor loops polling the provider for ~19h against an already-terminated pod.

> [!NOTE]
> **Correction.** An earlier version of this file claimed `Stop-AuditRun.ps1`'s account
> sweep "terminates teammates' GPUs." That is false, and was verified false by reading the
> code: `Stop-AuditRun.ps1` calls `Stop-RunPodPodHard` exactly once, on its own
> `$podId`; `Watchdog-Loop.ps1` likewise terminates only `$RunState.instance_id`; and the
> sweep is `Test-RunPodNoActivePods`, which only calls `Get-RunPodPods` (a GET) and
> returns counts and ids for reporting. The sibling machinery is safe on this account.
> The reason to keep a separate copy is the shared state file above, not termination
> behaviour.

`AGENTS.md` says never to terminate a resource this repository did not provision. That
still governs everything here — it just is not what the sibling scripts were doing wrong.

## The files

| file | what |
| --- | --- |
| `New-ToolcallingPod.ps1` | provisions one H100 80GB SXM, writes `runtime/toolcalling/run-state.json` in the same breath as creation |
| `Watchdog-Toolcalling.ps1` | independent local process; terminates **this pod id** on hard deadline or budget cap |
| `Stop-ToolcallingRun.ps1` | planned teardown; verifies absence by direct lookup of the id, records evidence, stands the watchdog down |
| `Invoke-ToolcallingRemote.ps1` | scp+ssh script runner reading *this* run's endpoint file |
| `bootstrap_toolcalling.sh` | pinned training stack + the 54GB base model |
| `collect_toolcalling_artifacts.sh` | gathers logs/state/config for publication |

## Order of operations

```bash
pwsh scripts/gpu/New-ToolcallingPod.ps1          # 1. provision + register
pwsh scripts/gpu/Watchdog-Toolcalling.ps1        # 2. arm, in its own background process
# 3. bootstrap, smoke, train via Invoke-ToolcallingRemote.ps1
pwsh scripts/gpu/Stop-ToolcallingRun.ps1         # 4. terminate + verify absent
```

Step 2 must not be skipped. The watchdog is what guarantees the pod dies if the
orchestration does.

## Deliberate omissions

**No idle-shutdown trigger.** The sibling watchdog has one, driven by a heartbeat file
with activity leases. That only works if something keeps taking leases; a long unattended
training run does not, and a mis-fire would kill the run mid-epoch. The hard deadline and
the budget ceiling are sufficient and cannot mis-fire onto someone else's GPU.

**A read-only account sweep at teardown, kept deliberately.** It terminates nothing; it
counts pods and reports whether this run's id is among them. It is the check that proves
nothing was stranded, and skipping it would remove the only evidence that teardown worked
rather than merely appeared to. Other users' pods will show in its count — that is
expected on this account and is reported, not acted on.

**No credential on the box.** `Qwen/Qwen3.6-27B` is public, so the bootstrap unsets
`HF_TOKEN` and downloads anonymously. Nothing secret is written to rented disk. The
adapter is pulled back and published from the local machine instead.

## Gotchas hit while writing these

1. The secrets wrapper runs its scriptblock **in-process**, so `$using:` is a parse
   error — plain closure variables are correct.
2. `pgrep -f train_lora.py` over ssh matches the ssh command line itself. Use the bracket
   trick (`train_lor[a].py`).
3. `WANDB_DISABLED` does not stop the TRL callback. Install wandb and set
   `WANDB_MODE=offline`.
4. The trainer's `--smoke` takes the *first* 8 rows, which in this mixture are mostly
   short TULU3 examples. To probe the sequence-length memory ceiling you must build a
   file of the longest rows and smoke on that instead.
5. `causal-conv1d` / `flash-linear-attention` are not installed and should not be — they
   only help the transformers path and cost an hour of failed builds.
