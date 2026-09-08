<!-- ABOUTME: Explicit pod deadlines and eval-owned teardown using the standard CLIs. -->
<!-- ABOUTME: Documents local watchdog limits and live API ownership checks. -->

# Pod lifecycle

```bash
uv run runpod up --name <you>-eval --eval <hf> --max_hours 4 --push_env
uv run evals --name mask --target <hf> --server <printed-address> --terminate-pod
```

`up` starts a detached local deadline watchdog immediately after provisioning,
before waiting for SSH. `max_hours` defaults to six and must be finite and positive.
The deadline includes setup time. The provisioning process can exit normally without
terminating the pod. Failed setup tears it down immediately.

`--terminate-pod` explicitly grants the eval ownership. SSH aliases are resolved using
`ssh -G`; the live RunPod API must identify exactly one matching public IP and SSH port.
Pods must carry the repository marker and deadline that `up` writes into their provider
environment (`LASR_POD_OWNER`, `LASR_POD_DEADLINE`). These are generated metadata, not
credentials or user configuration. Older/unmarked pods and proxy/jump hosts are refused.
No local address registry is maintained. Do not opt into ownership of a shared server.

The eval adds a process watchdog, checking its PID and birth time every 30 seconds.
This is a second guard, not a handoff: the provisioning deadline guard remains active.
Both enforce the original deadline. Eval completion or failure terminates the pod after
the existing publication flow exits, reports remaining account pods and balance, then
stops its process guard. The deadline guard exits when it observes removal.

Without `--terminate-pod`, eval never acquires pod ownership. Manual cleanup remains
`uv run runpod down --pod <id>`. Other account resources are reported, not terminated.

Both guards run locally: keep the machine awake and connected. Neither detects idle
GPUs or guarantees cloud-side enforcement during a local outage. A deadline can stop
an unfinished job, including publication. The retired `managed` and `moralbench` aliases
are replaced by the standard `runpod` and `evals` commands.
