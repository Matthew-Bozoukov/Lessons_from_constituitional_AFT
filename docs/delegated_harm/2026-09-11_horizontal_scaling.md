# Live horizontal scaling of recovery episodes

The user authorized four additional H200s (two per adapter) and requested that future
workers support live coordination. The experimental queue is `scratch/delegated_harm/episode_queue.py`;
the delegated-harm worker reads it between episodes. GPU servers still only serve
inference. The local worker drivers receive control through the shared queue, then
send requests to their owned GPU through the existing SSH tunnel.

At the handoff there were eight active control episodes and eight active DA episodes.
Their ownership stays with the existing H100 drivers. The remaining 126 control and
61 DA episodes were assigned to the new queue, independently of response outcomes.
All frozen requests, model/base revision pins, episode IDs, decoding seeds and
recovery context/output/turn settings are reused. H200 hardware and worker identity
are recorded per new episode. Original 445 scored completions remain unchanged.

The original drivers had already submitted every episode to an in-process executor
and cannot receive a new queue command. This one-time compatibility handoff therefore
allows those drivers to drain the pre-handoff episodes. They may also start overlap
while draining; those unassigned attempts are excluded based on the frozen ownership
manifest, never based on whether they succeeded. Once every retained episode has a
saved final record, the coordinator verifies the original launcher's process identity,
stops that owned process tree, and terminates its exact pod. It never interrupts a
retained active episode. Unassigned legacy traces are retained separately for audit.

New workers claim one episode at a time using SQLite transactions. The primary key
is adapter arm plus episode ID. Finishing requires the same owner that claimed it.
A heartbeat timeout never silently transfers an in-flight episode to another worker.
Workers write to separate directories; one coordinator merges the recorded owners'
results, verifies their hashes, restores original completed observations, scores only
new completions, and generates distinct recovery charts.

Operational commands from the isolated worktree:

```powershell
# Add a worker while the batch is already running.
uv run scratch/delegated_harm/scale_live.py add --root output/delegated_harm/2026-09-11_horizontal_recovery --arm control --count 1

# Inspect claims, pending work, worker modes, and heartbeat times.
uv run scratch/delegated_harm/scale_live.py status --root output/delegated_harm/2026-09-11_horizontal_recovery

# Finish a worker's current episodes and stop taking new ones; its launcher frees its GPU.
uv run scratch/delegated_harm/scale_live.py drain --root output/delegated_harm/2026-09-11_horizontal_recovery --worker control-h200-1

# Pause or resume new claims globally; in-flight episodes are preserved.
uv run scratch/delegated_harm/scale_live.py pause --root output/delegated_harm/2026-09-11_horizontal_recovery
uv run scratch/delegated_harm/scale_live.py resume --root output/delegated_harm/2026-09-11_horizontal_recovery
```

Use this queue pattern for future long evaluation batches that may need more workers;
do not pre-submit the entire batch to a fixed executor. This implementation is wired
into delegated harm; other eval runners need an explicit integration before claiming
that they support live scaling. The local coordinator remains a host dependency:
process crashes leave durable claims for inspection, not automatic duplicate retries.

All rentals use `src/infra/runpod.py` and the normal evaluation serving entrypoint.
Each H200 was quoted at $4.59/hour and has a 2.5-hour independent watchdog and cleanup.
Workers never publish competing HF results; publication belongs to the coordinator.
The old two-controller chart finisher was stopped and replaced by the new coordinator.

Validation: 17 offline tests passed, including concurrent exclusive claims, owner-only
completion, arm isolation, pause/drain behaviour, and the prior context-budget tests.
Live readiness and actual claim progress are checked separately after GPU bootstrap.
