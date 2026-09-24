<!-- ABOUTME: Measured CPU, memory and Docker storage demand for the first Lite control campaign. -->
<!-- ABOUTME: Capacity recommendations distinguish observed use, configured limits and untested scaling. -->

# SWE-bench Lite CPU capacity audit, 2026-09-23

## CPU-only qualification update, 2026-09-24

The user authorized testing higher CPU concurrency without inference GPUs, while
keeping four conversations per GPU. The existing host ran cached base-image tests
from Django, SymPy, sklearn, matplotlib, Astropy and pytest. Separate Python agent
processes imported the pinned real agent stack to account for process overhead.
Two complete load exercises passed. The final exercise used the production tool
admission lock; its 526 commands produced no failures and no cgroup OOM kills.

| Conversations | Simultaneous tools | Phase time | Median command execution | Minimum available RAM |
|---|---:|---:|---:|---:|
| 40 | 40 | 50.05 s | 4.25 s | 185.96 GiB |
| 60 | 60 | 54.12 s | 6.53 s | 182.03 GiB |
| 80 | 80 | 56.11 s | 8.01 s | 178.29 GiB |
| 80 | 32 | 72.15 s | 3.55 s | 177.58 GiB |

Each concurrent phase runs two commands per task, so phase durations across counts
are not equal-work throughput comparisons. With 80 conversations/32 tools, p95
command execution was 22.69 seconds and maximum admission wait 19.59 seconds.
Waiting is recorded separately and does not consume the command execution timeout.
Agent startup causes CPU spikes; these are not steady inference-cycle utilization.

The selected policy is **80 conversations, 32 admitted commands, 12 grading workers**
on this host class. A second CPU is unnecessary for this tested workload. This is
not exhaustive qualification of every possible model-generated test suite. The
four-GiB per-container cap remains a limit rather than a reservation; global memory
and disk guards, bounded descendants and OOM evidence still matter. Raw evidence is
under `metadata/cpu-qualification` in the Sep24 infrastructure HF repository linked
from the runbook. The original exploratory audit follows unchanged below.

The current 40-agent ceiling is conservative policy, not measured hardware
capacity. Do not interpret each container's 4 GiB memory / two-CPU cap as a
reservation or constant use. The previous 320 GiB estimate for 80 agents summed
all memory caps; it was a worst-case allowance, not a measured requirement.
No running evaluation settings were changed for this read-only audit.

Evidence is in `metadata/cpu-capacity-audit-20260923.json` in
https://huggingface.co/datasets/dougalldeepmind/2026-09-23-swebench-qwen36-0-nosynth.

## Measurements

Host: 61 guest logical CPUs, 197.93 GiB RAM, no swap, 500 GB nominal rented disk.
Do not interpret guest-reported physical CPU topology as dedicated physical cores.

161 completed tasks have final cgroup resource records, all started after the
thread/descendant resource fix. Container lifetime peak memory was:

| Statistic | Peak memory |
|---|---:|
| Median | 75 MiB |
| 90th percentile | 183 MiB |
| 95th percentile | 224 MiB |
| Largest observed | 632 MiB |

These records have zero OOM kills. Coverage is incomplete: only 11 Django tasks
and no Astropy tasks are represented, versus 66 SymPy tasks. Earlier historical
Django OOMs are recorded separately in the resource migration and remain a
limitation; the low post-fix peaks do not erase them. The sample also excludes
infrastructure-invalid attempts without a final resource file.

The eight live Python agent processes outside the containers use approximately
169-171 MiB proportional resident memory each. Container numbers alone therefore
understate the total cost per agent. The duration-weighted container CPU use in
the 161 records is 0.037 CPU cores per conversation, including GPU waiting time.
This is not the CPU requirement while a test is actively executing.

Historical telemetry was grouped into 30-second buckets across replicas. In 50
post-fix buckets with approximately 36-40 simultaneous agents, host CPU averaged
5.0%, the largest bucket average was 7.9%, and minimum available RAM was 184.23
GiB. Active counts were reconstructed from attempt timestamps and can include
bookkeeping delays. Sampling and averaging can miss short bursts. These data
cover about 25 sampled minutes, not a continuous stress qualification.

Docker uses native `overlay2`. Its disk report shows 301 images (the benchmark
plus the HTTP fixture) using 183.9 GB, and nine running containers' writable
layers using 52.27 MB at inspection. The filesystem as a whole used 196 GiB with
289 GiB available. Images live on disk; their full sizes are not allocated as
RAM for every container. Immutable layers are shared within a host, and only
modified files need writable-layer storage. We are not copying all 300 images
per agent. Large image caches are distinct from live memory demand.

## 40, 60 or 80 agents on the current host

- **40:** exercised, with substantial measured host headroom. It does not guarantee
  every possible model-generated command fits its individual container limit.
- **60:** a reasonable next qualification target on the existing host; the data do
  not justify buying another host solely to reach this count.
- **80:** plausible on this host under the observed workload, but not yet validated.
  It is wrong to assert either that 320 GiB is necessary or that 80 is equally safe
  without checking bursts, tool latency and workload changes.

Simple linear extrapolation of the observed approximately 40-agent period gives
about 7.5% host CPU at 60 and 10% at 80, and less than 30 GiB non-available memory
at 80. This is illustrative only: faster GPUs increase tool-call arrival rates,
different adapters choose different commands, and synchronized long tests can
produce much larger bursts. There is no swap safety net.

The preflight currently refuses more than 40 agents, sums all 4 GiB limits, and
limits summed CPU quotas to 1.5 times logical CPU count. Those are implementation
policies. Raising concurrency requires reviewing these checks, not just changing
the replica count. Do not reduce the per-task 4 GiB cap just to satisfy arithmetic.

Before freezing a higher concurrency, perform a bounded CPU-only replay of
representative and heavy tool commands in disposable containers at 40/60/80
concurrency, outside the live evaluation. Include agent-process overhead and
check cgroup OOMs, host memory/CPU/I/O pressure, high-percentile tool latency and
timeout incidence. Such a test uses no model inference and must not replace model
outcomes or be presented as an evaluation score. Keep grading's separate worker
limit: 80 waiting conversations does not justify 80 simultaneous grading suites.

If burst contention appears, separate conversation concurrency from tool-command
concurrency. A host-wide tool admission limit can keep many GPUs supplied without
allowing every test suite to run simultaneously. Queue wait must be outside the
command's execution timeout and recorded separately. This is a proposed change,
not implemented protection; it also needs to account for lingering processes.

## Two hosts versus one larger host

Two hosts with all 300 images cached can execute any task independently. Each
needs its own Docker daemon, pinned cache and readiness checks. A proper shared
queue needs transactional leases, expiry, stale-worker fencing and idempotent
result acceptance. Each host owns its containers and GPU connections; one
coordinator owns the budget, canonical results and HF publication. Do not simply
point two existing coordinators at the same JSON state or HF destination.

Static disjoint 150-task shards are simpler but can leave one host idle behind
the other's long tail. A shared queue balances better. Full caches make reassignment
possible; they do not recover the lost host's unsaved sandbox or make the central
coordinator highly available. Two physical machines are needed for failure isolation.

| Choice | Benefits | Costs/limitations |
|---|---|---|
| Existing host, qualified higher concurrency | No cache transfer or distributed queue; cheapest | Higher concurrency still needs qualification; one failure domain |
| Larger single host | One cache, queue and publisher; simpler operations | Potential price premium; still one failure domain |
| Two hosts, 40 agents each | More aggregate capacity; workers on surviving host continue | Duplicate cache/storage; coordination/failover work; extra bootstrap |

A read-only Vast search around 22:36 UTC found a similar VM-capable offer at
$0.571/hour including 500 GB disk (advertised 80 effective CPUs / 257,798 MB RAM),
and a larger high-reliability offer at $2.822/hour including disk (advertised 128
CPUs / 515,593 MB RAM). Existing host cost is approximately $0.539/hour including
disk. Existing plus the similar offer would be approximately $1.11/hour before
transfers, versus $2.82/hour for that particular large offer. These are transient
offers, not rentals or guaranteed guest allocations. The current VM already
demonstrated that advertised RAM can exceed what the guest receives.

Prefer qualifying the existing host before either purchase. If measured host
pressure warrants expansion, choose between a larger host's operational simplicity
and two hosts' capacity/failure isolation based on fresh offers. The duplicated
image cache is about 184 GB per host by Docker's current accounting; allow space
for environments, outputs and writable layers rather than renting exactly that.

References: [Docker resource limits](https://docs.docker.com/engine/containers/resource_constraints/)
and [Docker OverlayFS storage](https://docs.docker.com/engine/storage/drivers/overlayfs-driver/).
