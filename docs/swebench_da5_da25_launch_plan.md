<!-- ABOUTME: Pinned adapters and proposed shared-fleet settings for the September 24 DA pair. -->
<!-- ABOUTME: Preparation only: no evaluation or GPU rental has started. -->

# DA-5 and DA-25 pre-launch plan

Prepared from main `51925bdff180c92518e018f956b7ec0a13a5ee2a` on isolated branch
`codex/swebench-da5-da25`. The user requested this plan before launch.

| Arm | HF model | Exact revision | DA supervised tokens | DA rows |
| --- | --- | --- | --- | --- |
| DA-5 | dougalldeepmind/2026-09-24-qwen36-0-da-5 | 6633d50d2514a4fd791f58b360fb89c0f98a3eae | 243,098 / 4,842,083 (5.02%) | 208 / 9,724 (2.14%) |
| DA-25 | dougalldeepmind/2026-09-24-qwen36-0-da-25 | 92f7ba88c392d9052f660e4ece072aa4d6f82ed5 | 1,210,612 / 4,842,014 (25.00%) | 1,021 / 8,563 (11.92%) |

Both have published adapter weights and matching rank-64/alpha-128, seed-0,
one-epoch thinking-mode training metadata. Base:
`Qwen/Qwen3.6-27B@6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
Pinned mixtures are `dougalldeepmind/2026-09-24-da-5-mix` at
`3fedbcfefaa75aba5e14f4e66bdc4baba1c44c60` and
`dougalldeepmind/2026-09-24-da-25-mix` at
`0bff2f432ebb0a4016d174774372b4bab6e59f52`. Both use DA source
`dougalldeepmind/2026-09-24-da-synth@4f4ca43bac4eace1aee0fed25df4175ad9299399`.

- Both production preflights passed; inference service inactive and zero owned
  SWE-bench GPUs. Vast CPU 52229744 is running, persistent, at $0.5388889/hour.
  Qualified resources: 61 logical CPUs, 197.9 GiB RAM, 485 GiB filesystem and all
  300 task images cached.
- Twenty GPUs TOTAL, four conversations each, initially ten GPUs per arm.
  Longest-first queues; every ready pod starts independently. Prefer H100 NVL,
  then delayed H200 and RTX PRO 6000 Blackwell Server fallback. Drain each pod
  before swapping adapters while retaining its base server. One shared
  80-conversation ceiling and 32-command CPU gate.
- All 300 tasks per arm. Existing context, response-token, task-token and step
  limits remain. No 90-minute wall-clock task limit. Emergency GPU leases are
  at most six hours, shortened by budget; idle pods terminate immediately.
- Proposed shared cap: $300, within the standing $360 maximum. Account balance
  observed $339.59, with one unrelated pod billing $3.519/hour. Do not touch it.
  Observed GPU quotes: H100 NVL $3.19/hour; H200 $4.59/hour; RTX PRO 6000
  $2.09/hour. Refresh balance, quotes and inventory at actual launch.
- Historical replay estimate: about 100 minutes startup/inference and $102 GPU
  for both models, roughly two hours and $105 including active CPU time after
  grading/publication. H200-heavy fallback may cost around $145-150 GPU if task
  durations hold constant. These are estimates, not live paired-run measurements.
- One cumulative ledger, separate outcomes and HF repositories. Durable local
  checkpoints and asynchronous HF retries; infrastructure retries only, never
  reroll completed model failures. After GPU teardown, grade each arm with 12
  workers. Require 300 valid and graded per arm, artifact verification and empty
  owned GPU inventory before completion. Keep the persistent CPU running.
- Create one verified 15-minute progress/completion/failure monitor at actual
  launch. No monitor is needed while this plan awaits the user's review.

Planned result repositories, both checked absent:
- dougalldeepmind/2026-09-24-swebench-qwen36-0-da-5
- dougalldeepmind/2026-09-24-swebench-qwen36-0-da-25

Use the paired entrypoint in `docs/swebench_lite_runbook.md`, substituting the
two exact revisions above and the proposed $300 shared cap. No new calibration
or control model is required. The code has 92 passing regression checks and a
GPU-free real-agent/Docker/grader/HF smoke test; the actual pair will provide
the first live shared-fleet adapter-switching and throughput evidence.
