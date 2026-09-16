<!-- ABOUTME: Approved three-pass ODCV evaluation of refreshed low stakes and original nonmoral with new replay. -->
<!-- ABOUTME: Freeze model identities, continuous-server sampling, costs, and verification before rental. -->

# Three-pass control evaluations, 2026-09-16

The user authorized three fresh sequential ODCV passes per adapter, with one
single-H100 pod per adapter and the two pods running in parallel. The combined
ceiling is $60 including GPU rental, storage, judging and recovery. No training or
dataset changes are included.

| Arm | Model and pinned revision |
|---|---|
| Refreshed low stakes + September 8 replay | `dougalldeepmind/2026-09-15-qwen36-0-da-lowstakes-refresh-7` @ `095a9874a1ce54ab1faaa3a99d63642e6199c591` |
| Original 684 nonmoral + September 8 replay | `dougalldeepmind/2026-09-15-qwen36-0-nonmoral-original-7` @ `42232b52b52ed93245864548f37a3fe8179c7d75` |

Each arm has 40 scenarios in both variants, three passes, 240 planned rollouts.
Temperature is 0.7 for the evaluated model; thinking is on; context is 28,000.
Gemini 3 Flash scores both misalignment and progress at temperature 0, pinned to
google-ai-studio. Scenario concurrency remains six per model, twelve combined,
with four judge workers per model. Earlier one-pass observations are not pooled
into the three fresh passes.

The same vLLM process stays alive throughout each arm's three passes. The existing
harness sends no per-request inference seed. The server's startup seed and the
training seed do not mean that every request resets its sampling generator.
The vLLM 0.26.0 source distinguishes explicitly seeded request generators from
unseeded sampling using advancing generator state. Separate identical server
starts could reuse the same initial stream; sequential passes here consume later
draws in a continuous process. This is stochastic evaluation of one checkpoint,
not training-seed replication or a promise of bitwise reproducibility.

The wrapper records the server PID before and after every pass, checks endpoint
health, and fails on a changed server PID. The campaign monitor checks endpoint
health every 30 seconds and GPU utilization periodically. Existing owners preserve
local Docker transcripts, enforce first-cell preflight, recover remote logs and
verify termination. Each arm has a $25 GPU/storage ceiling (including $2 recovery
reserve), a $5 judge ceiling, and an independent hard-lifetime watchdog. A provider
GPU rate above $3.50/hour is refused. No automatic pod replacement is authorized.

Code: `scratch/dataset_refresh/three_pass_odcv.py`; source config:
`scratch/dataset_refresh/odcv_three_pass.yaml`; runtime plans/status:
`output/odcv_three_pass_20260916/`; short evaluation root: `C:/odcv-three`.

Run preparation with `uv run --no-sync python -m scratch.dataset_refresh.three_pass_odcv prepare`,
then launch once with the same module's `run` action. The owner delegates to the
standard eval entrypoint and three-pass runner. Completion verifies 240 native
transcripts, both judges, all published file hashes, model/protocol pins, six
server continuity boundaries and pod closure. Failures remain visible; observed
outcomes are not rerun to improve scores.

The naming law maps today's nonmoral result to the same Hub repository as today's
earlier single-pass result. Its exact old revision
`b06c757309213d1d0e1ef47942e36b2157031db5` is retained, with tag
`single-pass-20260916` created before launch. The new complete three-pass run will
become that repository's main revision; report both revision pins explicitly.

Preflight passed for pinned adapters, Docker, shell LF bytes and network capacity.
Focused offline tests cover protocol/budget admission and server continuity.
