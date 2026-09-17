<!-- ABOUTME: Three sequential ODCV passes for exact-original low-stakes conversations with new replay. -->
<!-- ABOUTME: Freezes model, sampling, shared-Docker isolation, budget and completion requirements. -->

# Original low-stakes plus new replay: ODCV

The user requested this evaluation on 2026-09-17: three sequential passes,
temperature 0.7, seed 0, local CPU/Docker and one rented RunPod GPU. The user approved the $30 combined GPU/judge/recovery ceiling after preparation.
The earlier $50 ceiling covered training and is separate.

Target: `dougalldeepmind/2026-09-16-qwen36-0-da-lowstakes-original-7`
@ `47f54dcb35f7e356e919933d7d5dde50ea101a40`.
Base: `Qwen/Qwen3.6-27B`
@ `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
This is the trained original 716 low-stakes conversations plus the exact new
9,284 replay rows, not the refreshed low-stakes model.

Protocol matches the September 16 three-pass controls: 40 scenarios in both
mandated and incentivized variants, 80 cells per pass, 240 planned rollouts.
Thinking enabled, temperature 0.7, context 28,000, scenario concurrency six,
2,400-second scenario timeout, four judge workers. Gemini 3 Flash via the pinned
google-ai-studio route judges misalignment and progress at temperature zero.

The vLLM process explicitly receives `--seed 0` once at startup. No per-request
seed is supplied. All three passes share one continuously running server; its
PID is checked before and after every pass. No inference requests are added for
these health checks. Each pass's local transcripts and audit are saved before the
next pass. Existing retry/reconstruction behavior is unchanged and must be
reported explicitly if triggered; observed outcomes are not silently replaced.

One H100 80GB is sufficient. Current quote: $3.49/hour. Proposed allocation:
$25 GPU/storage, $5 judging; $2 GPU recovery reserve; maximum admitted GPU rate
$3.50/hour, storage allowance $0.10/hour. The independent watchdog is registered
at provision, before SSH/bootstrap, and reacts to owner death and the fixed
lifetime. Do not rename the pod or kill the owner as a handoff. Proposed name:
`nika-low-stakes-original-odcv-3pass`. Any retry must retain failed evidence and
count prior spending against the same approved ceiling.

Local port 18123 is dedicated to this run. Evaluation root `C:/odcv-old-low`;
campaign `output/odcv_original_lowstakes_20260917`; frozen configuration and
`low_original_plan.yaml` live in the campaign. Shared Docker passes check network
capacity instead of globally pruning networks. Scenario Compose cleanup remains
scoped to the scenario. No other pod, container or local task is stopped.

Commands (run only after the spending approval):

```powershell
uv run --no-sync python -m scratch.dataset_refresh.three_pass_odcv prepare --single-plan output/odcv_original_lowstakes_20260917/low_original_plan.yaml --style da-lowstakes-original
uv run --no-sync python -m scratch.dataset_refresh.three_pass_odcv run --single-plan output/odcv_original_lowstakes_20260917/low_original_plan.yaml --style da-lowstakes-original
```

The existing campaign owner/monitor is reused for one arm. Local monitoring runs
every 30 seconds, with periodic GPU probes and a per-process sleep inhibitor.
No recurring app scheduler is created. The monitor verifies publication after
owner exit, checks owned-pod absence, writes `completion.json` and releases the
sleep inhibitor. Each raw rollout is already local; remote boot/server logs are
verified before ordinary teardown.

Completion requires all 240 transcripts, exact model/protocol pins, six matching
server-PID boundaries, complete MR/progress coverage, accounting for every judge
reservation, verified Hub payloads and owned-pod termination. If a provider blocks
a judgment, preserve it and the raw response, report missing coverage and bounds,
and request the user's disposition rather than retrying the same refusal or
silently changing judges. The older refreshed-low blocked case is a separate task.

Preflight verified the pinned model, Docker/Compose/network creation, 12-network
headroom, shell LF bytes and free port. Twenty-two focused tests passed, covering
pass continuity, explicit server seed, shared-Docker isolation, startup transport,
SSH byte preservation, pass packaging and failed-judge budget diagnostics.


## Launch, 2026-09-17 09:37 UTC

One owned pod `1j6anboez6ht4u` was allocated at $3.49/hour, named
`nika-low-stakes-original-odcv-3pass`. The watchdog was registered before waiting
for SSH, with a fixed 24,940-second cap. Owner and campaign monitoring are running
as hidden local processes. Source `5e566003`; approval and launch records are in
the campaign. No recurring app scheduler was created. Allocation is not evidence
of completed startup or rollouts; `health.json` and the owner state are live truth.
