<!-- ABOUTME: Current SWE-bench Lite launch and recovery procedure; historical protocols are separate. -->
<!-- ABOUTME: Offline qualification is not a model run or proof that sampling cures repetition. -->
# SWE-bench Lite

An optional [Inspect backend](swebench_inspect.md) is available with
`--agent-backend inspect` on a new launch. It has its own protocol, recipe and
qualification; the existing mini backend remains the default. Do not combine
their results as though the agent scaffold were identical.

## Current protocol: lite-v5

Use `configs/eval/swebench_mini/lite.yaml` as the source of settings. This protocol
supersedes the older v2/v3/v4 defaults. Historical configurations and results remain
immutable. DA-25 and paired inference are deferred unless explicitly requested.

- All 300 Lite tasks for one pinned Qwen3.6-27B rank-64 thinking LoRA.
- Up to **6 GPUs**, four independent conversations each, all H100 NVL preferred. Delayed H200/RTX PRO 6000 fallback retains price limits.
  Each ready GPU starts independently; each finished conversation claims another
  task without waiting for its three peers. Longest-first historical priority.
- **262,144 context tokens, 16,384 response tokens, 262,144 total generated tokens
  per task, 500 steps.** Admission reserves prompt plus output against 90% of the
  measured KV pool. Four conversations do not guarantee four simultaneous decodes.
- Sampling is explicitly pinned in YAML, including **temperature 1.0**,
  top_p 0.95, top_k 20, min_p 0, presence_penalty 0, frequency_penalty 0,
  repetition_penalty 1.0. Non-OpenAI parameters pass through `extra_body` and are
  verified on the wire. Repetition penalty 1.0 is neutral. The bounded live
  [loop probe](swebench_loop_probe_2026-09-25.md) found 4/10 greedy replays looping
  versus 0/10 at temperature 1, repeated at a second seed with the original 65k
  allowance. This supports mitigation, not a universal cure or full-task score.
  Never silently tune a run.
- Prior reasoning is preserved, including rejected responses. No-tool responses
  remain assistant turns. Rejected valid tool calls receive not-executed replies;
  syntactically unrenderable calls are quoted in assistant content, with the exact
  original retained in evidence. No commands in a rejected batch execute.
- Raw HTTP response bytes (including partial bodies), content-addressed request
  messages, sampling settings and request IDs are saved before client parsing.
  Tokenization counts and token-ID hashes are saved when the server provides IDs.
  This is not a capture of vLLM's internal generations before its reasoning parser.
- Loop and malformed-marker diagnostics are observational. They neither stop a
  generation early nor turn a model failure into a retry.

**CPU state:** consult the shared registry using the
[CPU lifecycle guide](swebench_cpu_lifecycle.md). The original host 52229744 was
destroyed on September 24; replacement 53118260 was prepared and qualified on
September 28 under persistent authorization. Never use an old address or rent a
duplicate because a document described the registry as empty. Code/tests/status-only
work does not authorize a new rental.

## Launch a newly authorized run

1. Read `CLAUDE.md` and the CPU lifecycle guide. Pin the adapter/base/dataset revisions;
   verify compatible base, rank and thinking mode. Use the selected worktree.
2. Reuse or prepare the registered native-Docker CPU: target 64 effective CPUs,
   >=190 GiB actual RAM and 500 GiB disk. Cache and verify all 300 images before GPUs.
   Forty conversation workers share a 32-command gate; each container has 2 CPUs,
   4 GiB RAM and 512 PIDs. Twelve official grading workers run after inference.
   Run the capacity qualification on a replacement host; the laptop tests are not
   a substitute for that host's CPU/RAM/disk readiness.
3. Deploy a reviewed committed source revision. Qualify
   `/srv/lasr/recipes/qwen36-lite-v5.json` against matching source hashes, sampling,
   real-client failure tests, Docker smoke/official grading, and host capacity proof.
   Never bless source drift by replacing hashes without tests.
4. On the CPU, from `/srv/lasr/repo`, run:

   ```bash
   uv run --project scratch/swebench_cpu_env --frozen python -m src.eval.run_eval \
     --name swebench_mini --fleet --target ORG/LORA --target-revision SHA --budget-usd 180
   ```

   From Windows add `--cpu-receipt RECEIPT --cpu-key KEY`, using the shared registry.
   The CLI reuses a receipted CPU; cold preparation follows the skill/lifecycle guide.
   `$180` is the maximum authorized GPU allowance, not expected cost. CPU cost is separate.
5. New v5 campaign paths and HF dataset subjects include `lite-v5`, preventing
   collision with the older temperature-0 campaign. Never splice new sampling/history
   behavior into its 27 completed DA-5 outcomes. Repeated identical v5 launches are
   idempotent; incompatible saved configs require review.
6. At actual launch, create one 15-minute progress/completion/failure heartbeat,
   verify ACTIVE and an initial real status read. Include exact campaign paths,
   ownership, budget and protocol. No monitor is needed for offline cleanup.

## Recovery and completion

New runs prioritize `sympy__sympy-11870` in the official grading dataset via
`grading_priority` in `lite.yaml`. The other eleven grading workers proceed in
parallel. Upstream treats `instance_ids` only as a filter, so sorting that list
alone does not schedule work; `results/grading/dataset.json` preserves all original
rows and changes only their order. Existing cached reports are still skipped.
The 1,800-second per-instance limit and upstream cache/test settings stay unchanged.

Official grading has a per-instance test timeout, not a wall-clock deadline for
the entire suite. `cpu_finish_reserve_seconds` is an admission allowance, not a
grader kill timer. Preserve cached reports on CPU-only recovery; the actual CPU
expiry and explicit stop remain binding. Diagnose any remaining harness errors
before treating them as scored outcomes.

A definitive 4xx rejection does not imply a still-running generation. Rate limiting
can use the bounded retry policy. Authentication/invalid-request/model-route failures
save a systemic diagnostic and require diagnosis before more rentals. Tokenizer
undercounts also halt globally rather than buying repeated broken replicas.

A timeout, dropped connection or uncertain 5xx still fences the affected replica:
we cannot release its KV reservation and retry while an orphaned decode may remain.
Other replicas continue. The infrastructure breaker counts **six distinct replicas**,
not four failed conversations per GPU; a global breaker requires diagnosis, never
an automatic new batch. Retries start a fresh isolated infrastructure attempt;
completed turns are archived but are not resumed into a reconstructed filesystem.
Valid model outcomes are never rerolled.

There is no shared batch cutoff, six-hour job deadline, or wall-clock task timeout.
The per-pod emergency ceiling is 24 hours, clipped to the cumulative affordable
lease at allocation. Reserve two hours for new-task admission plus boot/cleanup;
this admission allowance does not cap an admitted task. Independent watchdogs, cumulative budget and retry
bounds remain. Replacements receive fresh leases within the remaining budget.
Later HF backup failures do not cancel agents. Snapshots are atomic; official grading
and publication retry separately. Source/config drift, explicit stops, exhausted
budgets/retries and ownership mismatches fail visibly.

An explicitly authorized mid-run budget increase must preserve the cumulative
ledger. The running coordinator caches its configuration and manifest; editing
JSON alone does not change its admission decisions. `fleet_handover.py` supports
a reviewed coordinator-only handover: archive the old source/config/manifest,
record the new authorization, verify each worker's PID birth time and provider
ownership, and keep existing model workers and pod expiries unchanged. Before
replacing the coordinator, transfer parent-dependent watchdogs to deadline-only
guards, and use a maximum three-minute handover record for the independent
reaper. Restore normal service kill/restart behavior immediately after adoption.
The adopted workers occupy ordinary fleet slots, so the GPU ceiling still holds.
The initial handover checkpoint is nonfatal, like other running-campaign backups;
a surviving publisher can still hold its lock. Verify that adoption reaches the
fleet loop and arms the new parent-dependent monitors before calling it complete.
The adoption marker alone does not prove those later steps succeeded.
This is an explicit maintenance operation, not automatic recovery or permission
to raise a budget. Budget reservation refusals must not count as GPU scarcity
or trigger hardware fallback.

Completion requires 300 valid outcomes, 300 officially graded outcomes, HF readback
and rollout/result hash verification, and zero owned GPUs. Leave a persistent CPU
running only under its current authorization. Delete the run's monitor at completion.
Do not claim unattended success is guaranteed against provider/network failures.

## Qualification without inference

Run Linux regressions in the CPU-only environment, including `test_swebench_fleet`,
`test_swebench_session`, `test_swebench_admission`, `test_swebench_browser`, provider
lifecycle/watchdog tests and `scratch/test_swebench_lite.py`.
Run `tests/test_swebench_protocol.py` with unittest in the pinned **agent** environment;
it exercises real HTTP and LiteLLM, including partial disconnects and malformed calls.
The CPU-only driver intentionally omits unrelated evaluation dependencies.

`scratch/swebench_offline_integration.py` is the laptop Docker qualification: actual
production workers, existing pinned task images, synthetic responses/public gold patches,
and the official grader. It cannot validate CUDA, GPU cache behavior, provider scarcity,
or temperature's effect on looping. `scratch/swebench_lite_smoke.py` remains the prepared
host smoke, including HF publication. Qualification must include the protocol test log.

## Historical findings

Control: **136/300 (45.33%)**. DA-15: **134/300 (44.67%)**. Both are complete old-protocol
runs with documented operational changes; neither is a direct v4 comparison. DA-5 v3
stopped at 27 valid/graded outcomes, six resolved, with 39 infrastructure-invalid and
234 unattempted tasks. A longest-first partial subset is not a benchmark score.

All 144 full-response-cap outcomes examined in the original control/DA-15 runs had
literal repetition. Other token/step/context limits are different failure categories.
The subsequent [sampling probe](swebench_loop_probe_2026-09-25.md) supports the v4
temperature change on ten DA-5 failure prefixes. It does not retroactively
establish the cause of every control/DA-15 failure or change those outcomes.
The original August 7 comparison lacks full trajectories, so its looping rate is unknown.

See [historical incidents](swebench_lite_history.md), [gotchas](GOTCHAS.md), and
[the CPU lifecycle](swebench_cpu_lifecycle.md). Prebuilt serving images, a second CPU,
and in-place continuation of an interrupted workspace remain optional engineering
work, not prerequisites proved necessary by these audits.
