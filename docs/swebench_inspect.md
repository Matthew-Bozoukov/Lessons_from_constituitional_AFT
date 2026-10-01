<!-- ABOUTME: Optional native Inspect agent backend within the existing SWE-bench Lite fleet. -->
<!-- ABOUTME: Installation, qualification, differences and evidence; no new model run is implied. -->
# Inspect backend for SWE-bench Lite

The default remains mini-SWE-agent. Add `--agent-backend inspect` to a **new**
fleet launch to use Inspect AI 0.3.268 and inspect-evals 0.21.0. This is a separate
`lite-inspect-v1` protocol and result namespace. It must not resume mini outcomes
or be described as a scaffold-identical comparison.

## What changes

We instantiate the upstream `inspect_evals.swe_bench.swe_bench` task, supply the
existing frozen Lite instance and digest-pinned Docker sandbox, and run Inspect's
native ReAct agent with bash and submit tools. Inspect handles tool dispatch,
conversation state and native `.eval` logs. Our adapter supplies resource and token
admission, sampling verification and prediction export. There is no fork of Inspect.

The existing fleet still owns longest-first scheduling, four independent conversations
per GPU, model/LoRA serving, GPU allocation and fallback, leases, budget accounting,
infrastructure-only retries, checkpoint uploads, official grading and teardown.
Inspect does not supply our RunPod/Vast lifecycle or campaign-specific HF publication.

Upstream defaults require explicit overrides: Verified rather than Lite, a 30-message
limit, different tools, and Epoch images rather than our cached official images.
See the [upstream task documentation](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/swe_bench/README.md)
and [provider documentation](https://inspect.aisi.org.uk/providers.html).

Our changes are:

- Read each leased row from the frozen 300-task dataset, remove gold patch/test data
  before creating the Inspect task, and use its pinned cached image with `pull_policy: never`.
- Use network-disabled Docker Compose, 2 CPUs/4 GiB/512 PIDs per sandbox, the shared
  32-command gate, bounded test threads and command timeout/descendant cleanup.
- Keep all prior reasoning with `reasoning_history="all"`. Disable compaction and
  history truncation. Use the supported OpenAI-compatible provider against the
  already-running vLLM endpoint; it accepts the custom admission transport.
- Explicitly forward all seven sampling parameters. The generic provider requires
  `top_k`, `min_p` and `repetition_penalty` in `extra_body`. Verify the final wire
  fields and exact `/tokenize` versus chat usage before accepting a response.
- Reserve prompt plus allowed output against the shared measured KV pool. Clamp
  each request against remaining cumulative and context budgets. Count generated
  tokens only; a tool reply is not a model generation or a generated-token charge.
- Override the upstream message cap with our model-generation and output-token limits.
  At response/token/context limits, stop without executing the truncated call and
  preserve the tracked source patch. A malformed batch cannot execute its otherwise
  valid bash or submit calls. No automatic reflection, score-based retries or best-of-N.
- Save native `.eval` logs, full wire requests/responses, partial response bytes,
  transcripts and legacy-compatible predictions/diagnostics. Defer grading to our
  existing official SWE-bench 4.1.0 phase, after releasing GPUs. The upstream default
  scorer is not used; this retains the existing grading and requests-fixture policy.
- Disable SDK/model retries. A 429 or another infrastructure failure can receive a
  bounded fleet retry; uncertain inference completion fences that replica first.
  Systemic protocol errors halt for diagnosis. Valid model failures are never rerolled.

## Exact preparation and launch

On a prepared Linux CPU, before any GPU rental:

```bash
uv sync --frozen --project src/eval/capabilities/swebench_mini/envs/inspect
src/eval/capabilities/swebench_mini/envs/inspect/.venv/bin/python \
  -m src.eval.capabilities.swebench_mini.inspect_task --check-runtime
docker version --format json
docker compose version
```

Use Docker Engine with a modern Docker CLI and Compose v2. Local qualification used
CLI 28.0.4, Engine 29.6.2 and Compose 2.39.4. An old Docker 20.10 client failed Inspect's
JSON-version prerequisite check even when the daemon was modern. Install the Compose
plugin if absent; the backend preflight refuses to rent against a missing/drifted runtime.
The CPU bootstrap installs the isolated Inspect environment; the GPU installs no Inspect
packages and continues to serve the same base weights/LoRA through vLLM.

Run the existing host capacity, protocol/template, lifecycle and mini smoke checks in
[the runbook](swebench_lite_runbook.md), plus the Inspect checks:

```bash
src/eval/capabilities/swebench_mini/envs/inspect/.venv/bin/python \
  -m unittest tests.test_swebench_inspect
```

The local Docker test driver is `scratch/swebench_inspect_integration.py`. Its six
synthetic scenarios exercise native Inspect through production fleet workers, then
officially grade public reference patches and empty patches. This is infrastructure
qualification, not a model score. For a replacement CPU, retain the separate native
host capacity/OOM qualification; the socket-mounted laptop driver cannot read the
daemon host's cgroup paths.

Bind these **executed** checks to the Inspect recipe with the existing qualification
command's usual load/smoke/test/transport/protocol/template arguments, adding:

```bash
uv run --project scratch/swebench_cpu_env --frozen python -m scratch.swebench_qualify_recipe \
--load-dir PATH_TO_HOST_LOAD_TEST \
--smoke-root PATH_TO_HOST_MINI_SMOKE \
--test-log PATH_TO_FLEET_REGRESSION_LOG \
--transport-log PATH_TO_TRANSPORT_LOG \
--protocol-log PATH_TO_MINI_PROTOCOL_LOG \
--template-proof PATH_TO_TEMPLATE_PROOF_JSON \
--agent-backend inspect \
--inspect-smoke-root PATH_TO_PASSED_INSPECT_INTEGRATION \
--inspect-test-log PATH_TO_INSPECT_UNITTEST_LOG
```

This writes `/srv/lasr/recipes/qwen36-lite-inspect-v1.json`, including its own Inspect
qualification hashes. The mini recipe remains separate. Source or protocol drift
requires requalification; merely editing recorded hashes is not qualification.

For an authorized new run, from the CPU repository:

```bash
uv run --project scratch/swebench_cpu_env --frozen python -m src.eval.run_eval \
  --name swebench_mini --fleet --agent-backend inspect \
  --target ORG/LORA --target-revision SHA --budget-usd 180
```

The corresponding mini command omits `--agent-backend inspect` (or uses `mini`).
Status/resume/grade/publish use the campaign's saved `launch.yaml`; never apply a new
backend overlay during recovery. The exact same root cannot switch backend/protocol.
The official results remain in the campaign HF dataset. Native logs can be opened with
`uv run --project src/eval/capabilities/swebench_mini/envs/inspect --frozen inspect view PATH`.

## Limits after the looping probe

The ten selected historical prefixes, replayed twice at temperature 1 with presence
penalty 0, produced 20 responses without detected loops. Every response used fewer
than 16,384 output tokens; the largest used 6,543. That supports trying the old **16k
per-response cap**, with about 2.5 times the largest observed response as headroom.
It is not a test of a complete issue trajectory.

The old control/DA-15 audit found 144 full-response caps with detected repetition,
but also **34 cumulative 65,536-token exits and 15 exits at 250 model generations**
without that exact-loop signature. Those are task-arm outcomes across two runs.
Absence of a detected loop does not prove useful progress or eventual correctness.
It does mean the replay does not establish that those older cumulative/step caps
are comfortably sufficient. Keep 262,144 generated tokens and 500 model generations
until an entire new-protocol run shows otherwise. Context remains 262,144.

No default budget was changed by this implementation. The overlay initially inherits
the mini v4 allowances (65,536 response / 262,144 generated / 500 calls) to avoid
changing the scaffold and budget simultaneously. A max-token setting is a ceiling,
not a command to generate that many tokens, although it reserves KV space and can
reduce simultaneous decoding. A future 16k response protocol should be named and
qualified explicitly, using the same limit for every evaluated arm.

See the [live sampling probe](swebench_loop_probe_2026-09-25.md) for pinned provenance.
The original audit is published with the infrastructure evidence; it does not turn
historical failed outcomes into new results.

## Remaining qualification boundary

Local qualification on September 25 passed **173 fleet/provider/HF regressions**
(one platform-specific skip, two passing subtests), **15 Inspect contract/transport
tests**, and **nine existing mini-client protocol tests**. Six real Inspect/Docker
scenarios made 11 synthetic HTTP generations. Both public reference patches resolved;
four empty patches remained unresolved; official grading reported zero errors and
zero unstopped task containers. Native reasoning, no-tool/malformed histories,
sampling, whole-batch rejection, token/context/turn limits, resource caps and
completed-outcome preservation were checked.

[234 artifact files were hash-verified on HF](https://huggingface.co/datasets/dougalldeepmind/2026-09-24-swebench-lite-infrastructure/tree/0378f088a37a1bb1963565b114a2ecca81fc3123/metadata/audits/2026-09-25-inspect-backend).
The local grading wrapper optimizes tag inventory only: Docker's high-level list
method otherwise inspects every unrelated cached image. It leaves patch application,
test execution and grading unchanged. Runtime and wire evidence identify the tested
dependencies; no cloud host or model was started for this implementation.

The adapter is locally tested. No Inspect-backed real Qwen issue or 300-task campaign
has run yet, and no GPU throughput or score parity is claimed. The first requested
Inspect campaign must prepare/qualify its CPU and pass preflight. Inspect changes
agent prompts and submit behavior, so future capability comparisons should use the
same backend, sampling and budgets for every arm.

Inspect's filesystem checkpoint/resume is deliberately disabled. Native logs and
durable raw evidence survive failures; an infrastructure retry restarts that task
in a clean sandbox. Resuming an interrupted filesystem is separate work and is not
claimed here. The backend does not guarantee freedom from model loops or provider failures.
