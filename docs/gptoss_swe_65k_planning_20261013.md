<!-- ABOUTME: Offline context-cap score scenarios, repriced token evidence and conditional GPU estimates. -->
<!-- ABOUTME: Documents protocol-preserving infrastructure fixes and their measured validation limits. -->
# 65,536-token SWE planning scenarios, 2026-10-13

This is an offline extrapolation, not a new evaluation, quote, counterfactual grade, or causal estimate. No rentals or paid model calls. The response cap stays 16,384; total generated-token cap 262,144 and step cap 500 stay unchanged.

## Score scenarios

The pinned Oct 9 20-task results were Base 6/20, Control 5/20, DA15 7/20. Direct multiplication by 15 gives 90, 75, 105 resolutions per 300 at 131,072 context. At 65,536, 5, 2, 4 of these known passes fit entirely; 1, 3, 3 exceed it. Losing every over-limit win versus retaining all gives mechanical scenarios of 75–90, 30–75, 60–105. These are NOT confidence intervals or reliable bounds on a new run. Some saved limit-ending patches passed, and may have been correct earlier; other patches may have been corrected only later. We did not reconstruct and grade the patch at the proposed cutoff. Earlier stopping could also preserve a patch later broken. Twenty tasks per arm leave large sampling uncertainty. Initial ten excluded requests fixtures; not a uniform sample of all 300. Tinker observations do not establish identical MXFP4 results, especially for the not-yet-qualified RunPod adapters.

## Tinker cost replay

Downloaded all three immutable rollout archives at 09bc27d232fb52c6a6d46aa54f09c12e6ac4e9b5, verified archive SHA256 against the prior verified extraction, selected only final valid attempts. Sum recorded prompt/cache/output usage until first 65,536 crossing, truncate the crossing response's output at remaining context and stop. This is a fixed-path assumption. In a real run changed output allowances can change subsequent behavior. It excludes prior failed attempts, startup/infra waste and CPU/storage charges. Rates are $0.33/M input, $0.066/M confirmed cached input, $0.84/M output, matching the current public Tinker rate card checked October 13.

| Per 300 | Base | Control | DA15 | Total |
|---|---:|---:|---:|---:|
| 65,536, observed cache pattern | $126.45 | $69.15 | $46.03 | $241.63 |
| 131,072, observed cache pattern | $341.68 | $188.18 | $127.19 | $657.05 |
| 65,536, zero cache hypothetical | $552.22 | $275.76 | $171.65 | $999.63 |

These estimates reprice token usage; do not multiply historical cumulative ledgers (which include old prices and previous attempts). Source and intermediate requests: output/gptoss_mxfp4_cost_audit_20261013/cost65_audit.py and cost65/{base,control,da15,projections}.json. Cache behavior and task representativeness can change the bill substantially.

## RunPod sensitivity, not measured optimized throughput

The replay retains 11.534M / 12.647M / 10.644M generated tokens per 300. At the pilot's historical H200 quote $5.29/h, assume an effective aggregate 120–300 output tokens/s/GPU, including driver gaps, queueing and tools. This range is an explicit planning assumption, not a measured result of the fixes: the four-task pilot's task window delivered about 122 tokens/s, while raw request intervals divided across four workers imply approximately 423 tokens/s before excluded overhead. Neither is a production fleet throughput measurement. At 120–300, calculated GPU compute costs are $56–141 / $62–155 / $52–130. Round up for modest amortized startup/tail to $60–150 / $65–165 / $55–140, total $180–455. CPU, storage and transfer are extra. Lower GPU rates scale these estimates; no current allocation was quoted. Do not infer Control/DA15 runtime parity: exported Tinker adapter compatibility and exact MXFP4 serving have not been qualified. Use a same-protocol 4-versus-8-worker throughput check before treating these as a budget forecast. That check requires separately authorized paid work.

## Implemented infrastructure changes

- SWE hot trajectories, HTTP evidence, shim traces and raw replies use durable native Docker volumes instead of the Windows bind mount. A scoped helper makes incremental atomic host mirrors and final SHA256-verified copies. Existing volumes refuse automatic reseeding; all production primary volumes remain retained, including after failures. GPU inference-ended signal precedes final copying.
- Bounded in-process immutable message-blob cache eliminates repeated filesystem lookups. Existing blobs are verified after reopen; failed writes never enter cache. Atomic fsync writes, raw bytes and publication integrity verification remain mandatory.
- App-local bounded exact-history prompt cache shares admission and generation rendering; changed history/tools invalidate it. Mutable return objects are independent. Reference renderer access stays serialized; render/parse/disk work leave the asynchronous event loop. Sampling, tool feedback, history, token IDs and limits are unchanged.
- Client audit, bridge transport and shim stages record timings. Writes are awaited before dispatch/return. Fresh HTTP connections and zero ambiguous retries remain.
- GPU log collection keeps previous good evidence on SSH/empty failures, saves timestamped snapshots, and captures metrics plus periodic nvidia-smi utilization. No claimed throughput gain until measured.

The original pilot remains a historical base-only launcher, not a new full three-arm campaign. No adapters, new task selection or new run settings were enabled. The 4-versus-8 experiment and alternative hardware comparison are deferred because they need paid work. Full trace schemas stay readable by existing tooling; only already content-addressed message storage is deduplicated.

## Validation

52 locked Linux Harmony/raw-bridge tests passed; one optional installed-mini-client test skipped in that environment. Twelve pinned mini-SWE real HTTP protocol tests passed separately. Six Linux storage/cache/log regression tests passed, plus an actual Docker native-volume producer -> atomic Windows mirror -> final hash readback test. Windows focused pilot/bridge/storage tests: 15 passed, three Linux/environment skips. Network-disabled Harmony test initially lacked tokenizer vocabulary; rerun only fetched public vocabulary, no model calls. Native qualification resources were verified by exact labels and removed; production evidence and existing resources were untouched.
