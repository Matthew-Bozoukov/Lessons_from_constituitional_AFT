<!-- ABOUTME: Active evaluation repair scope, implemented changes and remaining qualification. -->
<!-- ABOUTME: Status distinguishes offline checks, protocol validity and live two-family readiness. -->

# Active evaluation cleanup

Worktree `eval-audit-20260928`, branch `codex/eval-audit-20260928`, based on freshly
fetched main `c38a29fc179c1068e18e78258cbbaed2558e3eeb`. Other checkouts/runs were
not changed. Scope is the user's ten instruments below. The broader inventory
remains in [the preliminary audit](eval_audit_2026-09-28.md).

"Offline checked" means deterministic tests with fake completions or schema
checks of pinned public data. It does not mean that a live Qwen/GPT-OSS run,
Docker qualification, or scientific judge calibration has passed.

| Eval | Kind | This repair pass | Remaining acceptance work |
| --- | --- | --- | --- |
| MMLU | Capability | Pin test and few-shot dev to one dataset commit; publish exact prompts; bind resume cache to model/revision and decoding. Existing scoring retained. | Live parse/truncation health on both models. Standard entrypoint produces fresh runs; legacy arm-ladder CLI remains historical. |
| Arena-Hard | Capability/preference | Fix base/API answer filenames; run-local harness copies; cache identity includes prompt/checkpoint/settings; retain reasoning/stop reasons; allow nonstreaming Tinker transport; enforce judge validation, exact reused prompts and both judge orderings for every requested question. Preserve raw validation judgments and remove generated credentials after subprocess completion/failure. | Real judge calibration and same benchmark/reference protocol on both endpoints. Smoke reduces counts to four per category and labels the comparison; that is not full qualification. |
| SWE-bench Lite | Agentic capability | Reviewed maintained `--fleet` / `lite-v5` path and offline protocol tests. No rewrite of the working fleet or scorer. | GPT-OSS serving/renderer/sampling recipe and exact-checkpoint Docker smoke. Do not confuse older generic 10%-Verified runner or separate Inspect backend with Lite-v5. |
| Agentic misalignment | Single-agent propensity | Forward target auth; remove shared `.env` symlink; explicit 10,000-token historic budget; retain out-of-band reasoning; reconcile every planned response and boolean verdict before aggregation. OpenAI-compatible targets enabled. | Live both-family transport, stop/trace behavior and classifier validation. Missing cells now stop publication rather than shrinking the denominator. |
| ODCV | Agentic propensity | Preserve maintained metric logic. Namespace Compose projects by run as well as model, remove global network pruning, and rewrite only loopback hostname when bridging Docker. | Native-Docker host check; GPT-OSS reachable/authenticated target and context/tokenizer contract. API registry remains closed until that path is qualified. |
| Psychosis | Interactive single-target propensity | Checkpoint every turn, retain partial conversations, stop on empty/truncated target or incomplete attacker/judge responses; require all personas/grades before a headline summary. | Live attacker/judge calibration and per-turn token headroom. Simulated persona behavior is not a clinical outcome measure. |
| MASK | Honesty under pressure | Normalize null/empty completed replies; save raw text and finish reasons; count incomplete generations as errors; per-archetype tallies no longer share mutable global counts. Default error cap is zero. | Live belief/pressure response and judge checks. Raising the error cap restores a legacy incomplete-data policy and is not qualification. Finished empty replies still follow the explicitly configured evasion/reasoning policy. |
| Dictator | Single-agent authoritarian-request propensity | Preserve every turn's trace/raw completion; exclude nonempty truncation as well as empty truncation; check judge completion; expose coverage and worst/best bounds for missing samples. Existing 1–5 rubric retained. | Independent judge calibration; all-sample coverage before interpreting a conditional resistance rate. |
| Petri constitution audit | Adaptive behavioral audit | Recovered pinned historical seeds and analysis; identified known control false positives and incorrect uncertainty. [Rebuild acceptance contract](petri_constitution_audit.md) written. User selected one shared current approved constitution for all models. | Pin the exact constitution file, then freeze seeds and calibrate the judge blindly. The historical instrument is not ready to reuse as an internalization score. |
| MoReBench | Procedural moral-reasoning capability | Added registered runner and [pinned public protocol](morebench.md), separate reasoning/answer scores, signed-weight scoring, strict completeness and resume checks. Actual 500-row/11,450-criterion release validated. | Live both-family smoke and independent-judge sensitivity for GPT-OSS targets. No full run yet. |

## Order of qualification

1. Review the deterministic repairs and freeze protocol versions. Changes to
   missingness handling are disclosed; do not silently overwrite historical results.
2. Settle GPT-OSS hosting. A generic API-capable runner is not proof that Tinker
   honors the request: the current shim still lacks locked SDK dependencies,
   effective-sampling guarantees, safe concurrent port/identity handling and the
   container/tokenizer interface required by ODCV. Native vLLM also needs a verified
   GPT-OSS profile/tool parser. This is a shared prerequisite, not ten benchmark bugs.
3. Run one real endpoint smoke per active eval and model, with native Docker for
   SWE/ODCV. Check exact target identity, transmitted settings, reasoning separation,
   token limits, tools, expected item counts and saved provenance. Then run a small
   frozen calibration panel for every judge-dependent instrument.
4. Calibrate Petri before scaling its audit. Freeze matched controls, primary
   endpoints and the sampling unit before the full comparison.

## Offline validation

The active-eval suite passed **314 tests, with six platform/optional-dependency
skips**, on Windows. This includes MMLU, Arena-Hard, MASK, agentic-client/coverage
regressions, Psychosis, Dictator, MoReBench, ODCV and the available SWE-bench
protocol/runner/watchdog checks. Linux-only fleet tests still need the Linux CPU.
The optional Vast SDK was supplied with `uv run --frozen --with vastai==1.8.0`;
watchdog provider actions were mocked. No root lockfile or dependency policy changed.

New regressions exercise real runner boundaries with synthetic completions:
request-cache invalidation, null/empty/truncated responses, retained turn traces,
complete coverage, paired Arena judgments, validation-failure evidence and
MoReBench signed weights/channel separation. They do not simulate model quality.

No CPU was resumed, no GPU rented, no paid inference/judging called and no HF
artifact published during this repair pass. The isolated branch contains the
implementation and offline tests; scientific and live serving qualification remain open.
