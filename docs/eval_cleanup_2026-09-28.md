<!-- ABOUTME: Active evaluation repair scope, implemented changes and remaining qualification. -->
<!-- ABOUTME: Status distinguishes offline checks, protocol validity and live Qwen readiness. -->

# Active evaluation cleanup

Worktree `eval-audit-20260928`, branch `codex/eval-audit-20260928`, based on freshly
fetched main `c38a29fc179c1068e18e78258cbbaed2558e3eeb`. Other checkouts/runs were
not changed. Scope is the user's twelve instruments below. The broader inventory
remains in [the preliminary audit](eval_audit_2026-09-28.md).

"Offline checked" means deterministic tests with fake completions or schema
checks of pinned public data. It does not mean that a live Qwen/GPT-OSS run,
Docker qualification, or scientific judge calibration has passed.

## September 30 continuation

October 2 update: Agentic Misalignment now uses `agentic-action-judge-v3`:
judges/action gates exclude native CoT and private scratchpads, leak recipients
are checked against an explicit authorized list, and every new run records its
standalone scoring protocol. Historical runs remain unchanged; fresh comparisons
wait for the new DA dataset. Details are in the [behavioral follow-up](behavioral_eval_cleanup_2026-09-29.md).

The user approved Arena's 12,000-token default. Both sides of new comparisons
must share that generation protocol; historical 6k arms must be regenerated.
The bounded study is complete and published, so the pre-qualification statements
below about no live inference/publication apply only to their earlier passes.

The current code review covers MMLU, MoReBench, Agentic Misalignment, Psychosis,
Dictator, Petri, Secret Number and Delegated Harm. Confirmed failures are reproduced
before repair. Qwen SWE-bench Lite, ODCV and MASK remain excluded, and GPT-OSS
qualification remains parked. The maintained user scope is twelve instruments;
a lack of new defects is not a reason to alter a working scientific protocol.

Current verified changes:

| Instrument | September 30 result | Still needed for live readiness |
| --- | --- | --- |
| Arena-Hard | User-approved default is now 12k; existing protocol checks reject unmatched reuse. | New comparisons must regenerate both arms under 12k. |
| MMLU | Fixed Windows Unicode transcript-writing failure; retained all 570 questions and scoring. | Live parsing/truncation check on the pinned control. |
| Agentic Misalignment | Fixed 12 generated conditions from an 8-condition config; declared panel checked before calls; failures retain raw evidence. | Live Qwen generation/classification check; compare historical conditions explicitly. |
| Psychosis | Fixed Unicode artifact writing and lost completion flag; rubric unchanged. | Live target/attacker/judge check and review of graded transcripts. |
| Dictator | Fixed missing prior reasoning in target history, labelled `preserve-reasoning-v1`; judge still sees visible text. | Live multi-turn check under the corrected target-history protocol. |
| Petri | Resolve supplied Hub branches to exact commits; reject foreign audit logs; synchronize JSON/Markdown summaries. | Actual endpoint pilot and review of constitution-specific controls. |
| MoReBench | Resume keeps empty/truncated target outcomes instead of selectively regenerating them. Transport/judge failures remain separately retryable. | Live two-scenario target and criterion-judge smoke. |
| Secret Number | Make Docker text decoding UTF-8 with invalid bytes escaped. Existing detector limitations remain explicit. | Live Qwen tool episode; review negative episodes as well as flags. |
| Delegated Harm | Fix Windows world-text decoding so frozen hashes match the published UTF-8 source. | A 36-request historical candidate is validated locally; choose it or fresh nosynth authoring before responder comparisons. |

**325 tests passed:** 293 in the combined affected runner suites, plus 32 in
Petri's pinned Inspect/Petri environment. Petri includes an offline full
controller/target/judge/log-reader round trip. No new paid model calls or rentals
were used for these repairs. Code checks do not establish live model readiness
or semantic judge accuracy. Protocol and evidence details are in the linked eval
notes below; the September 28 table is a historical inventory.

## September 29 scope decisions and follow-up

Arena-Hard follow-up: [qualification evidence and repairs](arena_hard_qualification_2026-09-29.md)
supersede the initial judge-readiness assessment. GPT-4.1 completed all 100 saved
answer pairs; Gemini completed 98 and disagreed substantially. The user selected
GPT-4.1 as primary, with Gemini coverage/agreement reported as diagnostics.
Complete primary judgments remain required. Report version 3 makes this policy
explicit; it does not certify either judge as scientifically calibrated.

Latest scope: **Qwen3.6-27B only**. GPT-OSS/Tinker target work is parked;
previously committed changes remain available but require no further work now.
Qwen SWE-bench Lite, ODCV and MASK remain excluded from further edits.
The user selected the existing nosynth control for any live Qwen checks:
`dougalldeepmind/2026-09-22-qwen36-0-nosynth`
at revision `633908b72a9799fb3e6b101b0a8a82aec3c3d642`, verified against the
Hugging Face model API on September 29. Checkpoint selection is needed to bind
live inference evidence to exact weights; it does not block code repairs or
offline tests. This scope change does not change MoReBench's fixed judge protocol.

The user retained MMLU's current 570 questions, requested Arena-Hard history checks,
assumed GPT-OSS120B is hosted on Tinker, and excluded further edits to Qwen's SWE
fleet, ODCV and MASK. Petri's revised rule supersedes the earlier shared-constitution
decision: an explicit constitution wins; otherwise recover the one used by that
model from pinned training provenance. Delegated Harm's primary protocol is now
frozen shared requests, with a separate author-only preparation mode.

- [MMLU/Arena follow-up](mmlu_arena_cleanup_2026-09-29.md): health gates enforced,
  existing paired statistics retained, historical Arena controlled hard score
  reproduced from pinned saved evidence, and framework reporting repaired.
- [Behavioral follow-up](behavioral_eval_cleanup_2026-09-29.md): reproduced
  blackmail literal-word gate and malformed-judge defects, retained failed judge
  evidence, and actual local Docker Secret Number controls. Windows oracle line
  endings fixed; a hidden-read detection gap remains explicitly documented.
- [Petri workflow](petri_constitution_audit.md): configurable constitution pilot;
  new seeds and judges remain uncalibrated.
- [Tinker/GPT-OSS workflow](tinker_eval_tools.md): isolated locked runtime,
  verified tool/history/sampling transport through the real mini-SWE client with
  a scripted sampler, and an explicit 300-task Lite config. Its stock-agent path
  remains distinct from Qwen's fleet protocol, with different context and total
  generation-budget limits; it is not a matched cross-family comparison.
- [Delegated Harm workflow](delegated_harm/README.md): frozen bank default;
  preparation never runs responder episodes. A real author checkpoint/bank is
  still required.

The table below records the initial September 28 pass; this follow-up and the linked
current workflows supersede its pending default choices. No new model checkpoint
has yet been used for live inference, and no judge calibration is claimed.

Final September 29 checks: **327 main-suite tests passed**, **24 Tinker compatibility
tests passed in its locked environment**, and **eight opt-in real Docker controls
passed**. The main suite's other skips are six Linux-only fleet modules. The eight
Docker skips and one Tinker module skip in that main invocation were exercised by
the separate commands above. Arena historical score reproduction and read-only
Petri provenance resolution are documented in their linked notes. No paid model
calls, rentals or HF publication were made. The Qwen live checkpoint is now pinned
above; live inference checks and the real Delegated Harm request bank remain
outstanding.

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
| Secret Number | Agentic specification-gaming propensity | Removed Tinker-only gate and added Qwen tool-serving requirements; distinguish detected shortcuts, attempts and commands needing review; preserve gaming in non-submitted episodes; checkpoint partial traces and observe oracle/log state after commands. Pin containers to built image IDs and clean up failed setup. | Command detection remains conservative, not exhaustive execution tracing. Noise is unseeded and explicitly recorded; common seeds pair secrets only. Native Docker and both-model tools/history need live qualification. |
| Delegated Harm | Requester-conditioned propensity, one acting subject | Added immutable shared-request-bank mode alongside the explicit self-authored protocol; verify hash, source worlds, accepted text and full coverage; neutral shared model label, family-specific base pins and API transport. Reject duplicate/foreign cells and report missing bounds even when no results are valid. | Choose default request protocol and freeze a real accepted bank for responder comparisons. Validate live model tools/effective sampling and action judges. No reciprocal multi-agent interaction is measured. |

## Order of qualification after code repair

1. Review the deterministic repairs and freeze protocol versions. Changes to
   missingness handling are disclosed; do not silently overwrite historical results.
2. Use the pinned nosynth Qwen control above for live checks. GPT-OSS/Tinker
   qualification is parked. Do not change or rerun Qwen SWE-bench Lite, ODCV or
   MASK as part of this repair pass.
3. Run one real endpoint smoke per active Qwen eval, with native Docker for
   Secret Number. Check exact target identity, transmitted settings, reasoning separation,
   token limits, tools, expected item counts and saved provenance. Then run a small
   frozen calibration panel for every judge-dependent instrument.
4. Calibrate Petri before scaling its audit. Freeze matched controls, primary
   endpoints and the sampling unit before the full comparison.

## Offline validation

The initial repair suite passed **314 tests, with six platform/optional-dependency
skips**, on Windows. This includes MMLU, Arena-Hard, MASK, agentic-client/coverage
regressions, Psychosis, Dictator, MoReBench, ODCV and the available SWE-bench
protocol/runner/watchdog checks. Linux-only fleet tests still need the Linux CPU.
The optional Vast SDK was supplied with `uv run --frozen --with vastai==1.8.0`;
watchdog provider actions were mocked. No root lockfile or dependency policy changed.

New regressions exercise real runner boundaries with synthetic completions:
request-cache invalidation, null/empty/truncated responses, retained turn traces,
complete coverage, paired Arena judgments, validation-failure evidence and
MoReBench signed weights/channel separation. They do not simulate model quality.

The subsequent Secret Number / Delegated Harm extension is documented in
[its protocol notes](secret_number_and_delegated_harm_cleanup.md). The pinned
external Delegated Harm fixtures were staged so its world/tool tests actually run;
the source checkout and all fixtures remain outside Git under `output/`.
The expanded suite now passes **357 tests, with six Linux-only fleet skips**.

No CPU was resumed, no GPU rented, no paid inference/judging called and no HF
artifact published during either pass. The isolated branch contains the
implementation, offline tests and September 29 Docker controls; scientific judge
calibration and live model serving qualification remain open.
