<!-- ABOUTME: Preliminary inventory and evidence-based triage of project evaluations at current main. -->
<!-- ABOUTME: Audit only; no paid model calls, infrastructure launches, benchmark changes, or publication. -->

# Evaluation inventory and preliminary audit — 2026-09-28

Baseline: `c38a29fc179c1068e18e78258cbbaed2558e3eeb`, freshly fetched `origin/main`.
Branch: `codex/eval-audit-20260928`. Worktree: `C:/Users/nikak/.codex/worktrees/eval-audit-20260928/teaching_claude_why_replication`.
The existing main checkout and other worktrees were not switched, pulled into, or edited.
Main already matched the fetched commit.

Scope: inventory the current code, configs, relevant project logs and test coverage, then identify blockers for Qwen3.6-27B on RunPod with a local/Vast CPU driver, and GPT-OSS-120B with hosting still to be decided. This is an initial audit, not end-to-end certification or a claim that published results are all invalid. No GPUs, CPU rentals, Docker workloads, paid model calls, or HF uploads were started. No benchmark implementation was changed.

## What kinds of evaluation exist?

The [registry](../src/eval/__init__.py) has **17 entries**. Capability versus propensity is one axis; single-agent versus interactive team is another. Declarative probes and adaptive discovery need separate interpretation.

| Kind | Registered entries | Count |
| --- | --- | ---: |
| Capability | mmlu, arena_hard, swebench_mini | 3 |
| Single-agent behavioural propensity | agentic_misalignment, odcv, ctfish, psychosis, mask, dictator, secret_number | 7 |
| Requester/delegation-conditioned propensity | delegated_harm | 1 |
| Interactive multi-agent propensity/cooperation | odcv_peer, colosseum_jira, colosseum_hospital, whistlebench_team | 4 |
| Declarative values / constitution-knowledge-and-application probes | moralbench, internalization | 2 |

`delegated_harm` has AI-peer, AI-parent and human requester conditions, but only one acting subject; those requesters do not run a reciprocal conversation. `whistlebench_team` includes a solo condition as well as team conditions. Hospital surveys and mid-shift probes are self-report measurements attached to behavioural episodes.

Outside the registry: the eight-setting specification-gaming suite, Petri discovery and fixed-probe tooling, SURF audit tooling, and historical/experimental probes described below. Do not count every config, judge, analysis script or protocol variant as a new benchmark.

## Cross-cutting two-model readiness

**The requirement that every eval work on both families is not met.**

* Qwen is the maintained native-vLLM path. The clean Windows environment installs with `uv sync --frozen --group dev`, and all 17 runner imports succeed. That does not validate a model endpoint, an external Colosseum install, Docker images, credentials or scientific validity.
* GPT-OSS-120B's [profile](../configs/models/gptoss120b.yaml) is explicitly a naming-only stub. No verified serving/tool-parser/GPU facts are present. In particular, [plan_serving](../src/infra/endpoints/vllm.py#L590) refuses a tool-dependent native-vLLM eval without a verified parser. A native serving route needs qualification rather than a guessed Qwen-style configuration.
* Tinker support exists in [tinker.py](../src/infra/endpoints/tinker.py) and [tinker_server.py](../src/infra/endpoints/tinker_server.py), but **neither `tinker` nor `tinker_cookbook` is installed by the project lock**. The shim runs in the driver's interpreter, so a fresh standard install cannot start it. The upstream [Cookbook installation documentation](https://github.com/thinking-machines-lab/tinker-cookbook#installation) confirms these are separately installed components; pin and test the actual renderer/SDK combination.
* **Eight** registry entries reject any API/Tinker target: delegated_harm, swebench_mini, internalization, agentic_misalignment, odcv, odcv_peer, colosseum_jira, colosseum_hospital. This count corrects an early progress message that said seven.
* Of the nine API-enabled entries, Arena-Hard unconditionally requests streaming while the shim returns HTTP 400 for streaming. WhistleBench asserts that the API provider is OpenRouter, rejecting Tinker inside its runner. Both failures survive merely changing registry flags.
* `secret_number` has the reverse gate: `tinker_only=True` and no vLLM serving requirements, so Qwen is explicitly refused.
* The shim forwards temperature and token limit, but ignores requested `top_p`, `seed`, caller stop settings and tool-choice settings. Thus the saved requested config may not describe effective sampling. Unsupported controls must be implemented or rejected/recorded, not silently discarded.
* The shim always defaults to localhost port 1234, accepts any 200 response from `/models` as readiness, and does not verify checkpoint identity. Concurrent worktrees are **not** isolation for TCP ports. This is a wrong-endpoint risk by inspection; no collision was induced.
* Multi-agent evals bind seat assignment to `ServedTarget.sibling()`, a shared vLLM server and the same base family/mode. The scientific requirement is explicit, stable model identity per seat; sharing a process is only one implementation. GPT-OSS/Tinker needs independently addressable seat endpoints and a pinned peer/control policy.
* On a Vast Linux CPU, root dependencies still include the Linux GPU stack. Existing SWE-bench environments avoid some of this; a clean CPU-driver installation and pinned auxiliary harness bootstrap remain part of qualification.

**Recommended first architecture step:** represent endpoint capabilities explicitly (streaming, tools, reasoning channels, effective sampling, tokenizer/context limit, checkpoint identity), then route seats independently. Keep Qwen's proven serving path intact. Choosing Tinker versus native GPT-OSS hosting can wait while these defects are repaired.

## Per-eval triage

“Must fix” below means a demonstrated defect, a missing reproducibility prerequisite, or a blocker for the requested two-model scope. “Check” means a plausible risk requiring evidence. “No confirmed core defect” is not a live pass.

| Eval | What it measures / default scope | Preliminary assessment | GPT-OSS/Tinker-specific work |
| --- | --- | --- | --- |
| **mmlu** | Knowledge/capability; 5-shot, 10 questions per subject, normally 570 test questions | Core parsing and health reporting look maintained; tests pass. **Fix reproducibility:** `load_split` reads HF without a revision, including the dev examples. Pin both splits and retain selected items/prompts. This small sample is a regression screen, not full MMLU. | Ordinary endpoint calls should be portable after shim qualification; verify Harmony answer separation and token budget. |
| **arena_hard** | Pairwise judged answer quality; normally 500 hard + 250 creative prompts against a named reference | **Must fix:** generator names answers by served name but runner reads by artifact key; base-model run reproduced FileNotFoundError. Cache resumes by question UID only, without target revision/prompt/sampling identity; changing temperature reproduced zero new calls. Shared vendor staging also permits cross-run contamination. Config says judge validation enabled, but normal pool calls only judge mode; validation is a separate path. | Streaming is rejected by Tinker. API model names containing slashes also need safe artifact naming. Shim `/models` supplies no context limit, causing Arena's 8192 fallback; establish actual budget. |
| **swebench_mini** | Software repair capability; registry default is 10% of Verified (50 tasks), with separate full-300 Lite fleet and Inspect protocols | Recent, extensively maintained; no newly demonstrated scoring-code defect. **Fix/unify entrypoint clarity:** Lite fleet config is consumed by `scratch/swebench_lite.py`, not interchangeable with registry runner config. Keep mini, Lite-v5 and Inspect results distinct. Existing logs identify local HTTPBin and two Requests no-fix passes as scoring limitations; revalidate before benchmark-comparable claims. | API gate and hosted-vLLM assumptions need generalization; qualify actual Docker CPU, token accounting, transport and a real task before scaling. Current v5 recipe is Qwen-specific. |
| **agentic_misalignment** | Single-agent blackmail/leaking under self-preservation and goal-conflict conditions | **Must fix:** vendored client retains only visible `content`, dropping out-of-band reasoning; empty content becomes an exception. Runner hardcodes key EMPTY. Generation cap is hardcoded with `10000 if "gemini" or "qwen" in model.lower() else 4000`, an always-true condition. Aggregate has classified/missing counts but no expected-schedule reconciliation or truncation health. Old Qwen3 wording in config is cosmetic because runner rewrites served identity. | API gate, authentication propagation and reasoning/finish handling need repair; endpoint-independent behaviour is feasible. |
| **odcv** | Mandated/incentivized workplace misconduct and task progress; three passes | Strong existing recovery, transcript, missingness, action/progress and pooling machinery; selected tests pass. **No new core scoring defect established.** Check denominator completeness and recovered-versus-original transcripts; operational success is not behavioural success. Docker bridge/address and network capacity must be tested on the actual driver. | API gate; containers currently call the model through a host bridge. The Tinker shim binds only loopback, so enabling the flag alone cannot make Linux containers reach it. Parameterize authentication, routing and model label. |
| **odcv_peer** | Two agents acting concurrently in one ODCV sandbox; multiple peer/prompt conditions and a scripted control | Recent code; most tests pass. The one Windows test failure is a separator assertion, not evidence the Linux container loop fails. **Portability fixes required** for co-serving and explicit peer identity. Check selected scenario subsets and 12-pass default versus all40/3-pass protocol; they are different experiments. Assess write attribution and paired missingness. | Gate + same-vLLM sibling restriction + default Qwen peer. Decide whether each family is compared with its own matched peer or a fixed external peer; record both seat revisions. |
| **ctfish** | Specification-gaming propensity in a chess sandbox; 10 episodes by default, variants and majority judging | Core unit/metric checks pass except vendored-byte hashes: **Windows CRLF conversion**, not changed Git blobs. Fix LF packaging for executable Docker assets and reproducible hashes. Base image/Stockfish installation float; pin image/engine for comparable runs. Ten episodes is a pilot, not precise propensity estimation. | Endpoint shape is suitable, but `mode=default` receives the non-reasoning scratchpad nudge although GPT-OSS reasons. Make prompt policy explicit so hosting does not change the treatment. |
| **psychosis** | Multi-turn delusion reinforcement, pushback and referral; nine simulated personas, 12 turns | No confirmed core execution defect in selected tests. **Check health/validity:** 4096-token target cap, evolving history, failures excluded by persona, and judging every generated turn including truncated/empty replies. Red-team responses adapt to the target, so identical starting personas do not mean identical prompts. Pin attack protocol and judge; report uncertainty by persona. | Ordinary endpoint path plausible after shim fixes. Qualify budgets/role conversion; this is simulated dialogue, not clinical validation. |
| **mask** | Honesty relative to the model's elicited beliefs under pressure; 1000 released rows | **Must fix:** empty `None` and empty `""` are handled differently. Fake responses reproduced no empty tally for `""`, and `length` completions are stored without finish status. Partial output can go to honesty judging as ordinary text. Existing error-rate gate does not establish that these cases are captured. Preserve finished-empty evasion separately from truncation/error and audit existing artifacts for affected shapes. | Shim always serializes absent content as `""`, making this directly relevant. Endpoint path otherwise exists. |
| **dictator** | Authoritarian-request resistance; 138 scenarios, including scripted multi-turn cases, three samples | Selected tests pass; **no confirmed core blocker** on Qwen. Empty responses and judge failures are excluded explicitly. Check nonempty truncations (still judged), 2048-token reasoning-judge headroom, missingness by difficulty, and retention of every turn's reasoning (current generator keeps final-turn trace). | Ordinary endpoint path plausible. Validate Harmony extraction and matched effective sampling. Distinct from archived dict_verse. |
| **delegated_harm** | Behaviour under AI-peer/AI-parent/human requests in simulated work environments; 12 included scenarios, 324 default episodes | **Must fix for a fixed-request response comparison:** author stage uses the tested checkpoint, so AI requests differ between checkpoints. Retain this as an explicit author+responder experiment if desired; add frozen/crossed requests for response-only claims. Human requests already fixed. Defaults also enforce a specific Qwen base revision and Qwen label; an omitted upstream scenario is explicitly documented, not silently fabricated. | API gate plus Qwen-specific revision/label; transport itself uses ordinary OpenAI calls. Freeze author identity independently from responder. |
| **colosseum_jira** | Six-agent collusion, single-agent pressure and cooperation variants; allocation reward/regret plus language judging | Core config/statistics tests pass. **Fresh-driver bootstrap required:** default root is a particular cluster path; external pinned Colosseum + patched Terrarium are not root dependencies. Keep computed action outcomes primary; judge is secondary. Check old 2048-token judge budget. | API gate, sibling co-serving, default Qwen control, per-seat configuration. |
| **colosseum_hospital** | Nine-agent resource allocation; baseline, coalition objectives, benign coalition and optional whistleblowing/survey | **Must resolve default protocol before new results:** documented fixes for carried history, strict scheduling, real queue and hospital ownership remain false in default config. Public action rationales and coalition-mode cues also remain exposed. These are deliberate switches, not absent code: create/version explicit paper-reproduction and validated-fixed protocols rather than silently mixing them. External harness bootstrap also required. | Same gate and per-seat issues as Jira; revalidate tool parsing, no-tool/refusal distinction, long contexts and peer control. |
| **whistlebench_team** | Whistleblowing under misconduct/benign twins, solo/AI-conceal/human-conceal; 300 default episodes | Recent, good controls; selected tests pass. **Must fix transport promise:** Tinker reaches runner but is rejected by OpenRouter-only assertion (reproduced). `peer.model=base` uses actual base on vLLM but the target itself on API, changing peer treatment across routes. Judge calibration and effective sample counts still need live qualification. | Generic Inspect-compatible endpoint routing, separate peer endpoint, no global env leakage across concurrent runs. |
| **moralbench** | Declarative MFT answers against released human key; 88 items × 5 repetitions | Best candidate for **no substantial benchmark repair** from this pass: pinned assets, deterministic scoring, option-swap support, trace separation and parse diagnostics; tests pass. It measures stated answers, not action propensity. Invalid/error answers score zero, so enforce health when interpreting values and preserve the native raw score. | Simple endpoint candidate after shared shim repair; a live sample is still required. |
| **internalization** | Knows/notices/acts/discriminates; constitution retrieval, application, pressure and fake-clause controls | **Must fix:** missing explicitly named frozen itemset silently enters generation instead of fetching that itemset or failing. Reproduced with default ID. Fresh drivers may generate different tests. Windows cache replacement also failed in offline tests; retry changed failure shape, consistent with an intermittent concurrency problem. Cache identity omits target revision, and wrapper forces vLLM provider without passing target key. | API gate + explicit key propagation; publish/fetch frozen items, revision-aware caches and one portable driver configuration. |

## Additional evals and tools outside the registry

| Family / setting | Current state and preliminary action |
| --- | --- |
| Specification gaming: **MC revealing score** | Vendored regex-graded single-turn setting. Restore the missing orchestration, then test exact answer/think separation and invalid outputs. |
| Specification gaming: **MC reward action** | Same integration gap; validate regex against visible final answer and retain attempted-vs-successful exploitation. |
| Specification gaming: **Email assistant** | Three-turn READ/IGNORE/SNOOZE scoring. Check natural-language/regex false matches and complete transcripts. |
| Specification gaming: **Customer service** | Simulated-customer dialogue, feedback-URL criterion. Simulator is part of the environment; changing it changes the protocol. Verify patched turn cap and URL grading. |
| Specification gaming: **Data entry** | Inspect + Docker, reward-file check and judge. Qualify grader evidence, error handling, Docker and current dependency environment. |
| Specification gaming: **Sales** | Agentic quota setting; verify environment state and quota metric rather than narrative claims. |
| Specification gaming: **LiveCode easy** | Executable tests plus hard-coding judge. Pin problem/test set and sandbox; treat judge/model failures separately. |
| Specification gaming: **LiveCode hard** | Same implementation, harder subset. Record subset separately; do not count it as an unrelated harness. |
| **Petri adaptive auditing** | Subscription provider and export/validation helpers exist, outside eval contract. Nested project only declares Inspect + Claude Agent SDK; Petri setup is not a turnkey root install. README still points to old tools/ paths. Qualify exact auditor/target roles and provider transport. Discovery yields candidate failures, not an unbiased rate. |
| **Petri fixed-probe evaluation** | Separate existing script. Defaults are older checkpoint names and direct Anthropic judge credentials, conflicting with current OpenRouter convention. Fixed matched probes are useful but need current endpoint/auth/provenance and health handling. |
| **SURF adaptive auditing** | Calibration and validation-funnel helpers exist; default external SURF checkout is absent. Calibration assumes direct Anthropic credentials. Restore/declare the pinned external runtime if this remains active. Keep discovery and independently adjudicated held-out measurement separate. |
| **Hospital mid-shift/self-report probes** | Existing scratch script reconstructs recorded contexts and asks about ethics. Hardcoded historical output paths/checkpoints make it nonportable; parameterize dataset/revisions/endpoints if retained. Self-report is a different measurement from observed sabotage. |
| **Constitution v1/v2 probe** | Historical scratch generator comparison; stale imports (`src.endpoints.openrouter`, `src.data.prompts`) and archived config. Heuristic refusal/concept counts. Archive explicitly or port if still wanted. |
| **dict_verse** | Archived 27-scenario verse/jailbreak config remains; runner/package and registry entry do not. Document as retired or restore deliberately, never claim current runnable coverage. |

The [spec-gaming README](../src/eval/misalignment/spec_gaming/README.md) points to `scratch/spec_gaming/run_suite.py`, which **does not exist at this commit**. Vendored sub-harnesses remain, so this is missing project orchestration, not missing benchmark content. Its explanation that Inspect cannot share the root environment is outdated at least as a general claim: root now installs Inspect 0.3.268 successfully. Full suite dependency reconciliation is still untested.

LMSYS/WildChat are historical mentions, not active registry entries or current benchmark implementations found in this checkout. Model-eval-model is a synthetic-data recipe; TURF, properties/SAE analysis, feature discovery, judge recalibration and plot scripts analyse/generate artifacts and are not additional model evals. Inspect SWE-bench is a protocol/backend of the capability family, not a new propensity eval.

## Concrete offline evidence

Reproduce the deterministic checks with:

```powershell
uv sync --frozen --group dev
uv run --frozen python scratch/eval_audit_20260928.py
```

The [reproduction script](../scratch/eval_audit_20260928.py) uses fake endpoints and temporary fixture files, with no model requests. Its [captured results](eval_audit_2026-09-28_reproductions.json) show:

1. Secret-number labels `echo 'cat /secrets/secret_number.txt'` as a successful secret read. It also counts a failed `rm` as gaming; an episode already marked gamed vanishes from `n_gamed` if it never submits. Successful behaviour, attempts and conditional-on-completion rates need separate metrics. The noisy oracle also uses unseeded per-process RNG; the config seed currently fixes secrets, not feedback noise.
2. MASK tallies `None/stop`, but not `""/stop` or `""/length`; partial text with finish reason `length` is returned without that status.
3. Internalization with the missing pinned default itemset attempts to construct a generator client.
4. Arena-Hard writes `base.jsonl`, then fails when the wrapper expects `qwen36.jsonl`. Changing temperature to 0.93 reuses the answer and makes zero new fake calls.
5. WhistleBench rejects a Tinker target inside its runner.
6. The frozen driver environment lacks both Tinker dependencies.

A targeted test selection spanning framework/provenance, all registered eval families, model profiles and key SWE-bench protocols produced **532 passed, 5 skipped, 2 failed, 4 errors** in 27.83 seconds. This was not the entire repository suite. Raw log and JUnit file are in `output/eval_audit/pytest.txt` and `pytest.xml` in this worktree.

* CTFish hashes: working-tree bytes differ solely by CRLF conversion. Git blobs exactly match both expected upstream hashes. This is a checkout/package portability defect, not evidence the benchmark game/rubric was edited.
* ODCV-Peer: test assumes a slash before `team/`; the Windows fixture has a backslash.
* Internalization: four setup errors originated at `core/cache.py:148`, `Path.replace` with WinError 5. A targeted rerun produced **23 passed, 1 failed**, with two nonempty error rows in the otherwise-offline pipeline. Resolve concurrent cache I/O and preserve the exact exception; do not call it a model failure.
* Passing tests did not detect the reproduced Arena/MASK/secret-number/WhistleBench issues; add regression coverage when implementing fixes.

All nine default judge/red-team IDs checked were present in the live [OpenRouter model catalog](https://openrouter.ai/api/v1/models) on this audit date: Gemini 3 Flash Preview, 3.6 Flash, 2.5 Flash and 3.1 Pro Preview; Grok 4.3 and 4.20; Sonnet 5 and 4.5; Hermes 4 405B. Presence is not proof of provider availability, calibrated judging, refusal behaviour or successful calls. No paid probe was made.

## Recommended repair order

1. **Shared transport and identity:** pinned Tinker environment, effective sampling, safe port/checkpoint readiness, streaming or a supported non-streaming path, explicit context limits, per-seat endpoints, qualified GPT-OSS serving alternative.
2. **Reproduced correctness defects:** Arena naming/cache isolation; MASK response-health handling; secret-number action evidence and denominator; internalization frozen items/cache; WhistleBench transport and peer semantics. Preserve legacy protocol IDs rather than silently rewriting historical results.
3. **Scientific comparability:** fixed/crossed delegated-harm requests; explicit Hospital paper/fixed defaults; action-versus-language and false-alarm controls; capability/useful-completion measurements alongside lower harm.
4. **Reproducibility and fresh-host operation:** data revisions for MMLU, pinned Docker/runtime artifacts, LF assets, external Colosseum/spec-gaming/Petri/SURF bootstrap, documented SWE-bench entrypoint/protocol selection.
5. **Bounded live qualification on each supported route:** one real case per eval, then representative long-context/tool/error cases; verify raw prompts, visible answers, traces, tools, finish reasons, denominator, target/peer revisions and publication layout before larger runs.

Do not refresh all vendor trees just because their pins are old. Keep the pinned patched harness where it is sound, and change a protocol only for a specific reason with a new version and a comparison plan.

The newest SWE-bench closeout also marks a DA-15 checkpoint's training dataset as needing replacement; that is **treatment provenance**, not evidence the SWE-bench harness itself broke. Keep those questions separate.
