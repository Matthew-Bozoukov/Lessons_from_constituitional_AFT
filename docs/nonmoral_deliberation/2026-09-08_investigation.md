<!-- ABOUTME: Evidence from the prelaunch repository, artifact, control-design and cost investigation. -->
<!-- ABOUTME: Separates verified facts, proposed experiments, and remaining user decisions. -->

# Prelaunch investigation — 2026-09-08

Status: preparation only. No paid inference, rental, training, capability evaluation or
publishing performed. The [research brief](research_brief.md) remains authoritative.

## Findings that change the plan

| Finding | Consequence |
|---|---|
| Historical nonmoral result is 73/400 = **18.25%**, displayed as 18.2%; one checkpoint, five passes, 40 scenarios × two variants | Repeated rollouts do not establish training-seed robustness |
| Its exact adapter revision is still available, now public | Reuse the adapter; no reproduction training needed |
| Table-2-only and Numina-control adapters also exist and are public | Math already has a historical baseline; inspect/reuse before creating another math corpus |
| Table-2-only metadata describes ordinary SFT, whereas nonmoral uses dynamic batching; its provenance is partly backfilled | Treat it as a useful historical anchor, not a fully matched causal control |
| Old evaluation used 16,384 context and one Gemini Flash judge; current config uses 28,000 and two different judges, with progress judging on | Freeze one explicit new evaluation config and rerun the primary historical reference with the new arms |
| Old holdout helper reads the wrong ID field | It finds zero training IDs; 29 of its first 30 purported holdouts are training examples |
| Only **18** of 702 original corpus rows were absent from the 684-row synthetic training subset | Do not silently run its default 30-example manipulation check; reserve new held-out scenario families before generating training data |
| Old quality checks target rhetorical repetition and the helper scores override | Neither establishes validity of our broader nonmoral-deliberation construct |
| Current mixture defaults rebuild a 10,000-row source blend and run filtering | Preserve the exact historical 9,284 base rows for this comparison instead of inadvertently changing the base data too |

The split bug was verified against the dataset revision actually named by the adapter,
not just a local filename. No inference was run with the broken checker. Claude's handoff
reports the original manipulation check was never run.

Current files are at `output/nonmoral_deliberation_claude_dump.md` and
`output/nonmoral_deliberation/20260902_013651/`; the initial research plan's old worktree
paths describe their location at the earlier inspection.

## Recommended smallest useful experiment

Two new datasets, **one LoRA each**:

1. **Broad nonmoral comparative reasoning:** defensible choices, relevant trade-offs,
   complete single-turn deliverables; no requirement to override the user.
2. **Paired execution/verification CoT:** same prompts, decisions and full final answers;
   CoT carries implementation/checking rather than explicit comparison of alternatives.

Use the same selected scenario IDs, base rows, ordering seed, supervision rules and training
recipe in the pair. Provisional size: 684 synthetic rows each, keeping the historical mixture
fraction, rather than treating 684 as a theoretically privileged number. Freeze selection
without looking at new ODCV outcomes. Use explicit domain tags rather than force new tasks
into the old nine craft categories. No additional training seeds now.

Primary evaluations: those two new checkpoints plus the original nonmoral checkpoint,
all under one protocol. This gives a paired test of CoT content and a comparison of the
new overall recipe against the historical recipe. The latter changes multiple things and
cannot identify which selection improvement caused an effect.

Proposed protocol: current fixed harness/context, temperature 0.7, **five complete passes**
on the same 80 cells, one consistently pinned primary judge for MR and progress. This
keeps the old repeat count; it does not make old transcripts protocol-matched. Keep exact
judging prompts/provider/version and all results. Preselect a blind secondary-judge audit
sample before scores; define how disagreements affect claims before running that audit.
This is a proposal, not an approved launch configuration or a promise that every pass fits
the envelope. If runtime forecasts fail, reduce scope before observing comparative scores.

Optional fourth evaluation: existing Numina control, conditional on a pre-evaluation cost
forecast and provenance review. Existing DA and Table-2 results remain descriptive context
unless reevaluated under the same protocol. Defer new math training, short-CoT conditions,
and stakes conditions to follow-up; they need not all fit the first session.

The exact research question for the pair is: **Does additional comparative reasoning in
supervised CoT help beyond execution/verification CoT, holding the prompt and final answer
fixed?** It does not test absence of all deliberation. A shared final answer can still
contain a rationale. See the [six complete paired examples](paired_examples.md), including
math and game boundary cases. These authored examples have not passed an independent
manipulation audit and are not ready-to-train data.

## Dataset checks before training

- Every accepted row must be nonmoral under the user's wrongness criterion; record stakes
  separately. Reject detected moral conflicts regardless of their monetary magnitude.
- Require viable alternatives and situation-specific support for the selected choice.
  A reasonable alternative judgement is not an error. Flag fabricated constraints,
  invalid technical premises, and metadata that contradicts the actual prompt.
- Require exactly one user/assistant turn, a decision, full requested output, and compliance
  with hard task constraints. Conditional cases need a real missing condition and executable
  branches, not a request for another turn. Their actual quota remains a pilot decision.
- Check BOTH CoTs and the shared prompt/final answer. Blind reviewers to condition names;
  assess comparison content, substantive execution reasoning, factual validity, completeness,
  and moral content. Never accept a pair solely because a generator was instructed to contrast it.
- Measure paired token counts before selecting a matching rule. Avoid padding procedural
  traces. If content and length cannot be separated naturally, explicitly narrow the claim
  or redesign before training. Preserve equal row presentations; extra repetitions to equalize
  tokens are another intervention. Dynamic batching averages within each row before averaging
  across rows, so longer CoT also changes the trace/final allocation of supervision.
- Freeze a random human-review sample plus every flagged/borderline row; preserve ratings,
  revisions and rejection reasons. Audit passing is evidence, not proof of zero contamination.
- Build genuinely held-out scenario families for an in-domain manipulation check; do not
  optimize data against ODCV or score only override. This checks whether the trained tendency
  changed. It is distinct from the broader capability tests the user has deferred.

Main-report quantities: exact MR counts/denominators, task-submission rate, progress-score
distribution and threshold counts, paired condition differences with scenario-level uncertainty,
and all failures/retries. Do not treat 400 rollouts as 400 independent scenarios. Report the
fixed-checkpoint scope. No capability-preservation or overall-best claim before the later tests.

## Cost evidence and proposed envelope

Verified from the historical files:

- Nonmoral generation manifest records **$47.3602**: response writing $5.5550,
  rewriting $37.3510, corpus pattern checks $4.4543. It records 800 response calls and
  750 rewrite calls for 702 final rows. Those stage totals do not establish all-in cost:
  earlier scenario/prompt stages and pilots are absent from this usage breakdown.
- The 400-rollout nonmoral result records **$1.6414 MR judging** with Gemini Flash.
  This excludes GPU serving and progress judging. It cannot be substituted for total eval cost.
- Current ODCV YAML comments estimate $7.60 per 80 transcripts for its two-judge MR+progress
  configuration: $38 over five passes before serving. These are code comments, not a current quote.
  Choosing a protocol implicitly by taking defaults can therefore alter both measurement and cost.
- RunPod's public pricing page checked Sept 8 lists H200 $4.59/hour, H100 PCIe $2.89/hour,
  and H100 SXM $3.49/hour. These are indicative listed rates, not a reservation or verified
  rate for the exact provisioned hardware. Billing includes startup and idle time.
  [RunPod pricing](https://www.runpod.io/pricing).

Proposed stage ceilings, **not expected prices or assurance of completion**:

| Allocation | Ceiling |
|---|---:|
| Pilot, generation of shared scenarios/final answers, paired traces, and dataset review | $100 |
| Two LoRAs, one seed each, including startup | $50 |
| Serving + MR/progress judging for the three primary evaluations; fourth only if forecast fits | $80 |
| In-domain manipulation check and independent judgement audit | $20 |
| Failure/cleanup reserve | $40 |
| **Total planned maximum** | **$290** |

Before full generation, measure cost per accepted PAIR in a small pilot, including discarded
pairs, retry costs and all stages. Before training, use actual token lengths and measured recipe
timings to forecast the two fits. Before full evaluation, measure operational throughput without
using early MR scores to decide which conditions deserve more evaluation. Do not silently spend
the reserve on scope expansion. Keep committed remaining cost + accrued cost + cleanup exposure
within the session ceiling. If an essential stage cannot fit, pause it and report the feasible
revised package; no automatic increase or scientifically invalid half-comparison.

The synth cost helper currently prices unknown models at zero. Launch preparation must validate
every selected model/provider's price and reserve for in-flight requests. A per-stage tally is
not itself a hard global spending limit, particularly with retries and parallel workers.

## Overnight readiness

Subsequent Sept 8 update: the user started Docker; `docker info` now reports engine
29.6.2. The Windows liveness bug below is fixed locally in `src/infra/runpod.py` using
an observation-only process-handle wait. The launcher now explicitly detaches on Windows
and closes its local log handle. All 41 focused watchdog, managed-run, pod and serving
tests pass, including a real disposable child's live/exited states. No paid resources
were used. This addresses those local issues, not the remaining full overnight preflight.

Verified locally: Python 3.12.10 via `uv run --no-sync`; Docker CLI installed, but
`docker info` cannot connect to `dockerDesktopLinuxEngine`; current sleep-after settings
are zero on both AC and battery. No settings changed or applications started.

ODCV needs a working Docker engine here, not on the RunPod container. Need a tiny harness
preflight before renting evaluation hardware and a reliable driver host for the night.
The power setting alone does not verify Docker stability, network continuity or lid behaviour.

`src/infra/runpod.py` supplies watchdog helpers, but ordinary `up` does not automatically
supervise or terminate a pod. A long-lived owner must arm cleanup from provisioning onward,
use a lifetime/spend cap, persist pod ownership, and verify termination. Its current local
watchdog uses `os.kill(pid, 0)` for parent liveness: **unsafe on Windows as written**.
Python documents that Windows signals other than the two console-control signals terminate
the target process; zero is not a safe liveness probe there. This is source/documentation
evidence, not a test performed against a running owner. Replace and test the platform-specific
mechanism before use. [Python os.kill documentation](https://docs.python.org/3/library/os.html#os.kill).
Do not bind the owner to a short-lived CLI that exits immediately after provisioning.
No watchdog or autonomous scheduler was installed in this investigation.

## Questions that still require research intent

Update: the user accepted (1), agreed with the recommendation in (2), and tentatively
accepted (3). Retain the original questions below as the decision history; these are
no longer unanswered intent questions. Exact evaluation/audit settings and budget
feasibility still need preparation, and paid work has not started.

1. Does comparing alternatives in CoT versus reasoning about execution/checking in CoT
   address the desired baseline, given the shared answer still contains a rationale?
2. Should valid explicit user constraints always be respected, with departures restricted
   to genuine impossibility/contradiction or discretion granted by the prompt? Recommended
   yes. The old recipe's systematic override is a substantial unresolved behavioural choice.
3. Is the first paid package the two paired new conditions plus the historical reference,
   with shorter reasoning/math/stakes experiments sequenced afterward, or should a different
   question displace that pair? Recommended: resolve the first two questions with example
   review, then this narrow package, rather than a large grid.

## Reproduction and provenance

Read-only inventory script: `uv run --no-sync python -m scratch.nonmoral.inventory_readonly`.
Output: `output/nonmoral_investigation/20260908/inventory.json`, `local_audit.json`, and
small fetched HF provenance/result files. No weights downloaded. Corpus SHA256
`be21271e4ed73df064931279d2b52c63d1b5abb1374b5c80ced597bbd86c984d`;
mixture SHA256 `0517ef85d288f14e42bc77f371d3e2e48866879b5824984feaf4a7451ce60561`.

Local fixture verification: executed the complete Python function and checked empty,
scalar and nested-list cases; checked the timetable totals and six parameter values;
checked the odd-number identity for n=1..100 (in addition to the written proof); solved
game states 0..10 by backward induction and confirmed taking two is the unique winning
move from ten. These are artifact checks, not model capability evaluations.

| Artifact | Verified revision |
|---|---|
| Nonmoral adapter | `2225547cec8bd312a1e025f02fb6b3321c047e4a` |
| Nonmoral training mixture actually used | `6364505df02b0020b030bf379bd42285a14de6a5` |
| Nonmoral 5-pass eval | `1fcdacdc38c2850a60e7eb08532d40019e3f5b1b` |
| DA principle-scoped 5-pass eval | `77f77480585769246aa75cd11aac678b6ef7c062` |
| Table-2-only adapter | `2c513ea7513baf792bd2becf0900b5c9d858c92d` |
| Numina-control adapter | `edfb4287c10f553c541ba28216f202d0c0f47055` |

The inventory records full HF IDs. Table-2 training metadata's referenced config was
not present at its recorded Git revision; its fetched `run_meta.json` supplies the recipe
but includes `git_sha: nogit`. Numina training metadata also says `nogit`. Preserve these
provenance gaps rather than describing all historical arms as exactly reproduced.
