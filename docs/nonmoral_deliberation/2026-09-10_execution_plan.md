<!-- ABOUTME: Revised, reviewable next nonmoral experiment with bounded generation, training, evaluation and preservation. -->
<!-- ABOUTME: Supersedes the framing/shortening and reasoning-only drafts; execution requires user approval. -->

# Nonmoral deliberation: revised execution plan

Proposed on 2026-09-10 after challenging the first draft. **No paid work has started.**
A later “execute this plan” authorizes the scope below, incorporating any user edits.
This is the one current execution proposal; both the earlier C/S package and its
reasoning-only replacement are superseded. The final revision followed inspection of
actual selected training rows, not only further speculation about possible mechanisms.

## 1. What was wrong with the first draft

1. **Correction was poorly operationalized.** Adding “I tentatively prefer B” while
   keeping the assistant text unchanged chiefly tests a prompt-framing intervention.
   It does not establish that we have strengthened constructive disagreement, and its
   permission to choose otherwise differs from the original dataset's forced overrides.
   Repeating a conspicuous wrapper could itself create a surface cue.
2. **The experimental priority drifted.** Shortening the weaker broader dataset is a
   legitimate ablation, but less directly aimed at the user's primary objective than
   improving the quality of its deliberation. It should not automatically consume
   half the available training/evaluation allocation.
3. **Words were standing in for substance.** Original reasoning averages about 469
   words versus239 for broader, but we have not measured that the extra words contain
   better weighing of alternatives. “Make it longer” could add rationalization and
   false claims, precisely one of our earlier failure modes.
4. **The generation caps were allocations, not demonstrated costs.** We had not
   established that two full Opus transformations plus review would fit $33. Stop
   rules limit loss; they do not guarantee a useful dataset before the deadline.
5. **Half-transformed arms weaken interpretation.** Retaining originals is a useful
   recovery policy, but a 50% transformation threshold could produce very different
   intervention strengths across arms. A null result would be hard to interpret.
6. **Operational work was expanding again.** Adding concurrent preservation for two
   trainers and multi-target evaluation ownership could become an infrastructure
   project. One candidate can use the existing one-arm protected paths.
7. **A clean mechanism claim was still unavailable.** Historical controls, one SFT
   seed, changed authoring style and changed reasoning-token exposure prevent us from
   attributing an outcome purely to “deliberation depth.” More controls are needed
   eventually, but the next result can honestly be a recipe screen.

8. **Freezing final answers could preserve known defects.** On a small fixed local
   sample, broader_batch13_046 asks for just the finished four-line poem, but the
   answer adds explanatory notes. Its reviewer explicitly waived that instruction.
   In broader_batch06_029, the rationale mostly compares truncation with wrapping
   that the fixed layout forbids, despite the source also permitting abbreviation.
   These are concrete reasons to permit coherent revision of both rationale and
   answer. They are not a measured corpus-wide error rate or an explanation of MR.
9. **Demanding hundreds of "improved" rationales creates pressure to invent improvement.**
   Distinguish successful revisions, textual changes and demonstrated improvements.
   Do not accept a judge's vague better/worse label as proof of quality or force
   useful additions where a short explanation already suffices.

The two inspected examples were confirmed in the exact local training-mixture bytes
matching the pinned SHA256 below. Their rendered-text hashes are
cb572ff2acfffb72980111094076944e4610b74da60e76f2e0f04ae285290d97 and
07dda839f7efd862af9f94e4573ba65221c4b4187f8537dbfe2fe67ec97e6bef respectively.
The existing corpus has not been edited.

These criticisms change the funded experiment, not just its wording.

## 2. Funded scope and scientific question

**One revised broader dataset → one seed-0 LoRA → one full common-protocol ODCV run.**
A small, strictly bounded stakes pilot can run during the main GPU work.
No paid shortening arm, prompt-wrapper arm, training-seed sweep or formal capability
evaluation is included.

Candidate style: **nonmoral-grounded-revision**.

Question: can one grounded revision pass over the existing broader conversations
produce a more aligned checkpoint, while preserving the prompts and satisfying their
full deliverables? The intervention covers reasoning and answer fidelity together.

Primary reference: existing broader model,76/240 misaligned (31.67%).
Performance target: existing original nonmoral model,33/240 (13.75%).
Math92/240 and Table2-only90/240 remain contextual baselines, not new rentals.
The original's older18.25% is a different-protocol result.

This prioritizes an improved recipe. It does not claim to resolve why the original
worked or prove that broader domains themselves caused the regression. Reusing an
existing reference saves a LoRA; it also leaves historical-run and training-seed
uncertainty. An API model identity is not an immutable weight revision.

## 3. Exact data intervention

Start from the 684 selected broader conversations. Keep every retained user prompt
and system message unchanged. Preserve all 9,284 replay rows byte-for-byte at their
original positions. Make one revision pass over each complete assistant response:
reasoning plus final answer. This is the repository's familiar revise_responses
operation, not a new scenario-generation pipeline.

The revision instruction prioritizes:
- faithful comparison of viable options against criteria actually supplied;
- a defensible choice with its consequential tradeoff explained;
- complete fulfillment of the requested artifact and explicit output instructions;
- consistency between reasoning, final answer and source facts.

These are review criteria, not mandatory headings or a fixed rhetorical template.
Keep a reasonable original decision and correct artifact wherever possible. Correct
an indefensible decision or defective output when the complete prompt supports doing
so; record the change. Do not freeze an error or force an argument for an old answer.

A subjective choice need not be uniquely best. Math, code, geometry, games and
practical calculations remain eligible. Do not reject legitimate disclosed assumptions,
hybrid solutions or ordinary stylistic disagreements. Do not invent drawbacks of the
other option, new facts, missing code, morality, refusal language or elevated stakes.
In particular, "just/only the artifact" is an instruction: helpful explanatory extras
do not excuse violating it. General optional extras remain permissible when they do
not contradict the user's actual request.

**No minimum word count and no quota of newly invented reasons.** Remove redundancy
when that improves the example. Preserve an already adequate short explanation.
A new factor counts as grounded only if its supporting source fact or legitimate
assumption is identified. Do not turn every choice into a complicated dilemma.

Opus sees the original conversation and can revise it coherently. These remain
synthetic worked rationales, not verified records of its actual decision process.
Unlike the previous draft, the author is not forced to rationalize a fixed final answer.

This is a compound recipe intervention: reasoning, answer fidelity and authoring style
can all change. It is not a pure depth/length study, no-comparison control, masking
ablation or generator comparison. Both real reasoning and full answers are supervised.

### Admission and exact changes

One author revision and one independent Sonnet review per source. If the revision is
unusable but the original remains acceptable, retain the original explicitly. Never
fall back to a known error merely to preserve the 684 count.

After that review, allow one isolated objectively verifiable literal correction,
with an exact diff and one final independent review. No second semantic revision pass
or central-rationale repair campaign. Removing unwanted surrounding commentary can
be literal when the complete requested artifact is already correct; changing the
substantive solution is not a literal repair.

An incomplete prompt is excluded, never completed by us. If both response versions
are unusable, replace that source slot from the 21 previously accepted, unselected
broader examples in frozen hash order. Revise/review that replacement under the same
cap. No fresh sources and no duplicates. If the reserve is insufficient, stop before
training rather than silently change dataset size or replay ratio.

Process the source pool once. Require at least 512 valid, textually changed revisions
among the 684 final synthetic rows; the rest can be acceptable originals. This is
only an intervention-coverage gate, not a claim that512 examples are objectively
better or a power calculation. Separately record identical returns, reasoning edits,
answer edits, decision changes, source replacements and independently identified
improvements/regressions. The pilot and final sample must demonstrate concrete useful
changes, not just rephrasing, before a GPU is rented.

## 4. Preparation and pilot

1. Read the current CLAUDE.md, GOTCHAS.md, BASELINES.md, experiment card, validity review,
   process reset, SynthDoc README, stakes notes and HF comparison contract. Sync the
   execution branch without overwriting unrelated changes. Do not edit CLAUDE.md,
   AGENTS.md or docs/TODO.md.
2. Reconcile only our ledgers, including unknown reservations; check available credit,
   existing pods, local disk and HF organization without exposing secrets. The account
   is shared. Do not attribute others' charges or terminate their resources.
3. Retrieve and hash-check the pinned corpus/mixture/model/eval below. Validate source
   membership, reasoning parsing and exact replay positions. Source files stay immutable.
4. Spend at most45 minutes inspecting a small fixed sample from both existing corpora
   and checking the revision prompt against16 sources. The two examples already
   inspected are development fixtures, excluded from the fresh pilot. Keep it bounded;
   do not start another paid corpus-wide audit or assume the small sample is representative.
5. Use the existing engine and protected one-arm runners. Add only the minimum
   experiment-specific preparation, selection and publication adaptation in scratch.
   No new generic orchestration framework or multi-arm owner.
6. Test locally before any API call: rendered prompts/escaped braces, exact tagged
   field names, source identity, immutable fields, request limits, ledger locking,
   checkpoints/resume, naming and backup barriers. Offline validation makes no model
   calls; SynthDoc smoke calls count against the data allocation.
7. Run a16-example paid pilot. Opus revises reasoning and the full answer; Sonnet
   reviews source, old response and revision. Locally inspect all 16 pairs. Require
   at least 12 valid changed revisions, including at least four with a specific useful
   correction or grounded improvement; ties do not count as improvements. Reject
   revisions that introduce material errors and leave no unresolved material error in
   retained examples. These are feasibility gates, not statistical quality guarantees.
8. If a concrete common defect is found, allow one prompt change and one fresh16-row
   pilot. Otherwise stop this candidate. A failed pilot does not unlock another
   research direction or a series of prompt experiments.
9. Estimate full production cost from observed author/reviewer token usage, the observed
   yield and a30% planning margin. Scale only if the remaining source passes fit the
   $50 data cap while training, evaluation and recovery remain reserved. This forecast
   is uncertain; the per-request cap remains authoritative.

Preparation has a two-hour ceiling. Paid generation and review have a four-hour
wall-clock ceiling from the first pilot request. These are loss-limiting deadlines,
not promises about external service speed. Once reached, preserve partial work and
finish with an honest blocked/incomplete result rather than silently lower standards.
The pilot 16 rows are retained when they pass; no needless regeneration.

## 5. SynthDoc implementation and bounded review

Live engine: src/data/synth/ours/pipeline.py.
Public entrypoint: uv run synth run --config <yaml>.
Thin script: scripts/data/synth/build_dataset.py.
Use method: ours. The separate deliberative_alignment method supplies a constitution
and is not this experiment.

Declare the sequence in an undated YAML config under configs/data/synth/:
locally pinned load_source_run → llm_tagged revise_responses → one independent review
→ deterministic admission and chat export. The source loader can load the entire
snapshot regardless of total_scenarios, so stage an actual16-row file for a pilot.
Keep originals, outputs, hashes and review decisions in separate snapshot fields.

Author: anthropic/claude-opus-4.8, low reasoning effort, temperature 0.7.
Reviewer: anthropic/claude-sonnet-5, temperature 0.2, preserving the working recipe's
reasoning setting. These are the final production identities, not stale top-level
defaults. Use the repository OpenRouter client/provider pins; verify live availability,
prices and safe diagnostics. Do not silently switch to Haiku or another provider/model.
Allocate output headroom for reasoning plus the requested fields; never equate a
short visible answer request with a safe tiny max_tokens limit.

Reuse scratch/nonmoral/pilot.py::CappedClient around the shared engine, as
broader_data.py already does. The old pilot CLI has narrow assertions and cannot be
launched unchanged for this corpus. Adapt the existing scratch driver rather than
duplicate the framework. The engine's stage-level budget guard alone does not cap
a stage full of concurrent calls.

One durable ledger reserves every request before dispatch and retains conservative
charges for unknown failures. Maximum eight concurrent requests. Use interactive
calls, not24-hour batch jobs. Allow at most one extra attempt for a confirmed
formatting/transient error; count nested parser/client attempts in the same limit.
No semantic/lint regeneration retries, blind refusal retries or repeated rejudging.
Resume saved successes and partial checkpoints; do not delete partials or invoke
unbounded top-up to repay completed work.

Every revised conversation gets one independent review. Review source consistency,
nonmorality, explicit instruction compliance, viable-option comparison and output
correctness separately. The old response is context, not a gold answer. The reviewer
must cite a specific change and its source support when claiming an improvement.
A tie is acceptable; unsupported praise is not improvement evidence.
Do not use “could a third option exist?” or mechanically verifiable content as
rejection criteria. Don't demand a uniquely optimal answer to a subjective task.

After deterministic identity/length/mask checks, locally inspect all flags and48
accepted revisions in a fresh sample selected by a precommitted seed. Inspect substantive
source support, not just whether the reviewer said pass. A sample pass cannot certify
all 684 examples. Remove identifiable systematic defects from the complete affected
set; if their extent cannot be bounded within the same deadline, do not train.
Only the stated literal repairs get one final independent review. No second full
manual census or corpus-wide paid rejudge. In the 48-pair sample, require at least 12
specific useful changes with supporting evidence, and no retained unresolved material
error. This prevents a GPU run on rewording alone; it does not certify the whole corpus.

Freeze admitted rows, unchanged fallbacks, replacements, edits and corpus hashes before
ODCV. No selection uses benchmark prompts, scenario IDs, outcomes or failure examples.
Run this one candidate once; a negative ODCV result does not trigger another overnight
dataset-tuning loop.

## 6. Pinned parents and HF naming/publication

Broader mixture:
dougalldeepmind/2026-09-09-nonmoral-broader-7-mix
revision f1e61baf643c861920303c7ba1e9844df5f6ed48, file mixture.jsonl
SHA256 0545e014b518fdb9b8b40e37adc0b4a21da01c384553fe60e6a81f87098a2c22.

Broader corpus:
dougalldeepmind/2026-09-09-nonmoral-broader-synth
revision a3d266e2f0cc48e26e153caf078a5d641ecbbb5c.

Base Qwen/Qwen3.6-27B:
revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9.

Broader reference adapter:
dougalldeepmind/2026-09-09-qwen36-0-nonmoral-broader-7
revision d52838446ef134088841e8dc436094d93827487f.

Broader reference evaluation:
dougalldeepmind/2026-09-09-odcv-qwen36-0-nonmoral-broader-7
revision fe7b98403d7efca11764fc94a5e7443b720c77ee.

Original reference adapter:
dougalldeepmind/2026-09-02-qwen36-lora-table2-9284-nonmoral-deliberation-684-rank-64-dynbatch
revision 2225547cec8bd312a1e025f02fb6b3321c047e4a.
Original common-protocol evaluation:
dougalldeepmind/2026-09-09-odcv-nonmoral-lf-common-3x
revision 020266bea9bfdb86bc186e5951fea1a2e8fee1ab.

| Stage | Actual existing implementation |
|---|---|
| Corpus | SynthDoc pipeline and src/data/synth/ours/hf_cache.py |
| Mixture | src/data/mixture/build_mixture.py contract; scratch/nonmoral/publish_broader_mixture.py for exact replay preservation |
| Training/adapter | src/train/train_lora.py, uv run train, scripts/train/train_lora.py |
| Evaluation | src/eval/run_eval.py, uv run evals |
| Comparison | scratch/nonmoral/publish_broader_comparison.py and the general visualizer contract |
| All naming/uploads | src/naming.py builders and src/infra/huggingface.py gates/helpers |

Adapt the existing mixture publisher to the new manifest; do not blindly run its
old hardcoded source paths. Preserve684 synthetic slots plus9284 byte-identical replay
rows and their positions. Do not rebuild the historical mixture with a fresh shuffle.

Use synth_name, mix_name, model_name and eval_name. Do not handwrite an hf_repo
override. Current code orders the model name as date-model-seed-mixture subject,
even though some prose examples show an older order. The new adapter will therefore
include qwen36 and its seed automatically; the model-agnostic synth corpus records
Opus/Sonnet provenance in its card, not as a falsely claimed training-model identity.

All uploads are public under dougalldeepmind, via the shared publication contract:
constitution:none, actual generating commit/command, schema, model identity, sampling
settings and available source revisions. Include all stages, edits, failed candidates,
costs and admission reasons. Development-only work cannot be discovered as a completed
training corpus. Retrieve each published exact revision and verify required file hashes.

Compare the broader adapter's resolved training configuration and relevant code/rendering
facts before rental. Prefer the same scientific recipe and recorded trainer revision
where necessary. Material unresolvable training or evaluation drift blocks a matched
comparison; do not hide it or add a new paid reference without authorization.
Code-only diagnostics and fixes are permitted without further questions.

## 7. SFT: one LoRA, two H200s

Use scratch/nonmoral/train_pair.py's existing one-arm plan interface and
scratch/nonmoral/result_backup.py. Only src/infra/runpod.py provisions GPUs.

Before rental: final mixture public/pinned, local row validation passed, source commit
pushed to origin, SSH key present, sufficient local storage and complete cost/recovery
reservation. Register the watchdog at provisioning, before bootstrap. Check provider
rate including storage, real two-H200 hardware, CUDA-compatible driver and
torch.cuda.is_available(). Test network throughput early and abandon an unusably slow
host within the cap, rather than wait an hour for downloads.

Launch both GPUs for the one model:

    uv run torchrun --nproc_per_node=2 scripts/train/train_lora.py --config configs/train/sft.yaml model=qwen36 seed=0 wandb=false constitution=none data_repo=<generated-repo> data_revision=<verified-sha> base_model_revision=6a9e13bd6fc8f0983b9b99948120bc37f49c13e9

Preserve bf16 LoRA (not4-bit QLoRA), rank 64, alpha 128, dropout 0.05, one epoch,
global batch 16, lr1e-4 cosine, warmup 0.05, weight decay0.01, max sequence8192.
Dynamic batching uses the measured H200 padded-token budget8000; verify actual
world_size2 and resolved budget. For9968 rows expect623 optimizer steps.
No hyperparameter optimization or additional seed.

Verify length on all rows and labels on every synthetic row. No silent truncation.
The shared mask_gate must pass; masking.py supervises real reasoning and final answers
while masking prompts and generation prefills. Do not use TRL's all-zero assistant
mask or full-sequence loss. Keep reporting disabled through wandb=false.

The longest replay row may exceed the 8000 microbatch target yet fit the 8192 sequence
ceiling; preserve the existing measured single-row handling, not a new truncation rule.
Rewritten synthetic traces must remain within the recipe's validated limits.

Check actual LF bytes and bash syntax before remote execution; use fixed byte-oriented
SSH stdin and detached launches. Monitor first real steps, finite loss, gradients,
VRAM, checkpoints, PID/birth identity and progress. A completed loss log is not a
verified published adapter. Infrastructure recovery may resume the same checkpoint
within the cap; it cannot create another seed or rerun a completed model.

## 8. ODCV: local CPU/Docker, RunPod inference

Use the existing one-arm protected plan in scratch/nonmoral/overnight_baseline.py.
The actual harness/serving/publication entrypoint remains uv run evals.

Before inference rental: Docker Desktop/Compose work, physical shell build inputs
have LF endings, a Docker bash syntax check passes, at least 16 networks are available,
disk/RAM suffice, and the selected tunnel port is free. Do not prune others' resources.

Provision the profile's inference GPU, normally one H10080GB, via runpod up --eval.
Register its watchdog. Drive the evaluation on this Windows host:

    uv run evals --name odcv --target <published-adapter> --server <owned-ssh-address> --config <frozen-common-config>

run_eval owns vLLM and its tunnel; do not independently start another server.
Check container-to-host reachability, thinking mode from adapter metadata, qwen3
reasoning parsing, qwen3_xml tool parsing, reasoning carry-over and nonempty transcripts.
Keep evaluator/judge credentials local. Docker never runs inside the RunPod pod.

Freeze40 scenarios ×2 variants ×3 passes =240 rollouts, temperature 0.7,
context 28000, concurrency 8, scenario timeout 2400s, judge workers 4, progress enabled,
no exclusions or extra system preamble. MR and progress both use
google/gemini-3-flash-preview as in the reference comparison.
The current default ODCV YAML has different judges and two passes; do not inherit it.
Hash the material harness/template/protocol files and check against the reference.

If the historical judge is unavailable, retain rollouts and report judging blocked;
do not silently substitute or rejudge old datasets at unallocated cost.
Retain all valid outcomes, token-limit/partial traces and infrastructure failures.
Use only the existing bounded recovery for missing/empty cells; never rerun an
unfavourable valid outcome. Missing coverage remains missing, not a fabricated240/240.

## 9. Interpretation and deliverables

Freeze candidate-minus-broader as the primary contrast and candidate-minus-original
as the stronger performance comparison. Report both regardless of direction.
Use existing paired scenario-cluster uncertainty, grouping both variants and three
passes within each of 40 scenarios. Three eval passes are not three training seeds.

Publish exact MR counts, differences/intervals, submission and progress, token limits,
empty outputs, terminal API deaths, cycle limits and missing judgments.
Checkpoints with lower MR are promising only provisionally. Any evidence of increased
refusal or reduced capability disqualifies acceptance; observed proxy deterioration
must be investigated, not excused by an invented tolerance. ODCV completion/progress
cannot demonstrate preservation of general capabilities. Formal capabilities and
training-seed replication remain deferred until separately authorized.

Reasoning, final answers, supervised-token exposure and authoring style can change.
An improvement would support this revision recipe, not prove that depth, verbosity,
instruction fidelity or any one ingredient caused it. No claim of faithful internal
reasoning follows from these synthetic traces. A later ingredient ablation should be
chosen after inspecting the actual measured changes, not promised in advance here.

Update the HF comparison producer, not frontend hardcoded experiment data. Publish
results/dataset_comparison.json with the dataset-model-comparison tag and complete
immutable references. Include changed-row fraction, source replacements/answer repairs,
reasoning words and tokenizer tokens, supervised-token counts, final-answer/prompt
preservation, system-message counts, domain coverage and actual audited properties.
Design intentions and measured observations are labelled separately.
Validate through the frontend parser, provide real dated figures and a brief results
note, update the experiment card and experiment LOG, then test, commit and push.

## 10. Secondary stakes pilot, without blocking the main result

Only after primary data is frozen and its full GPU/eval reservation is secured,
run the existing stakes worktree's generic shared-engine workflow while SFT proceeds.
Scope:12 source pairs,24 independent Opus answers, one Sonnet review per pair,
$5 maximum. At most the globally bounded formatting retry; no new semantic-pilot loop.

Use natural differences in potential loss of one's own hobby resources, accumulated
single-player progress or irreplaceable personal work. Keep task structure, options,
probabilities and preferences fixed. Do not tack on an unrelated fund, change difficulty,
introduce effects on others/livelihood/safety, or tell the high arm to deliberate more.
Answers may choose differently or reason longer; these can be effects of stakes.

Inspect every pair. A usable pilot requires at least 9/12 valid complete pairs with
a real consequence manipulation and no unresolved material error in retained data.
If it fails, archive and stop. Its result is feasibility and cost/yield, not evidence
that stakes affect alignment or that they have no effect.

Full low/high training requires a separate corpus and two additional LoRAs/ODCV runs.
The existing broader corpus is not a measured low-stakes arm. No full stakes run is
included or implied by a successful pilot.

## 11. Budget and unattended behavior

Recorded prior exposure:165.230118 dollars; remaining under the cumulative300-dollar
ceiling:134.769882 dollars. Reconcile before execution; these are estimates/reservations,
not an invoice. Shared account balance movement is not attributable experiment spend.

| New allocation | Hard cap |
|---|---:|
| Main pilot, full-response revision and review | $50 |
| One 2×H200 SFT including boot/storage/backup | $40 |
| One ODCV including GPU/storage/judges/backup | $15 |
| Secondary stakes pilot | $5 |
| Recovery contingency | $15 |
| **Maximum new allocation** | **$125** |

This leaves about 9.77 dollars unallocated under the old ceiling. These are
caps, not predicted invoices. The previous115-dollar draft allowed less for author
output; allowing coherent final-answer revisions raises this proposal by 10 dollars.
The last broader SFT cost about 21.85 and its GPU+judge evaluation about 8.06.
Measure actual full-response revision/review cost in the pilot before scaling.

No training starts unless its complete evaluation and recovery are reserved.
Unused funds do not automatically authorize another candidate, control, seed or eval.
If reconciliation reduces funds, remove the stakes pilot first, then stop before
a primary launch that cannot be fully covered. Do not spend beyond300.

Publish durable state at least every minute: phase, unique-source counts, errors,
spend/reservations, owned PIDs/pods, last actual progress and backup receipts.
Send brief milestone/launch updates. For unattended execution, configure a15-minute
progress follow-up on approval and pause it at completion. Monitoring is separate
from the independent watchdog; the task owner's survival cannot be the only safeguard.

Two failed progress checks trigger diagnosis;15 minutes of unexplained paid idleness
triggers preservation/stop. Active downloads/checkpoint writes are distinguished from
a dead process. At most one justified infrastructure relaunch/resume within the same
allocation; no repeated rental lottery or silent scientific changes.

“Execute” covers these specified launches and routine repairs: announce and proceed
without another approval prompt. Routine failure choices are preserve, use a valid
unchanged source, apply the stated literal correction, skip the optional pilot or stop.
Outside this scope, do not treat30 minutes of silence as budget/protocol authorization.
Finish independent authorized work and report what could not proceed.

## 12. Preservation and cleanup

This revision uses one trainer, so no new overlapping two-arm backup feature is needed.
Keep the existing ordinary-teardown barrier. Before rental, calculate the work/recovery
window against the owner's actual conservative lifetime formula; a dollar allowance
does not automatically grant enough training time. The40-dollar single-run cap must
cover expected training plus at least 45 minutes of measured recovery and startup.
If projected time does not fit, allocate recovery contingency before launch or stop.
Do not silently extend a running watchdog's budget.

Fetch/hash-verify final adapter/tokenizer/configs, resolved metadata, full logs and
all retained checkpoints including optimizer/scheduler/RNG state. Verify the public
adapter at its exact HF revision. Prefer per-file receipts/resumable transfers; never
claim a truncated archive is a complete backup. Checkpoint files must be complete
before copying; do not copy an actively written optimizer state as a valid recovery.

ODCV already writes rollouts/judgments locally; verify their local and HF inventory
and fetch the remote server/launcher logs before normal inference teardown.
A generic “pushed” log line is not a completion marker.

If preservation fails, block ordinary teardown and use reserved recovery time.
A hard independent budget watchdog can still require emergency termination; prioritize
irreplaceable outputs and explicitly record any loss. Unlimited retention cannot be
promised alongside a hard spending cap.

Terminate only our pods through the shared RunPod path. Verify provider absence,
check for owned orphans, report unrelated pods without touching them, reconcile spend,
then stop our watchdogs/monitors. Final report states results and limitations, failed/
skipped work, public links, preservation receipts, spend and remaining owned activity.

## Documents and code checked

- [CLAUDE.md](../../CLAUDE.md), [GOTCHAS.md](../GOTCHAS.md), [BASELINES.md](../BASELINES.md).
- [Current evidence and cost](experiment_card.md), [process failures](2026-09-09_process_reset.md),
  [validity review](data_validity_review.md), [Opus findings](2026-09-09_opus_pilot.md).
- [SynthDoc](../../src/data/synth/ours/README.md), [SFT recipe](../../configs/train/sft.yaml),
  [Qwen profile](../../configs/models/qwen36.yaml).
- [Comparison contract](../../dashboard/docs/DATASET_MODEL_COMPARISONS.md).
- The existing stakes worktree's docs/nonmoral_deliberation/stakes_experiment.md.

BASELINES.md warns against recipe ranking without training-seed replication.
The user's one-LoRA preference governs this screen; it does not remove that limitation.
