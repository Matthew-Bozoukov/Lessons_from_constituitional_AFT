<!-- ABOUTME: Separate secondary experiment on the magnitude of nonmoral stakes. -->
<!-- ABOUTME: Defines matched task cores, cost allocation and exact training/evaluation requirements. -->
# Nonmoral stakes experiment

Owner: `codex/nonmoral-stakes` in the separate `teaching_claude_why_replication_stakes`
worktree. Main priority remains the broader nonmoral corpus, one LoRA per condition,
then ODCV. This lane asks whether **how much the chooser can lose** changes the
alignment effect of nonmoral deliberation SFT. The user's prior prediction is no effect.
No result in this document is a stakes-effect measurement.

## Intervention

Use the **same complete task, system message, supplied artifact, options and explicit
preferences** in both arms. Append a short low/high context paragraph scaling the
same personal consequence. Generate fresh reasoning and full answers independently
for both complete requests using the same Sonnet settings. A different decision,
response length or amount of checking can be an effect of stakes; do not equalize it.
Do not demand caution, deliberation, refusal or a particular decision in the stakes frame.

Low means a readily absorbed inconvenience; high means a serious loss of the chooser's
own discretionary resources, accumulated hobby work or scarce personal opportunity.
Neither option may wrong someone. No introduced client duties, deception, harm,
livelihood threat or fairness dilemma. Do not confuse stakes with morality, objective
task difficulty, response length, or whether an alternative is uniquely correct.

Naturalness has a limit: an unchanged core may not admit a plausible high-stakes frame.
Exclude that source rather than silently changing its purpose or supplying missing code.
A claimed absence of effects on others cannot cancel effects implied by the original task.

## First eight source pairs and paid scope

The eight complete offline-written design fixtures are in
`scratch/nonmoral/stakes/fixtures.py`; `prepare.py fixtures` renders both complete
requests. They are **Codex-written sources**, not Sonnet production output. The
generator writes only their answers and combined reviews. These pairs can be retained
as production candidates if valid, but are far too small and narrow for training.
Never repeat eight examples to fill a 684-row slot.

| Task | Low personal consequence | High personal consequence | Material check |
|---|---|---|---|
| Decorative mosaic | £12 replacement | £1,200 replacement, yearly hobby savings | 4×4, eight tiles of each colour |
| Imaginary-city zine opening | £8 reprint | £800 reprint, annual zine budget | Complete opening, at most four sentences |
| Toy simulation sampling | £6 compute batch | £600 compute batch, annual project funds | Six unique unseen points in [0,10]² |
| Single-player route choice | Lose ten minutes' collection | Lose a hundred hours' collection | One of two legal routes; exact probabilities |
| Whimsical Spanish art line | £5 replacement | £500 replacement, annual art savings | Complete Spanish line; no idiom facts required |
| Four-bar bass arrangement | £10 re-recording | £1,000 re-recording, annual studio savings | Four bars of four beats |
| Abstract print colours | £7 replacement | £700 replacement, annual artwork budget | Bijection of three colours and three shapes |
| Six-observation booklet | £4 reprint | £400 reprint, project savings | Every supplied observation exactly once |

**Those pound amounts are fictional scenario facts, not API/GPU spending.** The
initial paid allocation was **$5 total**, within the shared project ceiling of $300;
the later cumulative allocation and stopped production attempt are recorded below.
The initially proposed 16 pairs / 64 calls was narrowed before dispatch to
**8 pairs / 24 planned calls: 16 answers + 8 combined reviews**. No frame-author
call is needed because the fixed frames are already explicit. Format retries are
bounded by the shared engine and individually reserved; no semantic repair loops.

The owned ledger is `output/nonmoral_stakes/20260909_first8/spend.json` in this
worktree. A single-process exclusive lock prevents competing dispatches. Per-call
reservations survive uncertain failures; live pinned provider prices must match the
repository. Transport retries are disabled. No backup model. Sonnet's default
reasoning behaviour remains enabled by omission, matching the restored main recipe.

Source selection and substantive answer review are independent of ODCV. Local exact
checks cover dimensions, counts, unique points, bounds and bar durations when an
artifact is explicitly extracted; they do not establish all prose or moral validity.
Sonnet accepts are evidence to inspect, not automatic training approval. Retain all
original candidates and exclusions. Keep pairs only if both complete examples pass
material review; report pair attrition without an invented all-or-stop yield threshold.

## Reuse and independence

The old nonmoral corpus is locally available: 702 rows, exact file SHA256
`be21271e4ed73df064931279d2b52c63d1b5abb1374b5c80ced597bbd86c984d`, source
`LASR-Callum/2026-09-02-craft-tensions-nonmoral-deliberation`.
`prepare.py sources` verifies these bytes, preserves all original single-turn task
cores and system messages, and writes a hash-linked review template in stable
SHA256(seed:scenario_id) order. No old answer is copied into an arm. The historical
training mixture used 684 of these rows; using all 702 as a candidate pool does not
claim exact historical training membership or an independent held-out sample.

The first original row inspected requests missing schema information and contains a
team/sprint setting. It illustrates why the entire historical pool cannot be assumed
eligible for a private-personal-stakes intervention. The 702 candidates are prepared,
**not screened or approved**. The eight fixed sources avoid waiting on main while
that issue remains explicit. Later choose a sufficiently large common complete pool:
historical rows that actually fit, or a frozen broader pool. A new parent must be
declared and hashed; do not swap pools silently.

Historical moral low-stakes machinery already used `load_source_run`, then rewrote
contexts and regenerated responses. `configs/data/synth/da-lowstakes.yaml` records
domain relocation as a confound and warns that `total_scenarios` does not limit a
loaded source file. This lane reuses the same engine with exact input files and
preserves task cores, instead of importing its moral refusal/constitution prompts.
The original nonmoral recipe enforced instruction override and mostly artifact-only
consequences; those are historical properties, not current user requirements.

## Full two-arm study, after allocation

1. Freeze up to 684 valid source pairs (same IDs and order in both arms), without
   ODCV-based selection. Freeze pair rejection reasons before inspecting new scores.
   If fewer qualify, report the actual equal count and synthetic share; do not duplicate.
2. Build each mixture from the same byte-identical 9,284 Table-2 rows plus one arm's
   accepted examples, using the existing mixture path. Preserve full reasoning and
   assistant response supervision. Record source, render, mask and token-length audits.
3. Train **two LoRAs total**, one low and one high, seed0 each, base
   `Qwen/Qwen3.6-27B` revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
   Use `configs/train/sft.yaml`: rank64, alpha128, dropout.05, one epoch,
   global batch16, lr1e-4 cosine, warmup.05, max8192; RunPod **2×H200 dynamic
   batching**. Run both arms sequentially on the same owned pod unless a centrally
   approved schedule can reduce cost without changing the recipe. Use `model=qwen36`,
   `seed=0`, `wandb=false`, `constitution=none`, pinned `data_repo/data_revision`.
   Required git commit must be on origin before rental. Existing protected training
   and result-backup helpers are reusable; do not launch their old hardcoded arm paths.
4. Evaluate both adapters under the existing common ODCV protocol:
   **40 scenarios × 2 variants × 3 passes = 240 rollouts per arm, 480 total**;
   temperature.7, context28,000, concurrency8, timeout2,400s, thinking/bf16 serving.
   Model on RunPod; driver and Docker on this Windows host. Reuse pinned Flash
   MR/progress judges, four judge workers. Freeze all harness/template/model revisions.
5. Report low/high exact MR counts, submissions, progress, and within-scenario paired
   MR difference with scenario-cluster uncertainty (aggregate both variants/passes
   within each of 40 scenario clusters). A non-significant difference is not evidence
   of equivalence; report the interval. This has one training seed per arm and cannot
   measure training-seed variation. Capabilities remain untested until user involvement;
   completion is only the agreed rough over-refusal proxy, not capability preservation.
6. Publish full source data, LoRAs, transcripts, judgments and provenance under public
   `dougalldeepmind`. Verify complete required outputs locally, including checkpoint
   artifacts and full rollouts, before ordinary owned-pod teardown. Keep watchdog and
   recovery reservation within the allocated spending ceiling.

Cost planning, not a current market quote: 684 pairs may require roughly **$55–80
generation/review, $40 SFT, $15–25 ODCV**, about **$110–145 total**, with actual
yield and measured calls determining the next allocation. The $5 first batch is part
of that sum, not extra. **This estimate is not a reservation; only data generation was
later allocated $55 cumulative.** No GPU rental,
SFT, ODCV or full-corpus spending is authorized here; main has budget priority.

## Commands and readiness

Run from this worktree root. Parent `.venv` Python can be used without sharing output
or ledgers; `.env` is read from the main checkout at runtime and never copied.

```text
uv run python scratch/nonmoral/stakes/prepare.py fixtures --out output/nonmoral_stakes/<timestamp>_design
uv run python scratch/nonmoral/stakes/prepare.py sources --input <pinned-original-dataset.jsonl> --out output/nonmoral_stakes/<timestamp>_pool
uv run pytest -q scratch/nonmoral/stakes/test_prepare.py
uv run python scratch/nonmoral/stakes/run_first8.py answers --execute
uv run python scratch/nonmoral/stakes/run_first8.py review --execute
uv run python scratch/nonmoral/stakes/run_first8.py publish --execute
```

`prepare.py frames/answers/review` accepts hash-linked reviewed inputs and emits
shared-engine configurations. It makes no paid calls. Those generic prepared configs
have zero allocated budget; use a centrally capped launcher when a future batch is
authorized. First-eight launch is separately capped. Publication requires local review.

Offline checks: **12 passed** on 2026-09-09. Source preparation retained 702 originals;
fixture rendering produced 16 complete low/high requests.

## First-eight result — 2026-09-09

Completed **16/16 answers and 8/8 combined reviews**, 24 calls, no retries, no uncertain
reservations, **$0.609012 total**. Answer generation took 114.1s; reviewing took 75.9s.
Sonnet accepted all eight pairs. Local material review retains **four pairs** (mosaic,
zine, music, colours), holds **two** (translation offers extra variants; booklet uses
a weak textual analogy), and excludes **two**:

- Simulation: legal points, but false claims about which quadrants were unexplored
  and what the sparse observations establish.
- Game: legal choice, but invents a conversion from collection time to point value
  and argues numerical dominance without the required preference information.

No source or answer was repaired. All eight original pairs and reviews remain
available, including rejected ones. Exact grid, point-bound, bar-duration and supplied
observation-retention checks passed; these do not override reasoning defects.
The music pair chose quarter-note roots at low stakes and whole-note roots at high
stakes. This is an observation about two stochastic answers, **not an alignment
result or an identified stakes effect**. No training was run, and the four retained
pairs must not be duplicated into a training corpus.

Artifact: [complete candidates and local findings](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-nonmoral-stakes-first8).
The publication receipt in `output/nonmoral_stakes/20260909_first8/publication.json`
records the exact revision once the upload finishes. `review_first8.py` reproduces
local extraction checks and the separately recorded dispositions without model calls.
This initial $5 allocation was subsequently replaced by the $55 cumulative allocation
below; it is not an additional allowance.

## Historical-pool production — 2026-09-09

Root allocated **$55 cumulative**, including the first-eight $0.609012. The independent
production ledger is `output/nonmoral_stakes/20260909_production/spend.json`, capped at
$54.390988. Sources, answers and review use Sonnet through the shared synth engine;
each source/answer/pair is dispatched once, without semantic repair or rejudging loops.
Only transport/format failures use the existing bounded machinery; all costs remain
reserved or settled in the ledger. No GPU allocation has been granted to this lane.

All 702 original requests are screened before fresh low/high answers. Missing actual
code, schemas, rule systems or real observations are excluded; originally requested
fictional examples and new designs remain allowed. Positive complete/nonmoral screens
receive local source review in `scratch/nonmoral/stakes/source_notes.py`, with a
hash-linked full eligibility sidecar before answer dispatch. The true eligible count
will be reported even if far below 684. No repeats or threshold-driven relaxing.

Generated consequence drafts changed multiple things too often (deadlines, audiences,
prize opportunities). They remain archived but are not used. Root approved one fixed
frame for both arms: the user has a **£1,000 personal discretionary project fund** for
commissioning replacement work if disappointed. Replacement costs **£10** in low and
**£1,000** in high. The fee is the only changed text; original core and system remain
identical. These are fictional scenario amounts, not API/GPU budget. No external judge
defines a uniquely correct aesthetic choice. This intervention tests personal monetary
replacement loss, not every meaning of high stakes; professional source contexts can
still carry background consequences common to both arms.

Publication audit: first-eight is a developmental candidate archive, correctly kept
outside training-data discovery; no candidate default should silently include rejected
rows. Future retained low/high corpora use `synth_name` with undated style config stems,
`StageCache`, root `dataset.jsonl` as default, stage snapshots and `training_data_tags`.
Current shared naming does not put generator names in synth repo names; full Sonnet
model identity and provider details belong in card/run metadata. Model artifacts use
the base model and seed through the model-stage builder. No new ad hoc publication names.

## Stopped wrapper attempt: what failed and what remains

The source pass returned **699/702** complete screens. Local review read **207**
potential cores and admitted **119**, already **565 short of 684** before answer review.
The answer run was stopped at **57 saved complete answers / 27 complete pairs** after
early traces explicitly treated the fund paragraph as unrelated, incidental or a test.
A narrow keyword screen flags **12/57** saved traces; this is a diagnostic count, not
a calibrated manipulation score. Full examples are under
`output/nonmoral_stakes/20260909_production/manipulation_audit/`.

The concrete failure was attaching personal replacement costs to existing professional
requests without establishing an actual resource at risk from the requested decision.
For example, agenda title writing has no intrinsic commissioning transaction, so a
personal fund paragraph reads like an unrelated aside. In another pair, both answers
invented app buttons because the source review incorrectly treated a final UI-copy
request as complete without its actual controls. Sonnet's capacity to create scenarios
is not established as the cause. A later judge cannot repair either missing causal
connection or missing source facts.

Root requested a pause; the runner lacked a cooperative pause mechanism. Its verified
owned PID was stopped, saving completed checkpoints and retaining full charges reserved
for unknown/in-flight requests. The runner now checks `STOP_DISPATCH` before new paid
calls, letting future in-flight calls settle normally. The existing attempt has this
stop marker and cannot silently advance to review.

Exact exposure: **$17.735860 cumulative**, including first-eight. Production has
**771 reservations: 702 source + 69 answer attempts**. **13 unsettled calls retain
$1.623328** in full upper bounds, included in that exposure; actual bills for those
calls are unknown and no amounts were forgiven. `stop_receipt.json` and phase status
record checkpoint/ledger hashes. **No paired reviews, SFT or ODCV launched for this
attempt. No GPUs exist for this lane.** None of these 27 pairs is training approved.
This failure says nothing about whether naturally integrated stakes affect alignment.

The next bounded step is offline only: six complete proposed pairs in
`output/nonmoral_stakes/integrated_design/six_pairs.md`, reproduced by
`scratch/nonmoral/stakes/integrated_design.py`. Loss comes from the actual operation:
destroying one's glass/print, restoring one's paint/sounds, or losing private game
resources. Each low/high pair changes one numeric cost/time/points quantity only.
These are hand-written design fixtures awaiting root/user review, not production data.
No new source screening or paid generation is authorized for them.

Offline arithmetic checks: private-route expected final crystals are low **2,606 bridge
versus 2,792 cave**, high **2,408 bridge versus 2,000 cave**. The private word-puzzle
chain **COLD → CORD → CARD → WARD** satisfies all three one-letter transitions and
ends at **112 stars** in either arm. The latter deliberately checks whether a stated
loss should change behavior when success is fully verifiable; it is not evidence that
this easy puzzle provides rich deliberation. The four craft fixtures permit reasonable
preferences and require no invented aesthetic scores.

## Integrated craft trial — completed, no expansion

Root approved cases 1, 2, 4 and 5 only; the two calculable controls remain offline.
Four source pairs were frozen at SHA256
`a9dd9ab1fd8eb04b3004df4074555ea32592fdad2348ba1403e08385f870a1a5`.
Eight raw Sonnet answers returned. Seven parsed; low-paint mixed reasoning/response
tags and its retry was blocked without charge. Only the three complete pairs received
one review each: **11 calls, $0.244464, all settled**. Cumulative lane exposure is
**$17.980324**, including the previous unknown-call upper bounds.

The loss now appears in the actual decision reasoning. Sonnet accepted all three
reviewed pairs; independent full-content review retains **zero matched pairs**:

- Glass: both traces rewrite unranked tastes as an explicit equal preference, then
  claim risk minimization is the only justified choice. Either tool could have been
  defended; sameness of decisions is not the rejection reason.
- Paint: low tags malformed; both plans add towel/palette/scrap tools outside the
  supplied tool list. No paid review or repair of this incomplete pair.
- Music: low adds imperfect reconstruction as a new outcome; high says loading saves
  one minute when it costs one minute. Both bass parts pass note and duration checks.
- Print: low claims twelvefold greater reliability. Actual success probabilities are
  99% and 88%, a ratio of 1.125; the failure probabilities differ twelvefold. High is
  materially usable as an unpaired candidate, not a matched training pair.

All glass cell-count/connectivity and music chord-tone/four-beat checks passed.
`review_integrated.py` reproduces these checks and saves the full unedited requests,
reasoning, answers, raw malformed output, model reviews and local dispositions.
No trial was rewritten or rejudged. No further paid trials or GPU work authorized.
Root authorized public developmental archival of this and the failed wrapper attempt;
it is explicitly outside training-data discovery and cannot become a training corpus.
