<!-- ABOUTME: Reuse-first inventory of nonmoral deliberation baselines and related existing ablations. -->
<!-- ABOUTME: Records locally verified provenance, comparison limits, and the smallest useful next experiments. -->

# Baseline inventory — 2026-09-08

**We already have a nonmoral checkpoint, a math-control checkpoint, and a no-synthetic checkpoint.** A fresh corpus is not a prerequisite for testing them. The missing baseline is a convincing control for the *content of nonmoral reasoning*, not the existence of any control model.

Initial scope: local files and cached HF artifacts. Follow-up authorized public read-only HF retrieval; this is now complete, with no inference, rental, upload, or new evaluation. Git base: `42b6acffe4e24845280ae48890d659a4f2e1e296`. Paths below are relative to the repository. Let `I = output/nonmoral_investigation/20260908/` and `R = I/reuse_baselines/` for compact evidence references. The original artifact names are preserved; the nonmoral model/eval now redirect from `LASR-Callum` to `dougalldeepmind` at the same pinned revisions. The Numina adapter still resolves directly under `matboz`.

## Existing checkpoints and exact inputs

| Role | Artifact and revision | What it supplies; limitation |
|---|---|---|
| Historical nonmoral | `LASR-Callum/2026-09-02-qwen36-lora-table2-9284-nonmoral-deliberation-684-rank-64-dynbatch` @ `2225547cec8bd312a1e025f02fb6b3321c047e4a` | Qwen/Qwen3.6-27B, thinking, rank 64, seed 0, one epoch, dynamic batching. One trained checkpoint; no additional nonmoral training seeds. |
| Nonmoral training bytes | `LASR-Callum/2026-09-02-table2-9284-nonmoral-deliberation-684-train-mixture`, `t2_9284_nonmoral_684.jsonl` @ **`6364505df02b0020b030bf379bd42285a14de6a5`** | Verified locally: 9,968 rows = exact 9,284 replay + 684 nonmoral. Synthetic IDs are top-level `scenario_id`. SHA256 `0517ef85d288f14e42bc77f371d3e2e48866879b5824984feaf4a7451ce60561`. Use training revision, not inventory HEAD `acc93908e104359d5f49f2b772da256453385fb3`. |
| Nonmoral source corpus | `LASR-Callum/2026-09-02-craft-tensions-nonmoral-deliberation` @ `fed726d2db33bddb698ca349a6d76a4e7df7a7e9` | 702 complete rows and generation stages; only 18 absent from training. Local corpus SHA256 `be21271e4ed73df064931279d2b52c63d1b5abb1374b5c80ced597bbd86c984d`. |
| Math control | `matboz/qwen3.6-27b-lora-9284-numina-control-716-r64` @ `edfb4287c10f553c541ba28216f202d0c0f47055` | Same model family, rank 64, thinking, seed 0, dynamic batching. Exact 9,284 replay + 716 extra NuminaMath rows, with derivation in CoT and boxed-answer paragraph as final. 716 versus 684 rows, different tasks/style/length: useful category baseline, not an isolated deliberation manipulation. |
| Math input | `matboz/2026-08-19-numina-control-9284-plus-716`, `mixture_think_numina_control.jsonl` @ `c326b44f0c7d0b770516083e82dd76dc02d5190c` | Pinned by adapter stamp; stamp has `git_sha: nogit`. Archived recipe supplies seed/setup, not complete executable historical provenance. |
| Historical no-synthetic | `LASR-Callum/qwen3.6-27b-lora-table2-only-9284-r64` @ `2c513ea7513baf792bd2becf0900b5c9d858c92d` | Exact historical replay family, but ordinary SFT versus nonmoral dynamic batching; thinking stamp backfilled, no dataset revision in stamp, fetched run metadata says `nogit`. Historical anchor, not fully matched causal control. |
| New no-synthetic recipe | `LASR-Callum/2026-09-05-qwen36-0-nosynth` | Exists according to Sept 6 log; no pinned local adapter inventory in this workstream. New 10,000-source blend, not the old 9,284 rows. Do not substitute it silently for historical replay. |
| Moral reference | `LASR-Callum/2026-08-21-qwen36-lora-table2-9284-difficult-advice-chunk-only-702-rank-64-dynbatch` | Current baseline family is principle-scoped 702, per `docs/BASELINES.md`. Exact evaluated target name from cached Sept 4 results; adapter revision was not captured in that cached eval metadata. |

Evidence: `I/inventory.json`, `I/local_audit.json`, each named adapter's `training_meta.json` and `adapter_config.json`, Table-2 `run_meta.json`; `configs/train/archive/qwen36-numina-control-716-dynbatch.yaml`; `configs/data/mixture/nosynth.yaml`. Nonmoral historical recipe is recoverable with `git show 06e98e18df4b542db3619ddf3b579cc1f657a083:configs/train/2026-09-02_lora_qwen36_table2_9284_nonmoral_deliberation_684_dynbatch.yaml`.

## What was actually evaluated

| Result | Verified measurement | Reuse/comparison status |
|---|---|---|
| Nonmoral, Sept 4 | **73/400 = 18.25%**, scenario CI [11.1, 28.5]; 40 scenarios × 2 variants × 5 passes; one checkpoint. Temperature 0.7; 16,384 context; Gemini 3 Flash sole MR judge. Eval repo `LASR-Callum/2026-09-04-odcv-qwen36-0-nonmoral-deliberation-7` @ `1fcdacdc38c2850a60e7eb08532d40019e3f5b1b`. | Best verified nonmoral reference. Two pass-4 transcripts reconstructed from Docker logs; preserved in result provenance. **Submission block still covers only 320 rollouts (100%), not all 400. No progress result.** It does not establish full-run completion or capability preservation. |
| Principle-scoped DA, Sept 4 | **43/400 = 10.75%**, scenario CI [5.7, 19.3]; same 80-cell/five-pass/temperature/context/MR-judge settings. Eval repo `LASR-Callum/2026-09-04-odcv-qwen36-0-da-principle-scoped-7` @ `77f77480585769246aa75cd11aac678b6ef7c062`. | A more relevant moral reference than older circulating DA numbers. Submission 99.5% over400; progress mean4.91/5, ≥3 in98.5%. One reconstructed transcript. Separate generator/domain, so the 7.5pp descriptive gap does not isolate morality. |
| Math-control, Sept 6, newly verified | **98/240 =40.833%**, scenario CI [29.5,53.2]; same80cells,3passes, temperature0.7, Flash sole MR judge. `dougalldeepmind/2026-09-06-odcv-qwen3-6-27b-lora-9284-numina-control-716-r64` @ `b1038e001245007a7b8f1902684b534b3f1d96d9`; exact Numina adapter revision above. | Context **28,000**, versus nonmoral16,384, and changed transcript-budget harness. Submission87.9%; progress mean4.75, ≥3 in95%. A useful existing math reference, not a fully matched measurement. The older Aug30 score URL is not publicly retrievable under either org (HTTP401; private versus missing unknown). |
| New nosynth, Sept 6 | Log reports **36.2% [25.4,48.7]**, 240/240 clean; same0.7/context/Flash configuration,3passes. `LASR-Callum/2026-09-06-odcv-qwen36-0-nosynth`. | Log-level evidence only in this workstream; different replay recipe. Useful orientation, not the historical no-synthetic counterfactual. |
| Original nonmoral, Sept 2 | Handoff reports25.0%, one pass,56balanced cells, Grok4.20+Gemini3.1Pro. | Superseded for headline reporting; different cell set/judge. Do not pool its failed earlier passes or compare25.0→18.25 as a model improvement. |

Exact primary results/configs: `I/2026-09-04-odcv-qwen36-0-{nonmoral-deliberation-7,da-principle-scoped-7}/{results/results.json,metadata/odcv_config.yaml,metadata/run_meta.json}`. Other evidence: `docs/LOG.md` Sept4 nonmoral, Sept6 nosynth, Aug19 control entries; `scratch/plot_odcv_single_pass.py` names math score source. Rates use severity≥3. Repeated stochastic evaluation passes are **not training-seed replication**; pass number alone does not prove distinct explicitly pinned inference seeds. ODCV uncertainty must retain scenario clustering.

## Existing ablations we should not accidentally repeat

| Question | Existing work | What remains unanswered |
|---|---|---|
| Math? | Real-CoT Numina control above; no new generation/training needed to reuse checkpoint. | Nonmoral craft versus mathematics under one measurement protocol; a matched-length/content causal contrast remains separate. |
| Much shorter reasoning? | Moral DA verbose-CoT dataset, row-matched and token-matched LoRAs; Aug25 log reports3×expanded traces. ODCV incentivized-only30cells, temperature0,89/90 and87/90 rollouts,26.1% and31.1%. | These are **longer moral reasoning**, not tasks naturally requiring shorter nonmoral reasoning. Their paired contrast chiefly varies row/token allocation; it does not establish a nonmoral length effect. |
| No deliberation? | Moral CoT-only, answer-only and empty-CoT supervision experiments exist (`output/report/odcv_supervision_series_results.md`). | None implements retained procedural CoT without comparing choices on nonmoral tasks. No new baseline of this kind was trained in the failed pilot. |
| Stakes? | Moral low/high stakes work supplied by user and indexed in log. | Nonmoral stakes variation remains untested/deferred. User's no-effect prediction is prospective, not a result. |

## Smallest reuse-first options

1. **Zero new LoRAs: measurement baseline.** Retrieve and pin the existing math score artifact; compare existing nonmoral/math/DA on their shared cells only after verifying judge, temperature, context and harness differences. Existing transcript reanalysis can establish descriptive context now; a common-protocol rerun of existing checkpoints later removes measurement differences without regeneration. Repair nonmoral submission accounting before treating task completion as reassuring. Progress backfill would add judging cost and is not performed here.
2. **One new LoRA: exploratory rewrite control on historical684.** Preserve all historical prompts, final answers, replay bytes, row order and recipe; replace only synthetic CoT with execution/verification reasoning. Reuse the original checkpoint as reference. This avoids fresh-scenario failure costs, but historical CoT versus newly written CoT still bundles comparison with writer/style/length. It cannot claim pure isolation of deliberation. Audit first: if answers need repairs or rows must be removed, the old checkpoint ceases to be the matched dataset reference.
3. **Two new LoRAs: stronger paired-content test using reusable scenarios.** Reuse valid historical prompts/finals where possible, author both CoTs under the same current procedure, and use the exact same accepted IDs/replay/order/recipe in both. Costs more training, reduces writer asymmetry, and permits consistent exclusion/repair. It still tests comparative versus procedural **CoT**, not absence of reasoning or all deliberation in the full answer.

Recommendation: complete option1's cheap provenance work while the other workstreams decide whether option2's exploratory claim is worthwhile or option3 is needed. Broad-domain expansion can proceed independently; it should not block these baselines. No data selection should depend on ODCV scores. Respect the user's one-LoRA-per-condition limit and label all current conclusions checkpoint-specific; the repo's general three-training-seed ranking rule does not authorize extra seeds.

Initial evidence gaps were math score revision/protocol, full nonmoral submission/progress, absent independent nonmoral training seeds, historical base-model revision/trainer provenance, and whether the original 684 examples are valid enough to preserve their answers. Their current status is below.

## Follow-up: recovered baseline accounting and math provenance

The following resolves parts of the preceding inventory; it adds no trained model or new judgement.

**Nonmoral submission is398/400 =99.5%, not the stale100%/320.** Downloaded all400 published transcripts at the pinned evaluation revision and applied the exact historical `submission_stats` rule: substring `"action": "task_complete"`, averaged within each cell then across cells. Verified that rule in Git `d83b0470c22b7c2355df80d82dfde1a9c35d5230`. Each of80cells has5transcripts. Passes1,2,3,5 submit80/80; pass4 submits78/80. The two marker-absent transcripts are the two documented reconstructions. This is absence in the recovered record, not proof that an unrecorded action could never have occurred. The published `metadata/combine_manifest.json` still lists only the four initially kept passes, explaining its320-row submission block; MR already incorporates all400. No published artifact was changed.

**Submission is not success.** It records the agent invoking the submit tool, not whether the task was completed correctly. Nonmoral still lacks graded progress results; no progress labels or capability claims were inferred. Its incomplete historical accounting is now locally repaired, not externally published.

**The math baseline is reusable, but its evaluation needs matching.** Its Sept6 run has240judged transcripts,3complete80-cell passes,0recorded retries/reconstructions, and the same base-model revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` as the nonmoral evaluation. Exact scenario/variant sets match. Besides depth3 versus5, the meaningful protocol difference is28,000 versus16,384context plus the Sept5 transcript-budget patch. The patch changes how retained tool history is budgeted; it is not just a metadata rename. Source revisions are `d83b0470c22b7c2355df80d82dfde1a9c35d5230` (nonmoral) and `3477b1debcf04603a8df469a11ca4dfecf720639` (math rollout); math judging metadata records `72885077c8f0a972ed1f913c3d3c2399ef5315b4`. Do not quote22.58pp as an isolated nonmoral-versus-math effect. The published math submission/progress figures above were inspected, not independently rejudged.

**Reproduce:** `uv run --no-sync python -m scratch.nonmoral.reuse_baselines`. Public HTTP only, no credentials or model clients. The first run fetched 11.45 MB of artifacts; reruns reuse pinned local files. Directories under `R` preserve `nonmoral/`, `math/`, both adapters, and two failed-path records. `R/files.json` hashes every fetched artifact; `R/summary.json` records exact counts, configurations and identities; `R/nonmoral_submission_rows.json` records all 400 transcript decisions. `R/historical_passes.py.txt` and `R/nonmoral_to_math_harness.diff` preserve the source evidence. These ODCV transcripts are measurement artifacts only; they must not become generator examples or selection features.

Remaining bounded gaps: (1) common-protocol evaluation of existing checkpoints later, if the question requires a reliable between-arm estimate; (2) nonmoral progress grading later, with explicit paid-judge scope; (3) historical training provenance and whether the original684answers withstand independent review. The missing older Aug30 score repo no longer blocks having a pinned math reference. Additional training seeds remain explicitly deferred.

## Launch proposal: three existing checkpoints, one measurement protocol

**Execution update, 2026-09-09:** the user authorized this comparison within the $300
overall ceiling. The active baseline allocation is $60 GPU/storage plus $10 judging;
durable judge reservations, RunPod watchdogs and Windows path/LF checks are implemented.
The first CRLF-contaminated attempt was stopped without judging; a corrected run is
active. [Current experiment card](experiment_card.md) supersedes the historical
approval and implementation gaps in the proposal below. Checkpoint order/settings
remain fixed; no extra passes will be selected from observed scores.

Proposal following the [postmortem reset](2026-09-08_postmortem.md), **not a rental or evaluation authorization**. Question: does the historical nonmoral checkpoint retain lower misalignment than existing math and replay-only checkpoints when evaluated identically, without a corresponding loss of task submission/progress? This establishes a better baseline comparison; it cannot isolate deliberation from every training-data difference or establish capability preservation.

| Fixed order | Existing checkpoint (current resolved ID) | Required adapter revision | New passes / rollouts |
|---|---|---|---:|
| 1: nonmoral | `dougalldeepmind/2026-09-02-qwen36-lora-table2-9284-nonmoral-deliberation-684-rank-64-dynbatch` | `2225547cec8bd312a1e025f02fb6b3321c047e4a` | 3 / 240 |
| 2: math | `matboz/qwen3.6-27b-lora-9284-numina-control-716-r64` | `edfb4287c10f553c541ba28216f202d0c0f47055` | 3 / 240 |
| 3: Table-2-only | `dougalldeepmind/2026-08-04-qwen36-lora-table2-only-9284-rank-64` | `2c513ea7513baf792bd2becf0900b5c9d858c92d` | 3 / 240 |

Total **720 new rollouts, zero training runs**. Three stochastic passes per checkpoint give within-cell repeat information at lower cost than five; pass count is fixed before new scores. These are not three training seeds. All use base `Qwen/Qwen3.6-27B` revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, thinking mode, bf16 serving and the same pinned template/tool parser. Public metadata verified the Table-2 redirect and base revision during this proposal; historical names in the inventory remain provenance.

**Shared settings:** reuse `scratch/nonmoral/odcv-paired.yaml` unchanged with the recorded override `passes=3`: all 40 scenarios × mandated/incentivized variants, temperature 0.7, 28,000-token context/transcript budget, concurrency 8, 2,400-second scenario timeout, prefix caching enabled, and four judge workers. One `google/gemini-3-flash-preview` judge for MR and the same model for progress, using the existing rubric/provider pin. This reuses the already explicit evaluation choice; Sonnet remains the dataset generator, and no new judge-calibration project is proposed. Freeze the actual harness/config/template revisions and resolved provider settings before the first run and retain them for all three.

**Execution shape:** local Docker driver, one owned H100 80GB inference pod, three **single-target** `uv run evals` calls, with existing watchdog/teardown. The existing entrypoint restarts its server between calls, adding startup overhead. Do not put these different recipes into one multi-target invocation: current `run_eval.py` automatically pools ODCV targets as checkpoint replicates. That pooled quantity would answer the wrong question. No new runner is needed.

Command template, used once for each target in the table after authorization:

```powershell
uv run evals --name odcv --config scratch/nonmoral/odcv-paired.yaml --target <TARGET> --server <OWNED_HOST> passes=3
```

**Revision caveat:** this CLI has no arbitrary target-revision argument. `resolve_target` resolves HEAD and then pins the returned SHA for loading. Preflight must verify each returned adapter/base SHA equals the table; abort on drift rather than invent an inert `target_revision=` override. The older adapters lack training-time base revision stamps; a common verified serving revision closes the measurement mismatch, not that historical provenance gap. Windows output paths and Docker networks need their actual launch preflight; the runner currently injects its own output directory, so an `output_root=` override is not a verified remedy.

**Cost proposal: $80 total ceiling**, provisionally $60 serving, $15 judging and $5 reserve. These are proposed allocations, not measured costs or a guarantee of finishing 720 rollouts. Exact historical math judging was $1.6785 MR + $0.7682 progress for 240 transcripts; tripling that volume gives **$7.3401 as an extrapolation**, assuming similar token lengths/prices and no extra calls. New transcripts, retries and provider drift can change it. No independent rejudging is included. Serving has no defensible current point estimate: at an illustrative $3.50/hour, $60 permits 17.1 billed hours including startup/idle time; obtain the actual rental quote before approval. Historical concurrency-32 timing does not validate throughput at concurrency 8. Set the owned-pod lifetime cap from the accepted quote/allocation, track judging separately, and preserve an incomplete result if the budget cannot finish the frozen scope. Existing ODCV code does not itself enforce a dollar ceiling on judge calls; the launch owner must establish that control/monitoring before claiming a hard all-in cap. Do not buy additional passes based on observed MR.

**Deliverable and validity:** one compact three-row results table and paired-difference figure: MR numerator/denominator and scenario-level CI; submission counts over all 240 records; progress mean and ≥3 count; contextual truncations, errors, reconstructions and missing cells. Compare nonmoral against each control using the existing scenario-paired statistics, preserving rollout rates rather than treating 720 draws as independent scenarios. Keep recipes separate and report missing-cell overlap explicitly. A clean run completes all 80 cells × 3 passes per checkpoint; incomplete coverage is reported as incomplete, not silently repaired by changing the comparison set. Submission alone is not task success; progress alone is not a capability battery. Any observed worsening in completion/progress blocks an unqualified alignment-improvement claim, and formal capability testing remains deferred.

Remaining launch decisions are bounded: approve the $80 scope after a current rental/provider quote; verify revision equality and the common serving environment; confirm paid-judge spend control and public write access under `dougalldeepmind`. Table-2's backfilled/ordinary-SFT provenance and the 716-math versus 684-craft mixture difference remain limitations even after a perfectly matched evaluation. This measurement lane is independent of whether the new paired-data pilot succeeds.

Analysis implementation note: `src/eval/misalignment/odcv/stats.py:arm_difference` accepts per-cell lists through `_long`; use those repeated-rollout values. The older `odcv_compare.py` command also contains scalar-only McNemar comparisons and is not a verified end-to-end report command for these list-valued results. This does not require replacing the evaluation runner.
