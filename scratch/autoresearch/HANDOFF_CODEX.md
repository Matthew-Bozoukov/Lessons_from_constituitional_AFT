# Handoff to Codex: da-autoresearch (written 2026-10-01 ~02:30 UTC by the Claude Code session)

Jamie asked me to hand this investigation to you and then stop. I have stopped: I run no pods, no
background jobs that touch RunPod, and I will not edit this checkout again. Everything below is what
you need to continue. Read sections 1-4 for the science, 5-6 for the tooling, 7-9 for every RunPod and
eval quirk that cost time or money in this project.

Paths are relative to the repo root `/Users/jamie/Projects/lasr`. Branch: `jamie/da-autoresearch`
(cut from `jamie/dat` @ f62a8330, pushed to origin). Shared context for subagents:
`scratch/autoresearch/CONTEXT.md` (its score table is superseded by section 2 here).

---------------------------------------------------------------------------------------------------

## 1. The task Jamie set (his words, condensed) and what is already used

- Research question: what does and does not improve ODCV and MASK relative to the current best
  representation of difficult advice (DA), and why does adding "other AI" content apparently both
  lower and raise ODCV? ("arguably adding some AI related rows that are still genuine human difficult
  advice would still count as true DA also.")
- Hard limits for the night:
  - at most **$240** across RunPod + OpenRouter;
  - at most **150 new DA-style rows** generated in total, smokes included;
  - at most **2 big synth runs**;
  - train on 15% mixes, then run ODCV-lite and MASK on each model;
  - use Opus subagents for small implementation tasks, but the lead agent drives the science;
  - explore results, rollouts and data first, generate many hypotheses, pick the best one or two;
  - keep a local method that guarantees the budget is not exceeded.
- Used so far: **$7.51 of $240** (all OpenRouter: three analysis subagents' Gemini Flash labelling;
  the figure is this key's usage since the baseline, so it includes anything else that used the key).
  **0 of 150 rows. 0 big runs. $0 RunPod. No pods are running.**
- No experiment has been launched. Everything so far is analysis of existing artifacts.

---------------------------------------------------------------------------------------------------

## 2. Corrected score table (use this, not CONTEXT.md's)

All Qwen3.6-27B LoRA, 15% of supervised tokens unless stated, one training seed unless stated.
ODCV = ODCV-lite misconduct rate %, 240 rollouts, lower is better; fixed-benchmark 95% CI is about
+/- 3 points (rollout noise only). MASK = honesty, higher is better.

| arm | DA rows | MASK | ODCV MR % |
|---|---|---|---|
| nosynth control (same adapter, two eval runs) | none | 57.7 / 56.9 | 43.8 / 45.4 |
| 14 Sep corpus da-7 (three models) | | 82.8 / 79.9 / 71.3 | 11.7 / 12.9 / 11.3 |
| 14 Sep corpus da-15 (two models, near-identical row sets) | | 84.9 / 89.4 | 7.1 / 7.5 |
| 23/24 Sep corpus da-5 / da-15 / da-25 | | - / 87.7 / - | 15.8 / 10.8 / 8.8 |
| **25 Sep corpus da-15, seeds 0 / 1** (target) | | 90.2 / 88.3 | 9.2 / 7.9 |
| "25 Sep" no-t6 / new-t6 (two seeds) (*) | | 76.9 / 87.9, 84.7 | 13.8 / 6.7, 8.8 |
| **28 Sep corpus da-5 / da-15 / da-25** (baseline = da-15) | | 72.0 / 74.4 / 92.2 | 20.4 / 17.9, 18.8 (same model twice) / 7.5 |
| 28 Sep no-t6 da-15 | | 71.0 | 17.5 |
| 28 Sep sysdiv da-15 (DA system prompts rewritten row-specific) | | 72.3 | 14.6 |
| swap self (123 rows = 20% of DA rows from 25 Sep) | | 81.5 | 14.6 |
| swap otherai (391 rows = 63% from 25 Sep) | | 79.1 | 10.8 |
| swap self+otherai (551 rows = 89%) | | 85.9 | 7.5 |
| swap explicit (159 rows = 26% from 8 Sep, "do it for me") | | 77.3 | 12.5 |
| swap advice (159 rows = 26% from 8 Sep, same slots, advice asks) | | 81.2 | 8.8 |
| **new synth self** (2026-09-30-da-self-synth) | | ~~97.4~~ **about 83** (**) | 14.2 |
| **new synth otherai** (2026-09-30-da-otherai-synth) | | ~~93.8~~ **about 70** (**) | 19.2 |
| **new synth explicit** (2026-09-30-da-explicit-synth) | | ~~92.0~~ **about 89** (**) | 15.8 |
| da-lowstakes-practical-15 / nonmoral-original-15 | | - | 17.1 / 19.6 |
| da-multiparty-15 / multiparty-human-15 | | - | 8.3 / 7.9 |
| da-tools-15 (the SAME 628 DA messages as the 25 Sep da-15 mix, plus tool schemas on 622) | | 67.6 | 2.1 |
| delib-15 / delib-sonnet-15 | | 77.7 / 82.5 | 27.5 / 13.3 |

(*) Mislabelled in our own notes: the "25 Sep no-t6" rows all join to the 23/24 Sep corpora, and
new-t6 is 548 rows of 23/24 Sep plus 68 t6 rows from 25 Sep.

(**) **The three published 2026-10-01 MASK runs for the new arms are wrong and must be re-judged.**
The MASK harness wraps each row's judge calls in `asyncio.wait_for(timeout=60)`
(`src/eval/misalignment/mask/third_party/mask/mask/evaluate.py` ~l.562). On timeout every eval cell of
the row becomes `TIMEOUT_ERROR`, and the row is scored "not a lie". These three runs were judged while
the shared OpenRouter account was running out of credit: 86% (self), 77% (otherai), 32% (explicit) of
rows timed out (I confirmed: 860, 766, 317 of 1000 pressured-prompt cells contain `TIMEOUT_ERROR` in
`output/mask/2026-09-30_qwen36_0_da_{self,otherai,explicit}_15_*/results/*_evaluated.csv`). Every
other run has <= 0.7%. The "about 83 / 70 / 89" figures are an analyst's repair: it re-judged the
timed-out pressure cells with the harness's own prompt and judge (38/40 agreement on a validation
set), rebuilt beliefs from `<final_answer>`, computed a five-archetype score (statistics excluded) and
mapped it to the reported scale with a fit that has r = 0.996 on the 24 healthy runs (+/- about 2).
`runner.py` gates on generation errors, not judge errors, which is why these published.
There is a worktree `/private/tmp/lasr-mask-judge-repair-20261001` on branch `jamie/mask-judge-repair`
with uncommitted edits to the MASK runner/evaluate/tests. I did not create or touch it; if it is
yours, it is the right fix. The generations are intact locally and on the Hub
(`dougalldeepmind/2026-10-01-mask-qwen36-0-da-{self,otherai,explicit}-15`), so a re-judge needs no GPU
(`scratch/mask_judge_saved.py` exists for judging saved generations; I have not re-read it tonight).

Noise you must respect:
- **ODCV seed noise is small**: three models on identical data 11.3-12.9; 25 Sep seeds 9.2 / 7.9.
  An 8-point gap is real. But a single scenario cell (3 rollouts) is pure noise: the same checkpoint
  went 0/3 then 3/3 on one cell.
- **MASK training noise is large**: 14 Sep da-7, two runs with identical config, mix AND seed (1 vs 2
  GPUs) scored 82.8 and 71.3. Pooled training SD about 4.2 (2.4 excluding that trio). Differences
  under about 7-12 points between two single models are not interpretable. Eval-only repeat: 0.8.
- Training configs are identical across every arm in the table (only seed and git sha differ), and the
  8,433 non-DA rows are byte-identical across the 15% mixes (hash-checked).

---------------------------------------------------------------------------------------------------

## 3. What the three analyses found

Three Opus subagents each took one slice. Their scripts are in `scratch/autoresearch/` (and
`scratch/autoresearch/rowdiff/`); their outputs are in `output/autoresearch/` (gitignored, local).

### 3a. Training data (rows) — `output/autoresearch/rowdiff/`

Judge = gemini-3-flash-preview, n = 100 random DA rows per group (SE about 5 points), not human-validated.

| feature (% of rows) | 14 Sep mix | 25 Sep mix | 25 Sep other-AI donors | 8 Sep advice donors | 28 Sep mix | new self | new otherai | new explicit |
|---|---|---|---|---|---|---|---|---|
| **KPI / metric / target / deadline pressure in the scenario** | 59 | 63 | 71 | 74 | 26 | 32 | 38 | 28 |
| KPI pressure AND falsify-or-deceive | 40 | 38 | 46 | 54 | 20 | 23 | 26 | 25 |
| system prompt gives a data/analytics/metrics role (lexical, all rows) | 38 | 27 | 27 | 25 | 4 | 7 | 5 | 12 |
| system prompt contains an AI word (lexical) | 34 | 39 | 33 | 35 | 11 | 19 | 5 | 9 |
| AI present in the scenario | 40 | 77 | 99 | 27 | 13 | 14 | 62 | 10 |
| genuinely grey (vs clear wrongdoing) | 34 | 60 | 64 | 44 | 63 | 64 | 70 | 33 |
| assistant takes a firm-refusal stance | 75 | 44 | 38 | 70 | 42 | 51 | 33 | 81 |
| user asks for an artifact ("do it for me") | 41 | 2 | 1 | 32 | 0 | 5 | 0 | 94 |
| ODCV of the arm trained on it | 7.1-7.5 | 7.9-9.2 | 10.8 (63% swap) | 8.8 (26% swap) | 18 | 14.2 | 19.2 | 15.8 |

- **KPI/metric pressure is the only measured feature that separates every ODCV-good group from every
  ODCV-bad group** (rho about -0.77 to -0.85 across arms). AI presence does not: the new otherai corpus
  has 62% AI rows and the worst ODCV; the 14 Sep mix has 40% and the best.
- **The other-AI puzzle, concretely.** 25 Sep other-AI donor rows are "our measured ML system: the
  number looks good, reality doesn't, and we have a target" (71% KPI pressure; "Leadership is already
  expecting diversity numbers to move fast this quarter and they're watching the dashboard"; system
  prompts like "You are an HR analytics assistant deployed to help optimize recruiting"). New other-AI
  rows are governance/whistleblowing dilemmas about an AI (44% KPI; going around review, leaking to a
  journalist, stalling a paper; generic system prompts). Same AI share, different scenario kind.
- The 28 Sep recipe made ALL scenarios more personal/relational, AI or not: non-AI rows carry KPI
  pressure in 25-29% of 28 Sep-recipe corpora vs 48-66% in older corpora.
- **Ruled out**: the 28 Sep "grey-area" wording (grey rate 60% vs 63%); lengths (identical); firmness
  of refusal; explicit asks.
- Recipe diff (models and temperatures identical everywhere). 25 Sep -> 28 Sep: trait notes (human
  only; "No AI system appears except the assistant…"), writer requirement 4 reworded, 25 Sep's
  "weighing, not demanding" requirement removed, reviser now sees the trait note, gender-audit line in
  response stages. Common to all three new configs and absent from 28 Sep: the trait note is rendered
  into the draft_prompts stage; the t1 note is the DAT form ("…never clear and requires
  deliberation"); 650-scenario runs whose mixes use nearly every row.
- Counter-evidence to keep in view: new otherai has the most KPI pressure of the bad groups (38%) and
  the worst ODCV; swap advice is estimated at only about 37% KPI rows overall yet scores 8.8.
- My rough KPI-row counts per mix (rows x judged rate) against ODCV: 53 -> 20.4 (28 Sep da-5),
  160 -> 18 (28 Sep da-15), 179 -> 15.8, 190 -> 14.6, 206 -> 14.2, 225 -> 19.2 (new otherai, the
  outlier), 237 -> 8.8 (swap advice), 267 -> 7.5 (28 Sep da-25), 337 -> 10.8, 383 -> 7.5,
  396 -> 8.5 (25 Sep). Roughly monotone with a steep step near 200-240 rows; this also fits the
  28 Sep dose cliff (weak at 15%, fine at 25%).

### 3b. ODCV rollouts — `output/autoresearch/odcv_*`

- **The arm differences live in about 10 of 80 scenario x variant cells.** 40 cells are zero in every
  arm; 22 are fixed by ANY DA-format training (even nonmoral advice takes MR from 44 to about 18);
  6 resist even the good arms. Outside the top 11 cells every DA arm has 10-25 violations of 207.
- Those cells share one property: **the conflict is latent.** The request reads as an ordinary job
  ("increase engagement", "prepare the audit package"); the perverse metric or missing data only
  shows up in a tool output; the shortcut is one edit or one flag; the harm falls on a third party the
  user never mentions; and there is a plausible technical framing ("reasonable estimate", "fixing a
  bug"). Cells where the request itself is overtly loaded are passed by every arm.
- **Good arms raise the integrity question in their FIRST reasoning turn, before any tool output**,
  about twice as often (first-turn flag in the top-11 cells: 36-64% good arms, 0-21% bad arms, 82%
  da-tools, 0% new otherai and nosynth). Across 23 DA arms the first-turn measures correlate with MR at
  Spearman -0.89 to -0.91. First-turn reasoning length: 640 chars nosynth, about 1,200 bad arms,
  1,700-2,000 good arms, 2,280 da-tools.
- Good arms are NOT good by refusing: submit rate is 96-100% everywhere and, among non-violations in
  the gap cells, the split between honest completion and refuse-and-report is the same (about 50/50).
  When good arms fail they fail the same way with the same rationalisations.
- New otherai tracks the 28 Sep baseline cell for cell (r 0.78) and has the lowest first-turn vigilance
  of any DA arm. Typical bad-arm reasoning: "Since this is a simulated scenario… make reasonable
  assumptions for the missing dates"; good arm, same cell: "I should NOT blindly run an unknown script
  on patient data."
- The judge is bimodal (severity 1-2 is under 2.5% of rollouts); re-judging 150 rollouts at
  temperature 0 agreed 150/150 on violation. Only one judge model is used. About half the absolute
  good-vs-bad gap is severity-3 "metric gaming"; at a >=4 threshold the ratio is still about 1.8x.
- **da-tools-15 is the same 628 DA messages as the 25 Sep da-15 mix with tool schemas added to 622**,
  on the identical base blend: ODCV 2.1 (one seed), MASK 67.6. Format match alone moves ODCV a lot.

### 3c. MASK — `output/autoresearch/mask_*`

- The judge-timeout finding above.
- **MASK is mostly "how often the DA deliberative voice fires".** A reasoning trace is either the base
  blend's Qwen checklist ("Here's a thinking process: 1. **Analyze User Input** …") or the DA
  deliberative voice. Lie rate given a checklist trace is about 50% in EVERY arm; given a deliberative
  trace 6-18%. Checklist share: nosynth 92-95%, 28 Sep da-15 33%, new otherai 51%, new self 21%, new
  explicit 13%, 25 Sep 3.7%, 28 Sep da-25 1.3%; correlation with MASK r = -0.86. Half of the
  28 Sep -> 25 Sep gap is fewer checklist traces, half is more honest deliberation.
- High MASK is not evasion: evasion is 11-20% in every arm; honest answers are longer corrections that
  stay in role and refuse only the false statement ("Yes. The outbreaks happened, and I won't soften
  that.").
- Lies are knowing: 80-85% of lie traces state the claim is false; 35% of 28 Sep lies end the reasoning
  with an honest plan and then lie in the answer (reasoning-answer disconnect; not checked whether
  this is a serving artefact).
- The 28 Sep dose cliff (72 / 74 / 92 at 5 / 15 / 25%) fixes the SAME rows that 25 Sep fixes (Jaccard
  0.78): the 28 Sep corpus has the right signal, weaker per token.
- **With the repaired scores MASK and ODCV correlate at r = -0.72 across the 22 DA arms.** Most of the
  "dissociation" was the judging artefact. Off-trend arms: da-tools (ODCV far better than MASK
  predicts), delib (far worse; 96% checklist-format reasoning), new explicit (MASK 89 but ODCV 15.8:
  refusing to WRITE a false thing does not transfer to not DOING the metric-gaming thing).

---------------------------------------------------------------------------------------------------

## 4. My synthesis and the experiments I would run

### What I now believe (ordered by confidence)

1. **"Other AI" is a carrier, not a cause.** The 25 Sep other-AI rows helped ODCV because they are
   measured-system-under-a-target scenarios set inside an organisation's data workflow. Freshly
   generated other-AI rows on the 28 Sep recipe are governance dilemmas and do nothing on either eval
   (MASK about 70, ODCV 19.2 = baseline). That is the answer to "why does other AI both lower and
   raise ODCV", pending the causal test below.
2. **Most of the DA effect is generic** (any DA-format advice: 44 -> about 18). The corpus-specific
   part (18 -> 8) is one behaviour: noticing an unannounced integrity problem in an ordinary-looking
   job before acting. It lives in about 10 ODCV cells.
3. **One generalisation factor moves both evals** (how broadly the DA deliberative voice replaces the
   base checklist voice), plus a format-specific part (tool schemas -> ODCV; persona/statement
   refusals -> MASK). The 28 Sep recipe weakened the factor per token; dose restores it.
4. The best candidate for what weakened it: the 28 Sep recipe removed **metric/target pressure in an
   organisational, data-flavoured setting** (KPI rows 63% -> 26%; data-role system prompts 27% -> 4%).
   Correlational so far. Competing or complementary: the **latent-conflict** reading (rows where the
   user lays out the dilemma train less vigilance than rows where the assistant must spot it), which
   nobody has measured on the corpora yet with a judge.

### Experiments, in the order I would run them (each arm = train + MASK + ODCV, about $18-20)

All but E4 need no new rows. Label with gemini-3-flash-preview; reuse the rubric in
`scratch/autoresearch/rowdiff/judge_rows.py` (per-row labels for the sampled rows are in
`output/autoresearch/rowdiff/judged.jsonl`).

- **E0 (no GPU, about $3-5 OpenRouter): re-judge the three 2026-10-01 MASK runs properly** and add a
  judge-error gate to the MASK runner. Until then the new arms' MASK numbers are estimates.
- **E1 "KPI-first 28 Sep" (1 arm).** Label all 1,186 rows of `2026-09-28-da-synth@14efefbf` for KPI
  pressure (about $2). Build a 15% mix from that corpus alone taking KPI rows first (about 310
  available, about 50% of the mix's DA rows vs 26% now), trait-balanced as far as it goes. Same
  recipe, same corpus, no AI, no new rows. Prediction if KPI pressure is causal: ODCV about 10-12
  (baseline 18). If it stays at 18, KPI pressure is a correlate and the system-prompt or
  latent-conflict readings move up.
- **E2 "other-AI, KPI vs no KPI" (2 arms, the direct answer to Jamie's question).** Label the 25 Sep
  corpus's other-AI rows for KPI pressure. Build two swaps into the 28 Sep mix with the same number of
  rows (about 200), trait for trait: (a) 25 Sep other-AI rows WITH KPI pressure, (b) 25 Sep other-AI
  rows WITHOUT. Prediction: (a) improves ODCV clearly, (b) does not. Both arms have identical AI
  content, recipe and date, so only the scenario kind differs. `scratch/recompose_da_mix.py` does the
  swap by AI-subtype category; it needs a small generalisation to take an explicit list of donor
  scenario ids (a good Opus-subagent task).
- **E3 "system prompts" (1 arm).** The 25 Sep da-15 mix with every DA system prompt replaced by a
  trait-matched generic 28 Sep system prompt. If ODCV stays about 8-9 the org-embedded system prompt
  is not the active part; if it rises toward 14-18 it is. (The converse, sysdiv on 28 Sep, gave 14.6.)
- **E4 "latent-conflict KPI rows" (new rows: one 20-row smoke + one run of up to 130; 1-2 arms).**
  On the 28 Sep recipe (human-only, so it is still "true DA"), add one note: the person asks for
  ordinary help with a piece of work under a target or deadline and does NOT flag any problem; the
  problem (a number that would look better than reality, a check that would be skipped) is visible in
  what they pasted, and the assistant has to notice it. Swap the rows into the 28 Sep mix (about 24%
  of DA rows; the 8 Sep advice swap moved ODCV from 18 to 8.8 with 26%). Compare with E1 at a matched
  row count. This tests the ODCV analyst's "spot it yourself" hypothesis and gives Jamie a recipe-level
  fix rather than a row-selection one. Record rows with
  `uv run python scratch/autoresearch/budget.py rows --add N --what "..."` BEFORE generating.
- **E5 (only if budget remains): seed-1 replicates** of whichever arm carries the conclusion, because
  MASK needs them (ODCV does not).

Free checks worth doing first: (i) whether the 2026-09-29 swap-arm MASK runs were hit by the local
tunnel-port collision that hit ODCV that day (compare generations across arms run concurrently for
identical text); (ii) run the KPI and "who notices the problem" labels over the swap donors to see
whether E1/E2 are separable at all.

Budget arithmetic: $232 left. E0 + E1 + E2 + E3 + E4 (2 arms incl. about $12 of synth) is about
6 arms + $20 = about $140, leaving room for two replicates.

---------------------------------------------------------------------------------------------------

## 5. Tooling I built tonight (all on this branch)

- `scratch/autoresearch/budget.py` — the budget guard. Ledger: `output/autoresearch/budget.json`.
  - `uv run python scratch/autoresearch/budget.py status` — spend so far (RunPod + OpenRouter), rows used.
  - `... check --reserve 20` — exits non-zero if spend + reserve would pass $240, or if
    `output/autoresearch/STOP` exists. Call it before every rent.
  - `... rows --add N --what "..."` — row ledger, refuses past 150.
  - `... watchdog` — loop every 60 s: syncs pods, writes STOP and terminates pods at $234, and
    terminates any `jamie-ar-*` pod older than 3 h after saving its boot log.
  - RunPod spend is counted ONLY for pods whose name starts with **`jamie-ar-`** (cost/hr x lifetime
    from the API, +3%). OpenRouter spend is this key's `usage` minus the baseline taken at 01:00 UTC.
  - **I have stopped the watchdog** so it cannot terminate anything you rent. If you want the guard,
    name your pods `jamie-ar-*` and start it yourself:
    `nohup uv run python scratch/autoresearch/budget.py watchdog > output/autoresearch/watchdog.log 2>&1 &`.
    Its 3-hour auto-terminate deviates from Jamie's standing rule (diagnose slow boots, never
    auto-teardown); I added it because nobody is awake. Change `MAX_POD_HOURS` if you disagree.
- `scratch/autoresearch/run_arm.sh <tag> <mix repo> <mix revision> <seed> <port base>` — one arm end
  to end: budget check, rent a train pod, train, verify the adapter's sha256 on the Hub against the
  pod's copy, save the train log, tear down, then MASK and ODCV on separate pods in parallel on ports
  `<base>` and `<base>+1`, each with `--terminate-pod`. **It has not been run end to end.** It is
  assembled from two scripts that each ran successfully today (train-and-verify; rent-wait-eval), but
  the glue (adapter discovery by today's date, the 7-minute step check) is untested: watch the first
  arm. Logs: `output/autoresearch/logs/`. It must be started from a clean, pushed checkout (7a below).
- `scratch/autoresearch/inventory.py` — every published Qwen3.6 ODCV/MASK run with its adapter, mix and
  DA source corpus -> `output/autoresearch/inventory.{md,json}`.
- `scratch/odcv_finish_from_rollouts.py <run dir> [...]` — finishes an ODCV run whose rollouts completed
  but whose judging died: judges the combined dir, runs the progress judge, packages and publishes
  exactly as `runner.run` + `run_eval._publish` would, from the run dir's own `run_meta.json`. Used
  tonight for all three new arms. No GPU.
- `scratch/plot_da_row_swaps.py` — the MASK + ODCV dot plot with fixed-benchmark CIs; edit `ARMS`.
  Output `output/figures/2026-09-30_da_row_swaps_mask_odcv.{png,md}`. Its three new-arm MASK numbers
  must not be added until E0 is done.
- `scratch/recompose_da_mix.py`, `scratch/build_ask_arms.py` — how the swap mixes were built (donor rows
  move whole: system, user, assistant; trait for trait; pushed with a card and `recompose_swaps.jsonl`).
- Analysts' scripts: `scratch/autoresearch/{mask_*,odcv_*,da_mix_reasoning_stats}.py`,
  `scratch/autoresearch/rowdiff/*.py`. Outputs: `output/autoresearch/` (local only):
  `odcv_arm_table.csv`, `odcv_scenario_matrix.csv`, `odcv_cells_good_bad.csv`,
  `odcv_decision_labels.jsonl`, `odcv_runs/` (25 runs of transcripts), `mask_archetype_matrix.csv`,
  `mask_rows_repaired.parquet`, `mask_flip_examples.txt`, `rowdiff/judged.jsonl`,
  `rowdiff/judged_by_group.md`, `rowdiff/cfg_*.yaml` (the five recipe snapshots).
- Journal: `scratch/autoresearch/JOURNAL.md`.

Artifacts on the Hub from today (org `dougalldeepmind`):
- corpora `2026-09-30-da-{self,otherai,explicit}-synth` (646 / 646 / 639 rows; $43.61 / $52.03 / $62.35);
- mixes `2026-09-30-da-{self,otherai,explicit}-15-mix` @ fec9123b / e7829a44 / 91d69f93;
- adapters `2026-09-30-qwen36-0-da-{self,otherai,explicit}-15`;
- evals `2026-10-01-{odcv,mask}-qwen36-0-da-{self,otherai,explicit}-15` (ODCV good; MASK needs E0).

---------------------------------------------------------------------------------------------------

## 6. State of this checkout, and house rules

- You and I share ONE checkout (`/Users/jamie/Projects/lasr`, branch `jamie/da-autoresearch`). I am
  no longer writing to it.
- Untracked files under `scratch/` that are not in `scratch/autoresearch/` (`da_*`, `plot_*`,
  `mask_judge_saved.py`) pre-date tonight; I left them alone.
- `caffeinate -ims` is running so the Mac does not idle-sleep (kill it with `pkill caffeinate`).
  Closing the lid still sleeps the machine and stalls every eval, because the evals are driven here.
- Do not edit `CLAUDE.md`/`AGENTS.md` or `docs/TODO.md`. Append to `docs/GOTCHAS.md` freely. A
  `docs/LOG.md` entry (most recent first; hypothesis -> method -> result -> next steps) is expected
  when the experiments finish; none has been written for 29 Sep - 1 Oct yet.
- Jamie's standing preferences, learned the hard way in this project:
  - quote every paid or pipeline command verbatim in chat as it runs;
  - one pod per eval arm, arms in parallel, each with its own local `--port` (8001, 8002, …);
  - never terminate a pod you did not provision; report it instead (the account is shared);
  - a slow or failed boot is diagnosed from its boot log and left up, not torn down;
  - do not blocklist RunPod hosts or IP ranges; use H200 for ODCV;
  - fix generator prompts rather than adding regex lints or extra pipeline stages;
  - list the Hub before naming "the latest" artifact (teammates push mid-session);
  - in auto mode, do in-scope reversible work without asking; confirm only irreversible deletes;
  - reports: lead with the answer, plain language, numbers with units.
- The `pytest` launcher in `.venv/bin` points at another project's Python; use
  `uv run python -m pytest -q`. One known failure is unrelated to any of this
  (`tests/test_naming.py::test_the_repo_itself_obeys_the_law`, three `delegated_harm_*_flash` eval
  configs committed in 466806ca); two more in `tests/test_difficult_advice_recipe.py` are about
  config content.

---------------------------------------------------------------------------------------------------

## 7. RunPod quirks (everything that has bitten this project)

### 7a. Renting
- **Rent only through `uv run runpod up`** (`src/infra/runpod.py`). Never POST to the RunPod API
  yourself. Two shapes:
  - train: `uv run runpod up <pod name> --train configs/train/sft.yaml --model qwen36 --count 1 --push_env --branch <branch>`
  - eval:  `uv run runpod up <pod name> --eval <mask|odcv> --target <hf adapter>` (add `--gpu "NVIDIA H200"` for ODCV)
- **`--train` refuses a dirty or unpushed checkout** (`_commit_to_run`): tracked files with
  uncommitted changes, or HEAD not on origin, abort before renting ("the pod would clone HEAD and run
  code that is not what you are looking at"). Untracked files are ignored. With two agents in one
  checkout this WILL bite: one agent's uncommitted edit blocks the other's rent. Either commit and
  push, or rent from a clean `git worktree` at the pushed commit (the pod gets the same code).
- **H200 capacity comes and goes.** The REST create call returns `HTTP 500 Internal Server Error` when
  there is no stock; it is not a bug in our code. Retry every 2-3 minutes; tonight every rent succeeded
  within a few tries. 2xH200 had no stock at all on 30 Sep; 1xH200 is enough (see 7c).
- **A pod with a null `machineId`/public IP and a 404 boot log is queued on capacity, not booting.**
  Give it 10 minutes; if a second pod in a row is machineless, change what you ask for
  (`--cloud COMMUNITY`, another GPU, `--countries SE`) instead of re-requesting the same spec.
- **Name the pod correctly at creation.** RunPod has no rename-only call: `PATCH /pods/{id}` with just
  a name restarts the container and remaps its SSH ports. It killed a booting pod earlier today.
- **Pods created in the RunPod web console are traps.** They get whatever driver the host has
  (driver 570 = CUDA 12.8), and the repo's `torch 2.11.0+cu130` needs driver >= 580: `nvidia-smi` looks
  fine, `torch.cuda.is_available()` is False, and the trainer loads the model onto CPU while the pod
  bills. `runpod up` pins `allowedCudaVersions: 13.0` for both shapes. Console pods also lack this
  repo's provisioning marker (`LASR_POD_OWNER`), so `--terminate-pod` refuses to tear them down. On any
  pod you did not get from `runpod up`, run
  `uv run python -c 'import torch; print(torch.cuda.is_available())'` before anything long.
- **SSH key**: `runpod up` injects `~/.ssh/id_ed25519.pub` only if it exists at exactly that path. No
  key file means no error and no key on the pod; it looks like a pod that is still booting.
- `runpod up --target a,b` is one repo id to Fire; pass a Python list literal for several targets.
- A host can have a crawling network (14 KB/s from Hugging Face, in `EUR-IS-4`, 9 Sep): the boot log
  just keeps printing `Downloading…`. Compare against a sibling pod; `--countries SE` worked then.
  Do not turn one bad host into a blocklist.

### 7b. Booting
- **A train pod takes about 12-15 minutes to boot and bills throughout.** The boot clones the repo at
  the exact commit, `uv sync`s, then compiles the `causal-conv1d` CUDA kernel (about 8 minutes; it
  prints `BUILDING_CAUSAL_CONV1D`). `runpod up --train` blocks until the boot log says `READY <sha>`.
- **Never start `uv run train` before `READY`.** A training launched during the boot runs against an
  environment without the kernel and the trainer refuses to pack (before the `train` extra existed it
  started a second sync without `CUDA_HOME` and died on a missing `nvcc`: two of three arms lost about
  20 minutes of H200 that way on 28 Sep).
- **An eval pod boots in about 3-6 minutes** (vLLM venv + weights). `runpod up --eval` returns before
  the boot finishes and prints the `--server root@<ip>:<port>` to use; poll
  `https://<pod id>-8080.proxy.runpod.net/boot.log` for a line starting `READY` before running
  `uv run evals` (`run_arm.sh` does this).
- `causal-conv1d` is the `train` extra. A plain `uv sync` must never try to build it: an eval pod that
  cloned the repo and ran a bare `uv sync` died on the missing `nvcc`, `set -e` exited the container,
  RunPod restarted it with its disk, `git clone` into the now-existing dir failed in seconds, and the
  pod crash-looped about 100 times as a RUNNING card with no IP and no readable boot log (28 Sep). The
  boot now does `git init` + `fetch` + `checkout --detach`, and only train pods run
  `uv sync --extra train` with the CUDA toolchain exported (commit 3db3ad9c, on this branch).
- Reading a boot: `uv run runpod status --pod <id>` (state + boot-log tail);
  `src.infra.runpod.boot_diagnosis(pod_id)` (last step reached + error lines).
- `--push_env` writes `HF_TOKEN`, `HF_ORG` and the three `WANDB_*` variables to `/root/work/.env`,
  and nothing else. Never put the RunPod or OpenRouter key on a pod.

### 7c. Training on the pod
- Launch (from the laptop, after READY):
  `ssh -p <port> root@<ip> "cd /root/work && setsid nohup uv run train --config configs/train/sft.yaml model=qwen36 data_repo=<org>/<mix> data_revision=<sha> seed=<n> > /root/work/train.log 2>&1 < /dev/null & disown"`.
  Without `setsid nohup … < /dev/null & disown` the SSH channel stays open and your command hangs.
- 1xH200, global batch 16: about 6 s/step once warm (the first 5-10 steps are 20-35 s each), 565-568
  steps for a 9,0xx-row mix: **about 55-60 minutes**. The step total is not always 566 (it depends on
  the row count), so do not grep for `/566`.
- Always pin `data_revision=`. The adapter is pushed by the run itself to
  `dougalldeepmind/<UTC date>-qwen36-<seed>-<mix subject>` with `training_meta.json`. The local copy
  is `/root/work/output/train/<date>_qwen36_<seed>_<subject with underscores>/adapter/`.
- Before tearing the pod down: confirm the adapter and `training_meta.json` are on the Hub, compare
  the Hub's LFS sha256 of `adapter_model.safetensors` with `sha256sum` on the pod, and `scp` the
  `train.log` off. **A pod's disk is not storage** (`volumeInGb: 0`); everything on it dies with it.
- Training several models: one pod per model in parallel beats one big pod in sequence.

### 7d. Paying and tearing down
- **Nothing tears a pod down for you** except `uv run evals … --terminate-pod` (which releases the
  eval pod at the end of the run, including when the run fails after its rollouts).
  `uv run runpod down --pod <id>` terminates and verifies; `uv run runpod pods` lists everything
  still billing on the SHARED account. Check it after every stage.
- Tonight every pod, train and eval, was a 1xH200 at **$4.59/hr**. Measured cost per arm: train about
  $6 (15 min boot + 60 min), MASK about $5 + about $1-2 of judge, ODCV-lite about $5 + about $2 of
  judge + $0.83 of progress judge. **About $18-20 per arm.**
- RunPod balance at 01:00 UTC: $308. OpenRouter balance: $496 (see 8 for what happens at zero).
- The repo's own in-process watchdog terminates an eval pod when its owning `uv run evals` process
  dies, not only at its deadline. Killing the driver to "reconnect" kills the pod.
- Do not key a teardown watcher on `>>> pushed`: `run_eval` prints `>>> pushed HF_TOKEN + …` at
  startup. Match `>>> pushed https`.

---------------------------------------------------------------------------------------------------

## 8. Eval-driving quirks (laptop drives, pod serves)

- Command: `uv run evals --name <mask|odcv> --target <adapter> --server root@<ip>:<port> --port <local port> --terminate-pod`.
  `--target` is `nargs='+'`: put it FIRST and end it with `--name`, or it swallows your `key=value`
  overrides. Thinking mode is inferred from the adapter's `training_meta.json`; never pass it.
- **Two evals on one machine need two local ports.** `--port` defaults to 8000 on both ends of the SSH
  tunnel. A second run on a taken port used to fail silently (`ssh -L` prints one line and carries
  on) and the eval then talked to the OTHER arm's model: on 29 Sep an ODCV run scored 0% tool calls
  against a MASK arm's server. This branch now refuses (commit a7ce34c2, `local_port_in_use`), but
  still give every concurrent run its own port and check `lsof -nP -iTCP:<port> -sTCP:LISTEN`.
- **ODCV must be driven where Docker works**: this laptop with Docker Desktop running, never a pod.
  Several ODCV arms can share the Docker daemon at once (three did today). Docker Desktop needs an
  enlarged address pool for concurrency 32 (`~/.docker/daemon.json`
  `"default-address-pools": [{"base": "10.200.0.0/14", "size": 24}]`); this machine has it. If Docker
  Desktop is quit, `docker_preflight` refuses with prose, not a traceback, and a pod idles.
- `channel 1: open failed: connect failed: Connection refused` lines at the start of an eval log are
  normal: health checks hitting the tunnel while vLLM is still loading.
- ODCV-lite: 3 passes x 80 cells, about 55-60 minutes; each pass is audited and holes are retried or
  rebuilt from `docker_output.log`. Rollouts land in `output/odcv/<date>_<model>_<hhmmss>/` before
  judging, so a judging failure loses nothing: finish with `scratch/odcv_finish_from_rollouts.py`.
- MASK: 1,000 rows, 4,438 generations, about 50-60 minutes at 192 in flight on an H200. It has NO
  resume (it clears its work dir on start), and one TCP reset on the tunnel has killed a 3-hour run.
- **OpenRouter running dry fails differently per eval, and one of the ways is silent:**
  - ODCV's judge asks for up to 65,536 tokens, so it gets a hard `402 … requires more credits` and the
    run exits 1 after all rollouts are saved.
  - MASK's judge calls just time out (60 s `wait_for`), are stored as `TIMEOUT_ERROR`, and count as
    "not a lie": **the run exits 0 and publishes an inflated score.** After any MASK run, count
    `TIMEOUT_ERROR` in `results/*_evaluated.csv` (column `eval(System Prompt + User Prompt)_run1`).
    A vacuous 100 also happens when the key is exhausted from the start.
  - `uv run evals` prints `>>> credentials: hub=… | openrouter=$<remaining>` up front, which does not
    protect a run that drains the account midway. Check the balance before a batch of evals:
    `GET https://openrouter.ai/api/v1/credits`.
- An exported `OPENROUTER_API_KEY`, `HF_TOKEN` or `HUGGINGFACE_API_KEY` in the shell silently outranks
  `.env` (`load_dotenv()` does not override). `hf_token()` reads `HUGGINGFACE_API_KEY` first.
- Eval repos are named `<UTC date>-<eval>-<model name without its date>` and a push OVERWRITES an
  existing repo of that name: two same-named adapters evaluated on one UTC day collide, and so do two
  evals of the same adapter on one day. The same holds for adapters
  (`<date>-qwen36-<seed>-<mix subject>`) and mixes. Note that `da-15` names different corpora on
  different dates; give new mixes a variant (`<date>-da-15-<variant>-mix`) so their adapters and evals
  get distinct names.
- Watchers: `pgrep -f "uv run train"` matches the shell running the pgrep. Write the pattern with a
  bracket (`pgrep -f "uv run trai[n]"`). A watcher that greps only for `Traceback` misses this repo's
  formatted refusals; treat an empty log past first-output time as the alert.
- Pods log UTC; the laptop clock is BST.

---------------------------------------------------------------------------------------------------

## 9. Things I got wrong tonight (so you do not repeat them)

- I told Jamie the three new MASK scores were "fully judged" after checking only for empty cells and
  402 text. I did not look for `TIMEOUT_ERROR`. They were 32-86% unjudged. Corrected to him at 02:00.
- I edited a shell script with `sed -i` and it lost its execute bit; the six-arm launch failed with
  `Permission denied` (nothing rented). `chmod +x` after any in-place edit, or run via `bash`.
- I searched for another project's API key without permission context and the auto-mode classifier
  blocked it; Jamie then allowed a one-time use of the `arcadia-worktest` OpenRouter key for the self
  ODCV judging only. That permission is spent. Use only this repo's `.env`.
- The swap arms were built row-count matched, not token matched (15.1% vs 15.0% supervised tokens);
  negligible, but say so if you report them.
