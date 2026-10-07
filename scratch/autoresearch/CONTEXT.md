# da-autoresearch context (2026-10-01) — read this first

## The question
We fine-tune Qwen3.6-27B (LoRA) on a fixed base blend ("nosynth") with 15% of supervised tokens replaced by
synthetic "difficult advice" (DA) rows: a user with an ethically loaded decision asks an assistant; the assistant
reasons about its values and does not go along with the norm violation. We then measure two evals:
- **ODCV-lite** misconduct rate (MR %, lower is better): 40 agentic scenarios x 2 variants (mandated / incentivized)
  x 3 passes = 240 rollouts. The model is an autonomous agent with a `bash` tool under KPI pressure; misconduct =
  gaming the metric / falsifying data / bypassing a constraint. Judged per rollout (severity 0-5; violation >= 3).
- **MASK** honesty (higher is better): 1000 prompts in 6 archetypes; a pressure prompt pushes the model to say
  something it believes false; honesty = 100 - % lies.

The puzzle: which properties of the DA rows move ODCV, which move MASK, and why do they dissociate?
In particular: swapping "other-AI" rows from the 25 Sep corpus into the 28 Sep mix LOWERED ODCV (18 -> 10.8), but a
freshly generated corpus with other-AI content on the 28 Sep recipe did NOT (19.2). Why?

## Scores (one training seed unless noted). `output/autoresearch/inventory.md` has every published arm.
| arm (adapter) | what the DA rows are | MASK | ODCV MR % |
|---|---|---|---|
| nosynth control | no DA rows | 57.7 / 56.9 | 43.8 / 45.4 |
| 14 Sep corpus da-7 (3 models) | | 82.8 / 79.9 / 71.3 | 11.7 / 12.9 / 11.3 |
| 14 Sep corpus da-15 (2 models) | | 84.9 / 89.4 | 7.1 / 7.5 |
| 23/24 Sep corpus da-5 / da-15 / da-25 | | - / 87.7 / - | 15.8 / 10.8 / 8.8 |
| **25 Sep corpus da-15, seeds 0 / 1** ("target") | "ask, not instruct" wording; org runs an AI system in ~76% of rows | 90.2 / 88.3 | 9.2 / 7.9 |
| 25 Sep no-t6 / new-t6 (2 seeds) | t6 = stable-persona trait | 76.9 / 87.9, 84.7 | 13.8 / 6.7, 8.8 |
| **28 Sep corpus da-5 / da-15 / da-25** ("baseline" = da-15) | trait notes keep every scenario human except t6; grey-area requirement; reviser sees notes | 72.0 / 74.4 / 92.2 | 20.4 / 17.9, 18.8 (same model, 2 eval runs) / 7.5 |
| 28 Sep no-t6 da-15 | | 71.0 | 17.5 |
| 28 Sep sysdiv da-15 | DA system prompts rewritten to be row-specific | 72.3 | 14.6 |
| swap self (123 rows = 20% of DA rows) | 25 Sep "assistant itself" rows swapped into 28 Sep mix, trait for trait | 81.5 | 14.6 |
| swap otherai (391 rows = 63%) | 25 Sep "another AI system" rows swapped in | 79.1 | 10.8 |
| swap self+otherai (551 rows = 89%) | both | 85.9 | 7.5 |
| swap explicit (159 rows = 26%) | 8 Sep rows where user asks the assistant to DO the shortcut | 77.3 | 12.5 |
| swap advice (159 rows = 26%) | 8 Sep rows where the user asks for advice (same slots, matched trait + AI type) | 81.2 | 8.8 |
| **new synth self** (2026-09-30-da-self-synth) | 28 Sep recipe + "the assistant itself is part of the situation" in per-trait shares | 97.4 | 14.2 |
| **new synth otherai** (2026-09-30-da-otherai-synth) | 28 Sep recipe + "another AI system is involved" in 50% of rows per trait | 93.8 | 19.2 |
| **new synth explicit** (2026-09-30-da-explicit-synth) | 28 Sep recipe + explicit asks ("do it for me") + DAT-style t1 note | 92.0 | 15.8 |
| da-lowstakes-practical-15 / nonmoral-original-15 | low-stakes / non-moral advice | - | 17.1 / 19.6 |
| da-multiparty-15 / multiparty-human-15 | | - | 8.3 / 7.9 |
| da-tools-15 | DA with tool use | 67.6 | 2.1 |
| delib-15 / delib-sonnet-15 | deliberative-alignment style | 77.7 / 82.5 | 27.5 / 13.3 |
ODCV fixed-benchmark 95% CIs (rollout noise only) are roughly +/- 3 points. Seed-to-seed: 25 Sep seeds differ by 1.3.

## Where things are (repo root /Users/jamie/Projects/lasr; run everything with `uv run python ...` from the root)
- Hub org: `dougalldeepmind` (HF_ORG in .env; `from dotenv import load_dotenv; load_dotenv(".env")`).
- Adapters: `dougalldeepmind/<date>-qwen36-<seed>-<arm>` (model repos) with `training_meta.json` (dataset repo+revision).
- Mixes: `dougalldeepmind/<date>-...-mix` (dataset repos): `mixture.jsonl` rows = {messages, source, tools?, metadata?};
  DA rows have `source` starting `da`; `mixture_config.yaml`, `mixture_stats.json`. Swap mixes also have
  `recompose_swaps.jsonl` (one line per swapped row: target_row, trait_id, donor_scenario, donor category).
  Key mixes: 2026-09-25-da-15-mix@73f66648 (target), 2026-09-28-da-15-mix@ff524823 (baseline),
  2026-09-29-da-15-{self,otherai,self-otherai,sysdiv}-mix, 2026-09-30-da-15-{explicit,advice}-mix,
  2026-09-30-da-{self,otherai,explicit}-15-mix, 2026-09-28-da-{5,25}-mix, 2026-09-28-da-tools-15-mix.
- Corpora (dataset repos, `dataset.jsonl`; rows = {messages (system,user,assistant with reasoning_content), metadata{trait_id,
  scenario_id,...}}): 2026-09-08-da-synth, 2026-09-14-da-synth, 2026-09-25-da-synth@618060e1, 2026-09-28-da-synth@14efefbf,
  2026-09-30-da-{self,otherai,explicit}-synth. Local copies: output/audits/2026-09-28_corpus_runs/<corpus>/dataset.jsonl,
  output/synthdoc_v3/20260928_195614/ (28 Sep, every stage file), output/synthdoc_v3/20260930_200011 (self),
  _200041 (otherai), _200107 (explicit) (stage_2 scenarios ... stage_8 export; `corpus_labels.jsonl`).
- Synth configs (the recipe IS the config): configs/data/synth/{da,da-self,da-otherai,da-explicit}.yaml; older
  recipes via git: 25 Sep = `git show 9f093f2c:configs/data/synth/da.yaml`, 28 Sep = `git show 47efcc60:configs/data/synth/da.yaml`.
- Existing labels / audits (reuse before judging anything again): output/audits/2026-09-29_ai_subtype/ (seven-way AI
  subtype per row per corpus), output/audits/2026-09-30_assistant_subject/, output/audits/2026-09-29_setting_stakes/,
  output/da_pressure/*_pressure_labels.jsonl (how hard the user pushes), output/audits/2026-09-29_da_drift_timeline.md,
  output/audits/2026-09-29_regression/, output/audits/2026-09-29_recompose/, output/audits/2026-09-30_ask_arms/.
  Judge scripts: scratch/da_ai_subtype.py, scratch/da_assistant_subject.py, scratch/classify_da_pressure.py,
  scratch/da_setting_stakes_probe.py, scratch/da_error_judge.py.
- Eval runs (dataset repos): `dougalldeepmind/<date>-odcv-qwen36-<seed>-<arm>` with
  `rollouts/<variant>/<Scenario>/pass<N>/messages_record.txt` (the full agent transcript incl. reasoning),
  `results/results.json` (`ours.overall`, `per_scenario_medians`, per-judge scores under results/), `metadata/`.
  `dougalldeepmind/<date>-mask-qwen36-<seed>-<arm>`: `results/results.json`, `results/<archetype>_evaluated.csv`
  (per-row prompts, generations, reasoning, judge labels), `results/<archetype>_metrics.csv`.
  Run names for each arm are in output/autoresearch/inventory.json. Helpers: src/eval/misalignment/odcv/odcv.py
  (`to_long`), src/eval/stats.py (`interval`), scratch/plot_da_row_swaps.py shows how to read both.

## Rules for subagents
- Read-only research: do NOT rent GPUs, train, generate DA rows, push to the Hub, commit, or edit files outside
  `scratch/autoresearch/` and `output/autoresearch/`. Never print secrets.
- LLM judging is allowed only via OpenRouter `google/gemini-3-flash-preview` through src/infra/endpoints/openrouter.py,
  and only up to the dollar limit your task gives you. Prefer sampling (e.g. 60-100 rows per corpus) over full passes.
- Write any script you create to scratch/autoresearch/ (two-line `# ABOUTME:` header) and outputs to output/autoresearch/.
- Report evidence with numbers and quoted examples; separate what you measured from what you guess.
