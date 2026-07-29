# Synthetic Finetuning for Constitution: Research Log

A local-only, file-driven React application for reading experiment notebooks,
evaluation results, model lineages, synthetic dialogue datasets, Petri audits,
structured metrics, and research findings.

The browser is deliberately read-only. Research is authored as Markdown, JSON,
JSONL, images, and other artifacts on disk; the app indexes those sources into
an explorable research console.

This application is a standalone project rooted at `Visualizer/`. It is one of
two top-level directories in the repository; the vulnerability investigation
lives separately under `Vulnerabilities/` and shares no build tooling with the
visualizer.

## Quick start

### Requirements

- Node.js `22.13.0` or newer
- npm, included with Node.js

From the `Visualizer` directory (`cd Visualizer` from the repository root):

```powershell
npm ci
npm run validate:content
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The development command
indexes the corpus before startup and watches `content/` for changes. Saving a
research file normally updates the app without restarting it.

On Windows, if PowerShell blocks `npm.ps1`, use the command shim instead:

```powershell
npm.cmd ci
npm.cmd run dev
```

To verify the complete application before committing:

```powershell
npm run validate:content
npm run lint
npm test
```

`npm test` creates a production build and runs the rendered-route tests.

## Netlify deployment

The project-local [`netlify.toml`](netlify.toml) lives inside `Visualizer/`.
Configure Netlify's package directory as `Visualizer`; Netlify then installs and
builds only this package, and the sibling `Vulnerabilities/` research workspace
is not part of the deployed application.

The Netlify build runs:

```powershell
npm run build:netlify
```

This first regenerates the content index and copies every colocated visual and
downloadable artifact into `public/`, then creates a fully static Next.js export
in `out/`. Netlify publishes that export directly to its CDN. Do not point
Netlify at an existing `dist/` or `.next/` directory: the indexing step is
required for new research content and visuals to appear.

The production site is connected to the repository's `main` branch. A push to
`main` triggers a new production build; pull requests receive deploy previews.

## How content reaches the app

```text
content/ source files
        │
        ▼
scripts/index-content.mjs
        │
        ├── lib/generated/content-index.json
        ├── public/content-assets/
        └── public/generated-datasets/
```

Treat files under `content/` as the source of truth. Do not hand-edit the three
generated outputs. Running `npm run content:index`, `npm run dev`, or
`npm run build` recreates them.

The validator is report-only: it emits errors and warnings but never rewrites
research files. The importer also leaves its input untouched, displays proposed
metadata, and creates a separate prepared copy only after confirmation.

## Content layout

```text
content/
  logs/
    <record-slug>/
      index.md
      assets/       # figures displayed in Markdown
      artifacts/    # small downloadable run outputs
  evals/
    <record-slug>/index.md
  findings/
    <record-slug>/index.md
  datasets/
    <dataset-slug>/
      index.md      # dataset card and construction notes
      data/*.jsonl  # dialogue records
  petri-runs/
    <run-slug>/
      index.md
      data/scenarios.jsonl
      results/transcripts.jsonl
      results/scores.json
```

Every visible research record is a directory containing `index.md`. Small
images and raw artifacts may be colocated in `assets/` or `artifacts/`; the
indexer copies them into the read-only browser preview. The `Visualizer/artifacts/`
directory is ignored and is intended for large local machine outputs that should
not be committed.

## Adding research

For a new log, evaluation, or finding, run:

```powershell
npm run new:content
```

This creates a dated scaffold under the selected collection. For an existing
Markdown file located elsewhere:

```powershell
npm run import:content -- C:\path\to\source.md --type=logs
```

The importer prints the inferred metadata and asks before writing. Pass
`--yes` only in automation after inspecting what will be generated:

```powershell
npm run import:content -- C:\path\to\source.md --type=evals --yes
```

After adding or changing content:

```powershell
npm run validate:content
npm run content:index
```

## Markdown record format

All records use YAML frontmatter followed by normal Markdown. A useful shared
baseline is:

```markdown
---
title: "Concise human-readable title"
date: 2026-07-28
summary: "One sentence stating what this record contributes."
status: complete
model_id: qwen3-32b
models:
  - qwen3-32b
tags:
  - sft
  - constitution
  - ood
---

# Concise human-readable title

## Summary

State the question, intervention, strongest result, and main caveat.
```

`title` is required. `date` is strongly recommended; otherwise the file
modification date is used. Unknown frontmatter fields are preserved, so new
research metadata does not require an application change.

The Markdown renderer supports:

- headings, lists, links, tables, code, block quotes, and footnotes;
- images such as `![caption](./assets/plot.png)`;
- inline LaTeX such as `$P(y \mid x)$`;
- display LaTeX enclosed by `$$`;
- callouts written as `> [!NOTE]`, `> [!WARNING]`, or `> [!CAUTION]`;
- links to colocated raw files such as `[raw metrics](./artifacts/metrics.json)`.

Prefer a short summary near the top, then methods, results, caveats, and links
to raw evidence. Do not paste large machine dumps into the readable document.

## Evaluation metadata and metrics

Evaluation records belong under `content/evals/`. Use stable identifiers so
the app can compare like with like:

```yaml
model_id: qwen3-32b
checkpoint_id: post-sft-reasons-v1
parent_checkpoint_id: qwen3-32b-base
training_stage: sft
training_method: lora
run_id: sft-reasons-seed-1
seed: 1
eval_suite: agentic-misalignment
eval_version: v1
dataset_version: reasons-v1
git_commit: abc123
status: complete
tags:
  - reasons-rich
  - constitution
  - ood
metrics:
  agentic_misalignment_rate:
    value: 0.07
    unit: proportion
    lower_is_better: true
  constitution_adherence:
    value: 0.84
    unit: proportion
    lower_is_better: false
  cost_usd:
    value: 12.40
    unit: USD
  runtime_minutes:
    value: 38
    unit: minutes
```

Metrics are optional and open-ended. Every metric must have a numeric `value`;
`unit`, `lower_is_better`, and additional fields are optional. Unknown metrics
still render. Comparative plots group only records with identical
`eval_suite`, `eval_version`, and `dataset_version` values.

Use `training_stage` to describe the checkpoint represented by the run—for
example `base`, `midtraining`, `sft`, `bounded-dpo`, or `rl`. Preserve branching
with `parent_checkpoint_id` rather than implying that every model follows the
same linear pipeline.

## Synthetic dialogue datasets

A dataset needs a Markdown card and at least one JSONL file below `data/`.
Recommended dataset frontmatter:

```yaml
title: "Reasons-rich constitutional dialogue mixture"
date: 2026-07-28
summary: "Synthetic dialogues for reasons-rich constitutional SFT."
dataset_id: reasons-rich-aft-v1
dataset_version: v1
format: jsonl
training_objective: sft
generator_model: teacher-model-id
status: draft
models:
  - qwen3-32b
tags:
  - synthetic-dialogues
  - constitution
```

Write one valid JSON object per line. `messages` is preferred:

```json
{"id":"rr-0001","messages":[{"role":"system","content":"You are a careful research assistant."},{"role":"user","content":"Does this one transcript prove the checkpoint is misaligned?"},{"role":"assistant","content":"No. It is a concrete lead, not a checkpoint-wide estimate."},{"role":"user","content":"What should we run next?"},{"role":"assistant","content":"Run matched scenario variants and compare against the parent checkpoint."}],"metadata":{"split":"train","category":"epistemic-honesty","turn_structure":"multi-turn","principles":["match claims to evidence"],"quality_score":0.97}}
```

The viewer also accepts `conversation`, `turns`, `dialogue`, or
`prompt`/`response`, but agents should emit `messages` unless converting a
pre-existing format. Metadata is open-ended and displayed without a registry.
Keep stable record IDs, explicit roles, and genuine follow-up turns in
multi-turn examples.

At index time, JSONL is split into browser-sized chunks while the source JSONL
remains available for download.

## Petri run format

A Petri run combines its Markdown research note with these optional structured
files:

- `data/scenarios.jsonl` — generated scenario seeds and hypotheses;
- `results/transcripts.jsonl` — multi-turn auditor/target transcripts, outcomes,
  scores, judge summaries, and tags;
- `results/scores.json` — aggregate outcome and category summaries.

The Markdown frontmatter should identify the pipeline:

```yaml
petri_run_id: petri-qwen3-reasons-v1
petri_version: v2.0.0
target_model_id: qwen3-32b
target_checkpoint_id: post-sft-reasons-v1
auditor_model_id: auditor-model-id
judge_model_id: judge-model-id
realism_model_id: realism-model-id
seed_set: sfc-petri-seeds-v1
max_turns: 8
realism_filter: true
realism_threshold: 0.6
status: needs-review
```

Each transcript should contain `id`, `scenario_id`, `category`, `outcome`,
`messages`, a `scores` object, `judge_summary`, and `tags`. Keep the exact
qualitative transcript even when aggregate scores are available: the interface
is designed to show both.

For a standalone contract that can be copied into an external Petri repository,
see [Claude Code Petri export guide](docs/CLAUDE_CODE_PETRI_EXPORT_GUIDE.md).

## Agent output contract

Research agents working in this repository should follow this contract:

1. **Write source material only.** Create or update files below `content/`.
   Never directly edit `lib/generated/content-index.json`,
   `public/content-assets/`, or `public/generated-datasets/`.
2. **Do not mutate original research artifacts.** When the input lives outside
   this repository, use the importer or create a prepared copy. Preserve source
   paths, hashes, run IDs, checkpoint IDs, dataset versions, and commits when
   known. Do not invent missing provenance.
3. **Choose the correct collection.** Training process and runtime notes go in
   `logs`; measured evaluation outputs go in `evals`; synthesized claims and
   counterevidence go in `findings`; training dialogues go in `datasets`; Petri
   pipeline outputs go in `petri-runs`.
4. **Make the first screen useful.** Provide a descriptive `title`, one-sentence
   `summary`, meaningful `status`, stable model/checkpoint identifiers, and
   specific tags. Put the conclusion and largest caveat near the top of the
   Markdown body.
5. **Separate narrative from machine data.** Explain what happened in readable
   Markdown. Link to colocated raw JSON, JSONL, CSV, logs, or images instead of
   pasting large dumps into the document.
6. **Emit structured metrics when available.** Use a numeric `value` and retain
   the original unit and preferred direction. Do not rename a metric between
   comparable runs. Do not add fabricated zeros for missing measurements.
7. **Preserve comparison keys.** Comparable evaluation runs must use identical
   `eval_suite`, `eval_version`, and `dataset_version` values. Use different
   versions when the evaluator or data actually changes.
8. **Represent lineage explicitly.** Set `model_id`, `checkpoint_id`,
   `parent_checkpoint_id`, `training_stage`, and `training_method` where known.
   SFT and bounded-DPO branches should remain distinguishable.
9. **Format dialogue data as real conversations.** Prefer `messages` with
   explicit `system`, `user`, and `assistant` roles. Keep each JSONL record on
   one physical line. Include multiple user–assistant rounds when the research
   depends on interaction effects.
10. **State epistemic status.** Mark fictional/demo data, failed or partial
    runs, uncertainty, counterevidence, and known confounds prominently. Match
    the strength of the prose claim to the evidence.
11. **Use portable links.** Link colocated files with relative paths:
    `./assets/...` or `./artifacts/...`. Add useful image alt text and captions.
12. **Validate the handoff.** Run `npm run validate:content` and
    `npm run content:index`. Report warnings rather than silently “fixing”
    uncertain metadata.

An agent completing a research task should leave a handoff containing:

```text
Created/updated:
- content/<collection>/<slug>/index.md
- content/<collection>/<slug>/<structured or artifact files>

Identity:
- model_id:
- checkpoint_id:
- parent_checkpoint_id:
- run_id or dataset_id:
- git_commit:

Evidence status:
- status:
- strongest result:
- largest caveat:
- missing or failed measurements:

Checks:
- validate:content result
- content:index result
```

## Command reference

| Command | Purpose |
| --- | --- |
| `npm ci` | Install the locked dependency set |
| `npm run dev` | Index, watch content, and start the local app |
| `npm run new:content` | Create a log/eval/finding scaffold |
| `npm run import:content -- <path> --type=<type>` | Preview and prepare an immutable imported copy |
| `npm run validate:content` | Report content errors and warnings without editing |
| `npm run content:index` | Regenerate the browser-facing content index and previews |
| `npm run lint` | Check application code |
| `npm test` | Build and run rendered-route regression tests |
| `npm run build` | Create the production build |
| `npm run start` | Serve an existing production build |

See [docs/RESEARCH_PROGRAM_STRUCTURE.md](docs/RESEARCH_PROGRAM_STRUCTURE.md)
for the research assumptions behind the data model.
