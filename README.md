# Synthetic Finetuning for Constitution: Research Log

A local-only, file-driven React application for browsing experiment notebooks,
evaluation results, model lineages, structured metrics, and research findings.
The browser is a read-only analysis surface: research content is authored and
updated as files.

## Content layout

```text
content/
  logs/       # readable experiment notebooks and linked machine artifacts
  evals/      # evaluation results with optional structured metrics
  findings/   # curated claims, uncertainty, and counterevidence
```

Each research record lives in its own directory as `index.md`. Images and small
artifacts may be colocated in `assets/` and `artifacts/`. Large artifacts should
live in the root-level ignored `artifacts/` directory and be referenced through
a generated manifest when that integration is added.

Original research files are immutable source material. Importing creates a
prepared copy with source path, timestamp, and SHA-256 provenance.

## Commands

```powershell
npm run dev
npm run new:content
npm run import:content -- C:\path\to\source.md --type=logs
npm run validate:content
npm run content:index
npm run build
```

`npm run dev` indexes the corpus, watches it for changes, and updates the local
application. The validator reports only; it never modifies content.

## Metric schema

Metrics are open-ended. Any metric with a numeric `value` renders:

```yaml
metrics:
  agentic_misalignment_rate:
    value: 0.07
    unit: proportion
    lower_is_better: true
```

Comparative plots group only identical `eval_suite`, `eval_version`, and
`dataset_version` values. Unknown training stages and metrics remain visible.

See [docs/RESEARCH_PROGRAM_STRUCTURE.md](docs/RESEARCH_PROGRAM_STRUCTURE.md)
for the research-specific assumptions behind the data model.

