# ResearchLog

Two independent top-level projects share this repository.

```text
ResearchLog/
├── Visualizer/       # local read-only research-log frontend (Node/Vite/vinext)
└── Vulnerabilities/  # Petri vulnerability investigation of Model Spec Midtraining models
```

| Directory | What it holds | How to work in it |
| --- | --- | --- |
| [`Visualizer/`](Visualizer/README.md) | The research-log web application, its content corpus, build tooling, and docs. Self-contained; treat it as a standalone project directory. | `cd Visualizer` then `npm ci`, `npm run dev` |
| `Vulnerabilities/` | Petri/Inspect seeds, audits, transcripts, analysis, figures, reports, provider monitoring, and cleanup evidence. | See `Vulnerabilities/README.md` |

Generated Petri files, Python environments, model logs, research reports, and
infrastructure scripts belong under `Vulnerabilities/` and must not be mixed
into `Visualizer/`.

## Conventions

See [`AGENTS.md`](AGENTS.md) for the rules that apply across every project here.

The one to know before generating anything: **datasets, generated corpora,
evaluation outputs and their caches are published to Hugging Face, not committed
here.** HF repo names are `<YYYY-MM-DD>-<short-experiment-description>` using the
date the data was generated, and every dataset card must state the experiment,
the generation date, and **which constitution or model spec it connects to** -
written as `none` explicitly when it connects to none. Code, configs, seeds,
rubrics, analysis and reports stay in git; bulk data does not.

## Repository-root files

Only files whose location carries repository-wide meaning stay at the root.
Every such exception is listed here.

| File | Why it stays at the root |
| --- | --- |
| `.gitignore` | Carries repository-wide meaning. It holds the environment/credential guard (`.env`, `*.env`, `*.pem`, `*.key`), which must apply to every directory in the tree, including `Vulnerabilities/`. Project-local ignore rules were split out into `Visualizer/.gitignore` and `Vulnerabilities/.gitignore`, whose paths are anchored to their own directories. |
| `README.md` | This file. Explains the top-level split and documents the exceptions above. The visualizer's own README moved to `Visualizer/README.md`. |

There is no `.github/` workflow directory, `.gitattributes`, `LICENSE`, or
editor-configuration file in this repository, so no further root-level
exceptions were required. Nothing else remains at the root: all visualizer
sources, assets, manifests, lockfiles, configuration, tests, scripts, and docs
moved into `Visualizer/`.
