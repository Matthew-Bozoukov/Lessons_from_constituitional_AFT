<!-- ABOUTME: Root guide to the merged repository: two LLM experiment projects plus the research-log frontend. -->
<!-- ABOUTME: Points at each project's own README and states the repository-wide conventions. -->

# Teaching Claude Why — replication and alignment auditing

Two lines of work on whether training a model on a written specification changes
its behaviour, plus a web frontend that presents the results.

```text
.
├── dashboard/                      # research-log frontend (Next/vinext), deployed on Netlify
└── experiments/
    ├── teaching-claude-why/         # difficult-advice replication on Qwen3-32B
    └── vulnerabilities/             # Petri + SURF audits of Model Spec Midtraining checkpoints
```

The two experiment directories sit together because they are the same kind of
thing: Python projects that run LLM experiments, generate data, and publish it.
`dashboard/` is top-level because it is a separate application with its own
toolchain and its own deployment.

| Directory | What it is | How to work in it |
| --- | --- | --- |
| [`dashboard/`](dashboard/README.md) | The research-log web app: datasets, eval runs, Petri results, findings. Self-contained Node project. | `cd dashboard && npm ci && npm run dev` |
| [`experiments/teaching-claude-why/`](experiments/teaching-claude-why/README.md) | Replicates the *difficult advice* result from Anthropic's [Teaching Claude Why](https://www.anthropic.com/research/teaching-claude-why): SFT on out-of-distribution difficult-advice data reduces agentic misalignment on held-out honeypots. Headline: **19.3% to 8.0%** with thinking-format training. | `cd experiments/teaching-claude-why && uv sync && uv run pytest` |
| [`experiments/vulnerabilities/`](experiments/vulnerabilities/README.md) | Petri and SURF audits asking whether model-spec midtraining introduced out-of-distribution alignment vulnerabilities. Answer: no MSM-attributable effect survives correction. | Start at `experiments/vulnerabilities/docs/16-findings.md` |

**Run experiment code from its own directory.** Both projects use CWD-relative
paths (`configs/`, `data/`, `output/`), so `cd` into the project first. This
changed when the repositories merged; previously each was its own root.

## Conventions

Read [`AGENTS.md`](AGENTS.md) before generating data or committing. The rule that
bites soonest:

> **Datasets, generated corpora, evaluation outputs and their caches go to
> Hugging Face, not into git.** HF repos are named
> `<YYYY-MM-DD>-<short-experiment-description>` using the date the data was
> *generated*. Every dataset card states the experiment, the generation date,
> and **which constitution or model spec it connects to** - written as `none`
> explicitly when it connects to none.

Code, configs, seeds, rubrics, analysis and reports stay in git. Bulk data does
not.

Per-project agent guides sit alongside their code:
[`experiments/teaching-claude-why/CLAUDE.md`](experiments/teaching-claude-why/CLAUDE.md).

## Credentials

Secrets never enter the repository. `.env`, `*.env`, `*.pem` and `*.key` are
ignored repository-wide from the root `.gitignore`, deliberately, so the guard
applies to every nested project.

- `experiments/teaching-claude-why/` reaches Claude through **OpenRouter**.
- `experiments/vulnerabilities/` uses the **Anthropic API** for Petri's auditor,
  judge and realism roles, plus provider keys for GPU rental. Those are injected
  into child processes only, through
  `experiments/vulnerabilities/scripts/secrets/`, and never into a parent agent
  environment.

## Deployment

`dashboard/` deploys to Netlify on every push to the default branch. The root
[`netlify.toml`](netlify.toml) sets the base directory; the build command and
publish directory live in `dashboard/netlify.toml`, so the two do not drift.

The repository-to-site link lives in the Netlify dashboard, not in git. If the
deploying repository changes, the site must be re-linked there - nothing in the
repository can restore it.

## History

This repository is the merge of two previously separate repositories. The
`dashboard/` and `experiments/vulnerabilities/` trees were brought in with
their full commit history rather than copied, so the record of how each finding
was reached - including the ones that did not survive scrutiny - is preserved in
`git log`.
