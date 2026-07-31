<!-- ABOUTME: Root guide to the repository: the src/scripts/scratch code layout, the -->
<!-- ABOUTME: dashboard app, repository-wide conventions, and where the audit record went. -->

# Teaching Claude Why — replication and alignment auditing

Two lines of work on whether training a model on a written specification changes
its behaviour, plus a web frontend that presents the results.

```text
.
├── src/                  # correctness-critical reusable code (human-verified; import as src.*)
│   ├── openrouter.py, utils.py  #   shared OpenRouter client + utilities
│   ├── data/             #   data generation: synthdoc/, the SFT/DPO dataset pipeline, mixtures
│   ├── train/            #   QLoRA SFT, DPO training, adapter merging
│   └── eval/             #   capabilities/ · misalignment/ (ODCV) · vulnerabilities/ (petri, surf)
├── scripts/              # pipelines: thin CLIs over src/ functions + GPU-box shell drivers
├── scratch/              # one-off and AI-generated scripts (default home for new code)
├── configs/              # OmegaConf YAML, one per pipeline step
├── tests/                # fast offline unit tests
├── docs/                 # reference material + docs/replication.md (the run guide)
├── dashboard/            # research-log frontend (Next/vinext), deployed on Netlify
└── LOG.md                # append-only research log
```

**Run everything from the repository root.** `configs/`, `data/`, `output/` and
`third_party/` are resolved against the current directory, and `uv sync`
installs `src/` editable so `import src.*` works from anywhere — locally and on
remote boxes alike; there are no `sys.path` tricks.

### Remote GPU boxes

Remote machines use uv exactly like local ones — install it, clone, sync, run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <this-repo> /root/work && cd /root/work
uv sync                          # base deps + src/ installed editable
uv pip install vllm==0.8.5 "transformers==4.51.3" trl==0.19.1 \
    peft bitsandbytes accelerate wandb          # pinned GPU stack (CLAUDE.md gotcha 1)
uv run --no-sync scripts/train_lora.py --config configs/train_lora.yaml
```

The GPU stack is layered in with `uv pip` rather than declared in
`pyproject.toml` because its pins conflict with the lockfile (vLLM 0.8.5
requires `transformers==4.51.3`, and vLLM has no macOS wheels), so the box venv
intentionally diverges from `uv.lock`. **Always use `uv run --no-sync` on the
box** — a plain `uv run` re-syncs the venv to the lock and silently undoes the
transformers downgrade.

| Area | What it is | How to work in it |
| --- | --- | --- |
| [`docs/replication.md`](docs/replication.md) | End-to-end guide to the *difficult advice* replication from Anthropic's [Teaching Claude Why](https://www.anthropic.com/research/teaching-claude-why) on Qwen3-32B. Headline: **19.3% → 8.0%** agentic misalignment with thinking-format training. | `uv sync && uv run pytest -q`, then `uv run scripts/<step>.py` per the guide |
| [`src/data/synthdoc/`](src/data/synthdoc/README.md) | Config-driven synthetic document generation (self-contained package). | `uv run synthdoc run --config smoke.yaml --smoke` (offline, free) |
| `src/eval/vulnerabilities/` | Generalized Petri + SURF audit tooling from the completed MSM audit. Inspect's dependency pins conflict with the root env, so petri tools run in the nested project's env. | `uv run --project src/eval/vulnerabilities/petri/petri-subscription python src/eval/vulnerabilities/petri/<tool>.py --help` |
| [`dashboard/`](dashboard/README.md) | The research-log web app: datasets, eval runs, Petri results, findings. Self-contained Node project. | `cd dashboard && npm ci && npm run dev` |

## The MSM audit record

The completed Petri + SURF audit of the Model Spec Midtraining checkpoints —
evidence, seeds, rubrics, eval logs, the 21 numbered research docs, JOURNAL.md,
and the provider/watchdog infrastructure — was removed from the tree tip during
the 2026-07-30 restructure. **Git history is the archive**; recover any of it
with:

```bash
git checkout b38da52 -- experiments/vulnerabilities
```

Its public-facing results remain in `dashboard/content/` (the focused-discovery
Petri run) and on Hugging Face; the reusable tooling lives on, generalized, in
`src/eval/vulnerabilities/` with each file citing its original at `b38da52`.

## Conventions

Read [`CLAUDE.md`](CLAUDE.md) — the agent operating guide and repository-wide
conventions — before generating data, running an experiment, or committing.
The rule that bites soonest:

> **Datasets, generated corpora, evaluation outputs and their caches go to
> Hugging Face, not into git.** HF repos are named
> `<YYYY-MM-DD>-<short-experiment-description>` using the date the data was
> *generated*. Every dataset card states the experiment, the generation date,
> and **which constitution or model spec it connects to** - written as `none`
> explicitly when it connects to none.

Code, configs, seeds, rubrics, analysis and reports stay in git. Bulk data does
not. New AI-generated one-off code defaults to `scratch/`; nothing imports from
`scratch/`.

## Credentials

Secrets never enter the repository. All credentials live in one gitignored
`.env` at the repo root — copy [`.env.example`](.env.example) and fill it in;
see CLAUDE.md's Secrets section for the rules. `.env`, `*.env`, `*.pem` and
`*.key` are ignored repository-wide from the root `.gitignore`, deliberately,
so the guard applies to every nested project.

- The replication pipeline reaches Claude through **OpenRouter** only.
- The audit tooling uses the **Anthropic API** (auditor/judge roles) plus GPU
  provider keys.

## Deployment

`dashboard/` deploys to Netlify on every push to the default branch. The root
[`netlify.toml`](netlify.toml) sets the base directory; the build command and
publish directory live in `dashboard/netlify.toml`, so the two do not drift.

The repository-to-site link lives in the Netlify dashboard, not in git. If the
deploying repository changes, the site must be re-linked there - nothing in the
repository can restore it.

## History

This repository is the merge of two previously separate repositories, brought
in with full commit history rather than copied. On 2026-07-30 it was
restructured: the Python project flattened from `experiments/teaching-claude-why/`
into root-level `src/` + `scripts/` + `scratch/`, `Visualizer/` renamed to
`dashboard/`, and the frozen audit record removed at the tip (see above). The
record of how each finding was reached - including the ones that did not
survive scrutiny - is preserved in `git log`.
