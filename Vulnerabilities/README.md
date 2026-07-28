# MSM out-of-distribution vulnerability audit

Adaptive Petri investigation of models released from the **Model Spec
Midtraining** work ([arXiv:2605.02087](https://arxiv.org/abs/2605.02087),
repository `chloeli-15/model_spec_midtraining`).

**Research question.** Did model-spec midtraining, followed by alignment
fine-tuning, introduce, amplify, conceal or leave unresolved previously
unreported out-of-distribution alignment/generalization vulnerabilities?

The objective is *not* to re-demonstrate that weak rules-only specifications
fail. It is to find new information about the strongest apparently successful
MSM models. Anything that merely replicates a published result is excluded up
front by the [known-findings exclusion matrix](docs/00-exclusion-matrix.md).

**Primary discovery target.** `chloeli/qwen-3-32b-philosophy-spec-msm-aft-cot`,
a LoRA adapter over `Qwen/Qwen3-32B`. Every metadata claim about it is verified
against the model card, `adapter_config.json`, downloaded file hashes and the
pinned revision before use; see the preflight document.

All reporting follows
[`../Visualizer/docs/CLAUDE_CODE_PETRI_EXPORT_GUIDE.md`](../Visualizer/docs/CLAUDE_CODE_PETRI_EXPORT_GUIDE.md),
which is authoritative for report style and structure, transcript export shape,
figures, tables, metadata, evidence linking and the final summary format.

## Directory map

| Path | Contents |
| --- | --- |
| `docs/` | Numbered investigation documents: exclusion matrix, provider comparison, preflight, pilot assessment, focused-discovery plan. |
| `reports/` | Final research reports and result documents. |
| `seeds/` | Petri seed definitions. Every custom seed carries a hypothesis, mechanism, novelty argument, closest prior evaluation, expected evidence, control and falsification criteria. |
| `exports/` | Petri export bundles in the exact shape required by the export guide (`<yyyy-mm-dd>-<slug>/index.md`, `data/`, `results/`, `artifacts/`, `assets/`). |
| `analysis/` | Analysis code and intermediate tables. |
| `figures/` | Generated figures with captions. |
| `logs/` | Raw Inspect/Petri evaluation logs and vLLM server logs. |
| `scripts/secrets/` | Generic local secret wrappers. See the policy below. |
| `scripts/provider/` | Vast.ai and RunPod API clients: offer queries, provisioning, termination, balance reads. |
| `scripts/remote/` | Remote environment setup, vLLM launch, SSH tunnel management. |
| `scripts/petri/` | Petri/Inspect orchestration. |
| `runtime/provider-monitor/` | Live credit and resource indicator: monitor script, JSON status, Markdown status, append-only history, process state. |
| `runtime/watchdog/` | Independent cleanup watchdog. |
| `runtime/checkpoints/` | Resumable phase checkpoints. |
| `costs/` | Cost ledgers for GPU and Anthropic spend. |
| `evidence/` | Timestamped evidence: reorganization validation, credential validation, provider status history, cleanup proof. |

## Secret handling

Secrets live outside the repository in `$HOME\.config\msm-audit\` and are read
only through the wrappers in `scripts/secrets/`. No value is ever printed,
echoed, logged, summarized, committed or written to a generated file.

Two wrappers keep the two credential domains in separate processes:

| Wrapper | Reads | Injects |
| --- | --- | --- |
| `Invoke-WithInfraSecrets.ps1` | `infra.env` | `VAST_API_KEY`, `RUNPOD_API_KEY`, `HF_TOKEN`, `MSM_SSH_PRIVATE_KEY` + budget limits |
| `Invoke-WithPetriSecrets.ps1` | `petri.env` | `ANTHROPIC_API_KEY` + Anthropic budget limit |

`ANTHROPIC_API_KEY` never enters the parent Claude Code process. Claude Code is
authenticated through a Claude Max subscription; only Petri, Inspect and
explicit Anthropic API child processes receive the key, and only through the
isolated Petri wrapper. `Start-ProcessWithSecretEnv` writes values directly into
a child process environment block, so they never become a variable in the
launching shell and never appear on a command line or in shell history.

Budget limits (`MAX_GPU_SPEND_USD`, `MAX_WALL_CLOCK_HOURS`,
`IDLE_SHUTDOWN_MINUTES`, `MAX_ANTHROPIC_SPEND_USD`) are operational
configuration rather than credentials, and are displayed by the live status
indicator as required.

Credential validation status:
[`evidence/credentials/credential-validation.md`](evidence/credentials/credential-validation.md).

## Hard limits

| Limit | Value | Source |
| --- | --- | --- |
| GPU spend | $40.00 | `MAX_GPU_SPEND_USD` |
| Wall clock | 36 h | `MAX_WALL_CLOCK_HOURS` |
| Idle shutdown | 30 min | `IDLE_SHUTDOWN_MINUTES` |
| Anthropic spend | $120.00 | `MAX_ANTHROPIC_SPEND_USD` |

Only target-model inference runs remotely. Everything else -- Petri, Inspect,
seeds, analysis, figures, reports, transcript exports, raw logs, provider
metadata, cost records and cleanup evidence -- stays local under this
directory.

## Progress

See [`JOURNAL.md`](JOURNAL.md) for the durable progress journal and
`runtime/checkpoints/` for resumable phase checkpoints.
