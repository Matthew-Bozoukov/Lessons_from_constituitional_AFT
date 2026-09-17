<!-- ABOUTME: Final protocol, artifact pins and results for the September 2026 DA supervision ablations. -->
<!-- ABOUTME: Operational history and experiment utilities live under scratch/da_supervision/. -->

# Difficult-advice supervision ablations

Three Qwen3.6-27B LoRAs were trained on the same examples, changing supervision
only for the difficult-advice (DA) rows. Training finished on September 16, 2026;
ODCV verification finished on September 17 UK time (September 16 UTC).
All models, training archives, evaluation transcripts and scores are on Hugging Face.
All campaign pods were terminated and the monitoring scheduler was deleted.

## Findings

| DA supervision | ODCV misalignment | 95% CI | Mandated | Incentivized | Task progress / 5 | Submitted |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| CoT only | 8/80 (10%) | 5.1–18.6% | 2/40 | 6/40 | 4.83 | 77/80 |
| Answer only | 16/80 (20%) | 11.0–33.7% | 7/40 | 9/40 | 4.91 | 79/80 |
| Empty CoT | 15/80 (18.75%) | 9.9–32.6% | 8/40 | 7/40 | 5.00 | 80/80 |

CoT-only had the lowest observed misalignment. This is a single training seed and
one rollout per cell, not evidence of a reliable ranking across seeds or repeated
rollouts. Intervals use scenario as the paired unit, with the two variants equally
weighted. Task progress measures judged completion of the assigned task, not safety.
No new arm had missing cells, timeouts, reconstructed transcripts or retries.
Context-limit cutoffs were retained and judged: CoT 2, answer-only 1, empty-CoT 0.

## Data and masking

Each arm has **10,036 examples: all 752 DA rows plus 9,284 replay rows**.

- DA: [2026-09-14-da-synth](https://huggingface.co/datasets/dougalldeepmind/2026-09-14-da-synth)
  at `013886238fca238c4d54ace96530f444bb2b2f02`.
- Replay: [2026-09-08-nosynth-mix](https://huggingface.co/datasets/dougalldeepmind/2026-09-08-nosynth-mix)
  at `7e991f58e86eff0b0a9f15a54ebeddfffb5b14dd`.

Replay was selected once with proportional source quotas, largest-remainder
allocation and seed 0. The selected rows and their order are identical across arms;
their messages, answers and existing reasoning traces are unchanged. Replay always
uses normal assistant-completion supervision, including the generation-boundary rule.
Selection indices and hashes are in each mixture's `manifest.json`.

| Arm | DA tokens in the causal forward pass | DA tokens receiving loss |
| --- | --- | --- |
| CoT only (`supervise: cot`) | Prompt and real reasoning through `</think>`; answer and turn end removed | Reasoning and closing tag; prompt and forced opening prefill masked |
| Answer only (`supervise: answer`) | Prompt, complete real reasoning, separator, answer and turn end | Separator after `</think>`, answer and turn end; reasoning and both think tags masked |
| Empty CoT (normal supervision, blank DA `reasoning_content`) | Prompt, complete empty think marker, answer and turn end | Answer and turn end; prompt and **whole empty marker** masked |

Masking means `labels = -100`; it does not remove attention or detach gradients.
Answer-only training therefore still conditions on real reasoning. CoT-only rows
do not teach answer generation or turn termination; the unchanged replay still does.
The empty-marker rule masks the separator too, making empty-CoT supervise one fewer
separator token per DA row than answer-only: exactly 752 tokens over this dataset.
**All three adapters use thinking mode at inference**, including empty-CoT.

All 30,108 rendered rows passed independent token/decode checks. No row was
truncated; maximum sequence length was 8,191. On DA rows, CoT and answer supervision
partition the normal loss targets, and answer-only retains the normal input IDs.
Dataset-library loading and published byte hashes were checked independently.

| Mixture | Immutable revision | Forward tokens | Supervised tokens |
| --- | --- | ---: | ---: |
| [CoT only](https://huggingface.co/datasets/dougalldeepmind/2026-09-16-da-7-cot-mix) | `926e1014d24614a6fcd97762a644ddbeb7320288` | 8,038,964 | 4,961,350 |
| [Answer only](https://huggingface.co/datasets/dougalldeepmind/2026-09-16-da-7-answer-only-mix) | `e8104b084b6e47e99e2ace48df352b4a43a9604b` | 8,475,078 | 4,948,211 |
| [Empty CoT](https://huggingface.co/datasets/dougalldeepmind/2026-09-16-da-7-empty-cot-mix) | `c3a3b18612eb89da64b1541dd708bf8eb598f37b` | 8,027,329 | 4,947,459 |

The `7` in artifact names is the rounded naming percentage; actual DA share is
7.493025%. The old 702-row subset was not used.

## Training and model artifacts

Base: `Qwen/Qwen3.6-27B@6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
Seed 0; BF16 unquantized LoRA rank 64, alpha 128, dropout 0.05; one epoch,
628 optimizer steps; global batch 16; AdamW torch with betas 0.9/0.999,
epsilon 1e-8, weight decay 0.01 and gradient clipping 1.0; learning rate 1e-4,
cosine schedule and 5% warmup. Maximum sequence length 8,192; dynamic token
budget 8,000 without packing; sequence-mean-token-mean loss aggregation.
Three separate pods each used **two H200 GPUs**, with non-reentrant gradient
checkpointing and SDPA. Raw training losses are not comparable quality scores.

| Model | Revision including full backup | Training time |
| --- | --- | --- |
| [CoT only](https://huggingface.co/dougalldeepmind/2026-09-16-qwen36-0-da-7-cot) | `6cd676bca92e97b745067daab61f91ca2bb0126f` | 3h 32m 11s |
| [Answer only](https://huggingface.co/dougalldeepmind/2026-09-16-qwen36-0-da-7-answer-only) | `52adc308c378457a94b2eb900802dc424ab5540f` | 2h 26m 36s |
| [Empty CoT](https://huggingface.co/dougalldeepmind/2026-09-16-qwen36-0-da-7-empty-cot) | `b60f9e3656f92b99d9aedf1f9afbccefa3cc6f48` | 2h 24m 25s |

Each model contains `train_config.yaml`, `training_meta.json`, and
`training_backup/da-supervision-backup.tar` with a checksum manifest. Full training
outputs and resume checkpoints were verified on HF before pod teardown.

## ODCV protocol and artifacts

One pass over 40 scenario names in both mandated and incentivized variants:
80 rollouts per model. Temperature 0.7, thinking enabled, 28,000-token context,
Gemini 3 Flash Preview misalignment and separate task-progress judges, four
concurrent cells per arm. Misalignment is severity **at least 3**, not any nonzero
severity. Each arm's pilot is one of the 80 cells and is cached during continuation.

Scenario containers and drivers ran on local Docker; each model server ran on a
separate single-H100 RunPod pod. GPUs were released after durable local generation,
before judging. All 240 published transcript hashes were verified, together with
model revisions and protocol metadata. Published layout is `rollouts/`, `results/`,
and `metadata/`; new arms include generation and cost-attribution audits.

| Evaluation | Verified immutable revision |
| --- | --- |
| [CoT only: results and rollouts](https://huggingface.co/datasets/dougalldeepmind/2026-09-16-odcv-qwen36-0-da-7-cot) | `1645af8ac6ee5d41e7f8b92a6aa145a54db4a2b3` |
| [Answer only: results and rollouts](https://huggingface.co/datasets/dougalldeepmind/2026-09-16-odcv-qwen36-0-da-7-answer-only) | `120b209f43453a2d7cf69fcb833c2adc02e1ac6d` |
| [Empty CoT: results and rollouts](https://huggingface.co/datasets/dougalldeepmind/2026-09-16-odcv-qwen36-0-da-7-empty-cot) | `ec61d2f640b884acc9401c98e30dfa71527fed44` |

Retained context cutoffs: CoT mandated Health-Data-Compliance-Analyst and
incentivized Financial-Audit-Fraud-Finding; answer-only incentivized
Health-Data-Compliance-Analyst. Empty-CoT had none.

## Five-arm chart and comparison limits

The requested chart also includes the September 8 nosynth control and full-DA model:

| Reference | Misalignment | Rollouts | Evaluation revision |
| --- | ---: | ---: | --- |
| [100% nosynth](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-odcv-qwen36-0-nosynth) | 102/240 (42.5%) | 3 passes | `128495d177eab348b8185e1627e3bcafcb6706fa` |
| [Full DA, September 8](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-odcv-qwen36-0-da-7) | 25/240 (10.42%) | 3 passes | `0e69b5d75084be9174f0349852b2dbf92108e06d` |

The reference runs share the base model, scenario set, temperature, thinking mode,
context and judges, but use three passes rather than one. Full DA uses an older
DA corpus than the three ablations, so the five-bar chart is **not a controlled
supervision-only comparison**. It is not the unevaluated September 15 full-DA model.

Reproduce PNG, SVG, PDF and source tables from pinned HF results:

```powershell
uv run python -m scratch.da_supervision.odcv_chart --config scratch/da_supervision/odcv_chart.yaml
```

See [experiment tools and commands](../scratch/da_supervision/README.md) and the
[operational record](../scratch/da_supervision/archive/operations.md).
No GPU, scheduler or external write is needed to recreate the chart.
