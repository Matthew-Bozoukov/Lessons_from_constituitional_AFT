---
license: apache-2.0
tags:
  - research-log
  - experiment-date-2026-07-31
  - sft
  - training-run
  - qwen3.6
---

# Run record — Qwen3.6-27B tool-calling 20/80 SFT

Everything the training run produced except the weights: the TRL log history, the resolved
config, the environment, the loss/accuracy figure and its greppable markdown mirror.

The adapter is at [`LASR-Callum/qwen3.6-27b-toolcalling-tulu-lora-20-80`](https://huggingface.co/LASR-Callum/qwen3.6-27b-toolcalling-tulu-lora-20-80); the training data is
at [`LASR-Callum/2026-07-31-toolcalling-tulu-20-80-mixture`](https://huggingface.co/datasets/LASR-Callum/2026-07-31-toolcalling-tulu-20-80-mixture).

## Required metadata

| field | value |
| --- | --- |
| `experiment` | One bf16 LoRA SFT run of Qwen3.6-27B on a 20% agentic tool-use / 80% TULU3 replay mixture, forming the pure-tool-calling cell of the constitution mixture family. |
| `date_generated` | 2026-07-31 |
| `constitution` | Claude approved constitution (7 principles plus a priority order), spec id `claude_approved_constitution`. The 20% target data is its `approved_agentic` sub-corpus. |
| `source_repo` | https://github.com/Matthew-Bozoukov/teaching_claude_why_replication @ `639d85c` |
| `models` | Base `Qwen/Qwen3.6-27B`, loaded as `Qwen3_5ForConditionalGeneration` via `AutoModelForImageTextToText`. No API model was called. |
| `generation_config` | Not applicable - this is a training run, not a generation run. Training hyperparameters are in `artifacts/resolved_config.json` and the table below. |
| `schema` | See "Layout" below. |
| `provenance` | `python src/experiments/train_lora.py --config configs/train_lora_toolcalling.yaml` on a 1xH100 80GB SXM RunPod pod. |

## Result

| | |
|---|---|
| steps / epochs | 126 / 1 |
| wall clock | 1h38m09s |
| tokens consumed | 1,492,498 |
| loss | 2.7528 → 1.057 (epoch avg) |
| loss at last logged step (125) | 1.0199 |
| mean token accuracy | 0.7071 (epoch avg), 0.733 at last step |
| final grad norm | 0.3573 |
| trainable params | 159.4M |
| GPU | NVIDIA H100 80GB HBM3 at $2.99/h |
| GPU cost | **$5.73** over 1.918 h |

![training curves](assets/training_curves.png)

## Layout

| path | what |
| --- | --- |
| `artifacts/trainer_state.json` | TRL's full `log_history` - loss, token accuracy, grad norm, lr per logging step |
| `artifacts/resolved_config.json` | the training config as the trainer actually resolved it |
| `artifacts/run_meta.json` | git SHA, base model, data path, example count |
| `artifacts/adapter_config.json` | the peft config the adapter was saved with |
| `artifacts/environment.txt` | package versions and GPU as reported on the box |
| `artifacts/training.log` | full stdout/stderr of the run |
| `assets/training_curves.png` | loss and mean token accuracy against step |
| `results/training_curve.md` | the same numbers as a greppable table |
| `results/training_summary.json` | the headline numbers above, machine-readable |
| `results/mixture_stats.json` | composition of the data this run consumed |

## Caveat carried from the data

Only 24% of the agentic rows carry a real reasoning trace, against 100% in the difficult-advice
20/80 arm. Reasoning density is therefore confounded with composition in any head-to-head against
that arm. Recorded here rather than left for a reader to rediscover.
