# ABOUTME: Assembles and publishes the tool-calling arm's Hugging Face repos: the training
# ABOUTME: mixture, the run record (logs/metrics/charts), and the LoRA adapter.

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

import fire
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import git_sha  # noqa: E402

MIXTURE_REPO = "LASR-Callum/2026-07-31-toolcalling-tulu-20-80-mixture"
RUN_REPO = "LASR-Callum/2026-07-31-toolcalling-tulu-sft-run"
ADAPTER_REPO = "LASR-Callum/qwen3.6-27b-toolcalling-tulu-lora-20-80"


def _api() -> HfApi:
    token = os.environ.get("HF_TOKEN")
    assert token, "HF_TOKEN is not set; publishing needs it"
    return HfApi(token=token)


def _push(folder: Path, repo: str, repo_type: str, message: str) -> str:
    """Create (if needed) and upload a folder. Everything here is public by house rule."""
    api = _api()
    api.create_repo(repo, repo_type=repo_type, private=False, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=repo, repo_type=repo_type,
                      commit_message=message)
    kind = "datasets/" if repo_type == "dataset" else ""
    url = f"https://huggingface.co/{kind}{repo}"
    print(f">>> uploaded {folder} -> {url}")
    for f in sorted(api.list_repo_files(repo, repo_type=repo_type)):
        print("     ", f)
    return url


def _mixture_card(stats: dict, sha: str) -> str:
    """The dataset card, carrying every field AGENTS.md requires."""
    ag, tulu, total = stats["agentic_toolcalling"], stats["tulu3"], stats["total"]
    return f"""---
license: odc-by
task_categories:
  - text-generation
language:
  - en
tags:
  - sft
  - qwen3.6
  - tool-use
  - agentic
  - tulu3
  - replay
  - constitution
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files:
      - split: train
        path: mixture.jsonl
---

# Tool-calling + TULU3 replay SFT mixture (20/80) for Qwen3.6-27B

The training mixture behind
[`{ADAPTER_REPO}`](https://huggingface.co/{ADAPTER_REPO}): **{total['tokens']:,} Qwen3.6
tokens** across **{total['examples']:,} pre-rendered conversations**, split
**{stats['agentic_share_pct']}% agentic tool-use / {stats['tulu3_share_pct']}% TULU3 replay**.

| Source | Examples | Tokens | Share |
|---|---:|---:|---:|
| agentic tool-use ({ag['docs_emitting_tool_calls']} of them emit `<tool_call>`, {ag['tool_call_spans']} spans total) | {ag['examples']} | {ag['tokens']:,} | {stats['agentic_share_pct']}% |
| TULU3 replay | {tulu['examples']:,} | {tulu['tokens']:,} | {stats['tulu3_share_pct']}% |
| **Total** | **{total['examples']:,}** | **{total['tokens']:,}** | |

## Required metadata

| field | value |
| --- | --- |
| `experiment` | The tool-calling arm of the Qwen3.6-27B constitution mixture family: hold total tokens and the 20% target share fixed, and make the *composition* of that 20% pure agentic tool-use data. Sits alongside the difficult-advice 20/80 arm and the equal three-way arm as the missing single-composition cell. |
| `date_generated` | 2026-07-31 |
| `constitution` | Claude approved constitution (7 principles plus a priority order), spec id `claude_approved_constitution`. The 20% target data is the `approved_agentic` sub-corpus generated against it — the one where the model itself is the actor holding live tools. Tracked at `experiments/teaching-claude-why/docs/claude_approved_constitution.md`. |
| `source_repo` | https://github.com/Matthew-Bozoukov/teaching_claude_why_replication @ `{sha}` |
| `models` | Tokenised and rendered for `Qwen/Qwen3.6-27B`. No model was called to build this mixture — both sources were already generated and published. |
| `generation_config` | `seed: 0`, greedy token-budget fill, `max_seq_len: 4096`. No sampling from any model. |
| `schema` | `text` (str) — the conversation already rendered by Qwen3.6's chat template, ready for SFT on a `text` field. `source` (str) — `agentic_toolcalling` or `tulu3`. |
| `provenance` | `uv run python src/experiments/build_toolcalling_mixture.py --config configs/mixture_toolcalling_qwen36.yaml` |

## Why the rows are pre-rendered strings, not `messages`

The think-block convention differs per source and has to be fixed at build time. Qwen3.6's
template renders `<think>{{reasoning}}</think>` for any final assistant turn, so trace-free data
would render an **empty** `<think></think>` — the documented pattern that trains a model to stop
reasoning. Re-rendering these rows from `messages` will **not** reproduce the training data.

| Data | Renders as |
|---|---|
| agentic turn with a source reasoning trace | `<think>real reasoning</think>` |
| agentic turn without one | no think block (the `fullthink` variant strips empties) |
| TULU3 replay | no think block at all |

Verified on the written artifact: **0** empty `<think></think>`, **0** TULU3 rows carrying any
think block, all {ag['tool_call_spans']} `<tool_call>` spans balanced and well-formed, 0 duplicate
rows, and every row ends on an assistant turn.

## Sources

- **80%** — [`LASR-Callum/tulu3-replay-80pct-qwen3.6-27b`](https://huggingface.co/datasets/LASR-Callum/tulu3-replay-80pct-qwen3.6-27b), taken whole ({tulu['examples']:,} rows / {tulu['tokens']:,} tok). This is the exact replay half of the shipped 20/80 difficult-advice arm, so the replay side is held constant across the family.
- **20%** — [`LASR-Callum/2026-07-29-synthdoc-approved-constitution-sft`](https://huggingface.co/datasets/LASR-Callum/2026-07-29-synthdoc-approved-constitution-sft) `runs/approved_agentic/sft_qwen36_fullthink.jsonl`, sampled seed-0 to an exact 20% token budget ({ag['examples']} of 151 docs).

## Why `max_seq_len` is 4096, not 2048

The sibling arms train at 2048. These agentic conversations run 9–13 turns with a median of 2,348
tokens, and **99 of the 151 source documents exceed 2048**. Measured: a 2048 cap keeps only 80.4%
of the corpus and severs **11 of its 98 `<tool_call>` spans**, all inside the long conversations
the tool calls actually live in. At 4096 this mixture is truncated **nowhere at all** — no row
exceeds the cap. The cost is that this arm differs from its siblings on one hyperparameter as well
as on composition.

## Known caveat

Only **30 of the {ag['examples']} agentic rows (24%) carry a real reasoning trace.** The
difficult-advice 20/80 arm had a trace on *every* target example. So this arm is markedly less
reasoning-dense, and if the dose-response in this family is driven by reasoning rather than topic
coverage, that is confounded with the composition change here. This is inherent to the source
corpus, not an artifact of the mixing.
"""


def mixture(
    run_dir: str,
    out: str = "exports/2026-07-31-toolcalling-tulu-20-80-mixture",
    push: bool = True,
) -> None:
    """Assemble and publish the training-mixture dataset repo.

    Args:
        run_dir: The build_toolcalling_mixture.py output directory.
        out: Export directory to assemble.
        push: Upload to Hugging Face. False assembles locally only.
    """
    src = Path(run_dir)
    dst = Path(out)
    dst.mkdir(parents=True, exist_ok=True)
    stats = json.loads((src / "mixture_stats.json").read_text())

    shutil.copy2(src / "mixture.jsonl", dst / "mixture.jsonl")
    shutil.copy2(src / "mixture_stats.json", dst / "mixture_stats.json")
    shutil.copy2(src / "run_meta.json", dst / "run_meta.json")
    (dst / "README.md").write_text(_mixture_card(stats, git_sha()[:7]), encoding="utf-8")

    print(json.dumps(stats, indent=2))
    if push:
        _push(dst, MIXTURE_REPO, "dataset",
              f"Tool-calling + TULU3 20/80 SFT mixture: {stats['total']['tokens']:,} tokens")


def _adapter_card(stats: dict, summary: dict, sha: str) -> str:
    """The model card for the LoRA adapter, matching the sibling arms' shape."""
    ag, tulu, total = stats["agentic_toolcalling"], stats["tulu3"], stats["total"]
    return f"""---
base_model: Qwen/Qwen3.6-27B
library_name: peft
pipeline_tag: image-text-to-text
tags:
  - lora
  - peft
  - sft
  - trl
  - alignment
  - agentic-misalignment
  - tool-use
datasets:
  - LASR-Callum/2026-07-31-toolcalling-tulu-20-80-mixture
  - LASR-Callum/2026-07-29-synthdoc-approved-constitution-sft
  - LASR-Callum/tulu3-replay-80pct-qwen3.6-27b
license: apache-2.0
---

# Qwen3.6-27B — tool-calling + TULU3 LoRA (**20/80** mixture)

LoRA adapter for [`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B). The 20% target
portion is **entirely agentic tool-use data** — conversations where the model itself is the actor
holding live tools — and the other 80% is TULU3 replay.

This is the pure-tool-calling cell of a family that holds total tokens and the 20% target share
fixed and varies only the **composition** of that 20%:

| Arm | 20% composition | ODCV-Bench | Agentic-misalignment |
|---|---|---|---|
| [base (no SFT)](https://huggingface.co/Qwen/Qwen3.6-27B) | — | 37.2% | 65.5% |
| [`…-difficult-advice-tulu-lora-20-80`](https://huggingface.co/LASR-Callum/qwen3.6-27b-difficult-advice-tulu-lora-20-80) | difficult-advice only | 19.2% | 25.3% |
| [`…-threeway-constitution-lora`](https://huggingface.co/LASR-Callum/qwen3.6-27b-threeway-constitution-lora) | equal thirds: embodied / difficult-advice / agentic | not yet run | not yet run |
| **this** | **agentic tool-use only** | **not yet run** | **not yet run** |
| [`…-tulu-100pct-lora`](https://huggingface.co/LASR-Callum/qwen3.6-27b-tulu-100pct-lora) | none (zero-dose control) | — | — |

## Training mixture

Published in full as
[`LASR-Callum/2026-07-31-toolcalling-tulu-20-80-mixture`](https://huggingface.co/datasets/LASR-Callum/2026-07-31-toolcalling-tulu-20-80-mixture).

| Source | Examples | Tokens | Share | Rendering |
|---|---:|---:|---:|---|
| agentic tool-use (`approved_agentic`, `fullthink`) | {ag['examples']} | {ag['tokens']:,} | {stats['agentic_share_pct']}% | reasoning kept where the source had it; **no** empty think blocks |
| TULU3 replay | {tulu['examples']:,} | {tulu['tokens']:,} | {stats['tulu3_share_pct']}% | **no** `<think>` block at all |
| **Total** | **{total['examples']:,}** | **{total['tokens']:,}** | | |

{ag['docs_emitting_tool_calls']} of the {ag['examples']} agentic documents actually emit tool calls
— {ag['tool_call_spans']} `<tool_call>` spans in Qwen3.6's XML dialect, all verified balanced.

## Training

bf16 LoRA (not QLoRA — bitsandbytes does not reliably cover this model's hybrid
linear-attention/SSM layers), 1×H100 80GB SXM, **{summary.get('train_runtime_hms', '?')}**.

| | |
|---|---|
| r / alpha / dropout | 32 / 64 / 0.05 |
| target modules | regex scoped to `model.language_model.*` (q/k/v/o/gate/up/down proj) |
| epochs / steps | 1 / {summary.get('max_steps', '?')} |
| batch x grad-accum | 1 x 16 |
| lr / schedule | 1e-4, cosine, 3% warmup, annealed to 0 |
| max seq len / packing | **4096** / off |
| `assistant_only_loss` | false |
| seed | 0 |
| trainable params | {summary.get('trainable_params_str', '?')} |

**Loss:** {summary.get('loss_first', '?')} → **{summary.get('train_loss_epoch_avg', '?')}**
(epoch average); the last logged step ({summary.get('last_logged_step', '?')}) read
{summary.get('loss_final', '?')}. Epoch-average mean token accuracy
**{summary.get('acc_epoch_avg', '?')}**, final grad_norm {summary.get('grad_norm_final', '?')},
{summary.get('num_tokens', 0):,} tokens consumed.

Curves and the full log history are in
[`LASR-Callum/2026-07-31-toolcalling-tulu-sft-run`](https://huggingface.co/datasets/LASR-Callum/2026-07-31-toolcalling-tulu-sft-run).

### Why `max_seq_len` is 4096 and the sibling arms use 2048

These agentic conversations run 9–13 turns with a median of 2,348 tokens, and 99 of the 151 source
documents exceed 2048. Measured: a 2048 cap keeps only 80.4% of the corpus and **severs 11 of its
98 `<tool_call>` spans**, inside exactly the long conversations the tool calls live in. At 4096 the
mixture is truncated nowhere at all. The cost is that this arm differs from its siblings on one
hyperparameter as well as on composition — read the head-to-head with that caveat.

## Known caveats

1. **Reasoning density.** Only **30 of the {ag['examples']} agentic rows (24%) carry a real
   reasoning trace**, against every target example in the difficult-advice 20/80 arm. If the
   dose-response in this family is driven by reasoning rather than topic coverage, that is
   confounded with the composition change here. Inherent to the source corpus.
2. **`target_modules` only half-applies.** Qwen3.6-27B is hybrid: `q/k/v/o_proj` exist in 16 of 64
   layers, the rest being linear-attention blocks with different module names. `gate/up/down_proj`
   attach to all 64. So this adapter tunes MLP throughout but attention in only a quarter of the
   stack. Same for every arm in the family, so comparisons are unaffected.
3. **Not yet evaluated.** No ODCV-Bench or agentic-misalignment number exists for this arm yet.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.6-27B", dtype="bfloat16")
model = PeftModel.from_pretrained(model, "{ADAPTER_REPO}")
model = model.merge_and_unload()  # vLLM LoRA support for this hybrid arch is unproven
```

Use `AutoModelForImageTextToText`, not `AutoModelForCausalLM` — this is a vision-language
checkpoint. Merging drops the base model's 15 `mtp.*` tensors, so speculative decoding needs them
grafted back.

## Provenance

Repository https://github.com/Matthew-Bozoukov/teaching_claude_why_replication @ `{sha}`.
Built with `src/experiments/build_toolcalling_mixture.py`, trained with
`src/experiments/train_lora.py --config configs/train_lora_toolcalling.yaml`.
"""


def _run_card(stats: dict, summary: dict, sha: str, cost: dict) -> str:
    """The dataset card for the run record: logs, metrics and figures."""
    return f"""---
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

The adapter is at [`{ADAPTER_REPO}`](https://huggingface.co/{ADAPTER_REPO}); the training data is
at [`{MIXTURE_REPO}`](https://huggingface.co/datasets/{MIXTURE_REPO}).

## Required metadata

| field | value |
| --- | --- |
| `experiment` | One bf16 LoRA SFT run of Qwen3.6-27B on a 20% agentic tool-use / 80% TULU3 replay mixture, forming the pure-tool-calling cell of the constitution mixture family. |
| `date_generated` | 2026-07-31 |
| `constitution` | Claude approved constitution (7 principles plus a priority order), spec id `claude_approved_constitution`. The 20% target data is its `approved_agentic` sub-corpus. |
| `source_repo` | https://github.com/Matthew-Bozoukov/teaching_claude_why_replication @ `{sha}` |
| `models` | Base `Qwen/Qwen3.6-27B`, loaded as `Qwen3_5ForConditionalGeneration` via `AutoModelForImageTextToText`. No API model was called. |
| `generation_config` | Not applicable - this is a training run, not a generation run. Training hyperparameters are in `artifacts/resolved_config.json` and the table below. |
| `schema` | See "Layout" below. |
| `provenance` | `python src/experiments/train_lora.py --config configs/train_lora_toolcalling.yaml` on a 1xH100 80GB SXM RunPod pod. |

## Result

| | |
|---|---|
| steps / epochs | {summary.get('max_steps', '?')} / 1 |
| wall clock | {summary.get('train_runtime_hms', '?')} |
| tokens consumed | {summary.get('num_tokens', 0):,} |
| loss | {summary.get('loss_first', '?')} → {summary.get('train_loss_epoch_avg', '?')} (epoch avg) |
| loss at last logged step ({summary.get('last_logged_step', '?')}) | {summary.get('loss_final', '?')} |
| mean token accuracy | {summary.get('acc_epoch_avg', '?')} (epoch avg), {summary.get('acc_final', '?')} at last step |
| final grad norm | {summary.get('grad_norm_final', '?')} |
| trainable params | {summary.get('trainable_params_str', '?')} |
| GPU | {cost.get('gpu', '?')} at ${cost.get('hourly_usd', '?')}/h |
| GPU cost | **${cost.get('estimated_cost_usd', '?')}** over {cost.get('elapsed_hours', '?')} h |

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
"""


def _trainable_params(artifacts: Path) -> str | None:
    """Read peft's one-off 'trainable params: N || ...' line, which trainer_state does not carry."""
    p = artifacts / "trainable_params.txt"
    if not p.exists():
        return None
    txt = p.read_text(encoding="utf-8").strip()
    if not txt:
        return None
    n = txt.split("trainable params:")[-1].split("||")[0].strip().replace(",", "")
    return f"{int(n) / 1e6:.1f}M" if n.isdigit() else txt


def _final_summary_from_log(log_path: Path) -> dict:
    """Recover TRL's end-of-run summary, which it prints but does not persist.

    `save_strategy="epoch"` writes checkpoint-N/trainer_state.json *before* the final
    train summary is emitted, so `train_runtime`, the epoch-average `train_loss` and the
    epoch-average `mean_token_accuracy` exist only on stdout. Parsed rather than dropped,
    because they are the numbers the sibling model cards quote.
    """
    if not log_path.exists():
        return {}
    line = None
    for ln in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "train_runtime" in ln and "train_loss" in ln:
            line = ln
    if not line:
        return {}
    out = {}
    for key in ("train_runtime", "train_loss", "mean_token_accuracy", "num_tokens", "entropy"):
        m = re.search(rf"'{key}':\s*'([^']+)'", line)
        if m:
            try:
                out[key] = float(m.group(1))
            except ValueError:
                pass
    return out


def _summarise(state: dict, log_path: Path | None = None) -> dict:
    """Pull the headline numbers out of TRL's trainer_state.json, plus the stdout summary."""
    hist = state.get("log_history", [])
    pts = [h for h in hist if "loss" in h]
    final = next((h for h in reversed(hist) if "train_runtime" in h), {})
    tail = _final_summary_from_log(log_path) if log_path else {}

    runtime = final.get("train_runtime") or tail.get("train_runtime")
    hms = None
    if runtime:
        h, rem = divmod(int(runtime), 3600)
        m, s = divmod(rem, 60)
        hms = f"{h}h{m:02d}m{s:02d}s"

    # num_input_tokens_seen is 0 unless include_num_input_tokens_seen was set; the per-step
    # cumulative `num_tokens` on the last logged step is the real figure.
    tokens = (state.get("num_input_tokens_seen")
              or (pts[-1].get("num_tokens") if pts else None)
              or final.get("num_tokens") or 0)

    return {
        "max_steps": state.get("max_steps") or (pts[-1]["step"] if pts else None),
        "num_tokens": int(tokens),
        "train_runtime_s": runtime,
        "train_runtime_hms": hms,
        "loss_first": round(pts[0]["loss"], 4) if pts else None,
        # Epoch averages from the stdout summary - what the sibling cards quote.
        "train_loss_epoch_avg": tail.get("train_loss"),
        "acc_epoch_avg": tail.get("mean_token_accuracy"),
        # Last logged optimizer step, for the shape of the curve's tail.
        "loss_final": round(pts[-1]["loss"], 4) if pts else None,
        "acc_final": round(pts[-1]["mean_token_accuracy"], 4) if pts and "mean_token_accuracy" in pts[-1] else None,
        "grad_norm_final": round(pts[-1]["grad_norm"], 4) if pts and "grad_norm" in pts[-1] else None,
        "last_logged_step": pts[-1]["step"] if pts else None,
    }


def run(
    artifacts_dir: str,
    mixture_stats: str,
    teardown_evidence: str = "",
    out: str = "exports/2026-07-31-toolcalling-tulu-sft-run",
    push: bool = True,
) -> None:
    """Assemble and publish the run-record dataset repo.

    Args:
        artifacts_dir: Directory holding the files pulled off the GPU box.
        mixture_stats: Path to the mixture's stats json.
        teardown_evidence: Optional teardown evidence json, for the cost table.
        out: Export directory to assemble.
        push: Upload to Hugging Face.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from plot_scripts.plot_toolcalling_training import (  # noqa: E402
        plot_training_curves,
        write_markdown_mirror,
    )

    src, dst = Path(artifacts_dir), Path(out)
    for sub in ("artifacts", "assets", "results"):
        (dst / sub).mkdir(parents=True, exist_ok=True)

    state = json.loads((src / "trainer_state.json").read_text(encoding="utf-8"))
    summary = _summarise(state, src / "training.log")
    stats = json.loads(Path(mixture_stats).read_text(encoding="utf-8"))
    cost = json.loads(Path(teardown_evidence).read_text(encoding="utf-8-sig")) if teardown_evidence else {}

    for name in ("trainer_state.json", "run_meta.json", "adapter_config.json",
                 "environment.txt", "training.log", "resolved_config.json"):
        p = src / name
        if p.exists():
            shutil.copy2(p, dst / "artifacts" / name)

    summary["trainable_params_str"] = (
        _trainable_params(src) or (f"{state['trainable_params'] / 1e6:.1f}M"
                                   if state.get("trainable_params") else None)
    )

    plot_training_curves(state["log_history"], dst / "assets" / "training_curves.png",
                         "Qwen3.6-27B — tool-calling 20/80 SFT")
    write_markdown_mirror(state["log_history"], dst / "results" / "training_curve.md", summary)
    (dst / "results" / "training_summary.json").write_text(
        json.dumps({**summary, "cost": cost}, indent=2), encoding="utf-8")
    shutil.copy2(mixture_stats, dst / "results" / "mixture_stats.json")
    (dst / "README.md").write_text(_run_card(stats, summary, git_sha()[:7], cost), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if push:
        _push(dst, RUN_REPO, "dataset",
              f"Tool-calling 20/80 SFT run record: {summary['max_steps']} steps, "
              f"loss {summary['loss_first']} -> {summary['loss_final']}")


def adapter(
    adapter_dir: str,
    artifacts_dir: str,
    mixture_stats: str,
    push: bool = True,
) -> None:
    """Write the model card into a pulled adapter directory and publish it.

    Args:
        adapter_dir: The adapter directory pulled off the GPU box.
        artifacts_dir: Directory holding trainer_state.json.
        mixture_stats: Path to the mixture's stats json.
        push: Upload to Hugging Face.
    """
    d = Path(adapter_dir)
    state = json.loads((Path(artifacts_dir) / "trainer_state.json").read_text(encoding="utf-8"))
    summary = _summarise(state, Path(artifacts_dir) / "training.log")
    summary["trainable_params_str"] = (
        _trainable_params(Path(artifacts_dir)) or (f"{state['trainable_params'] / 1e6:.1f}M"
                                                   if state.get("trainable_params") else None)
    )
    stats = json.loads(Path(mixture_stats).read_text(encoding="utf-8"))

    assert (d / "adapter_model.safetensors").exists(), f"no adapter weights in {d}"
    (d / "README.md").write_text(_adapter_card(stats, summary, git_sha()[:7]), encoding="utf-8")

    if push:
        _push(d, ADAPTER_REPO, "model",
              "Qwen3.6-27B tool-calling + TULU3 20/80 LoRA (1 epoch, seq 4096)")


if __name__ == "__main__":
    fire.Fire({"mixture": mixture, "run": run, "adapter": adapter})
