# ABOUTME: One-off: package the Table2(7,999)+self-reflection(2,000) training bundle
# ABOUTME: (code.tar.gz + mixture + configs + card) and push to HF for the RunPod trainer.

"""Run AFTER the mixture is built:
    uv run python scratch/publish_selfreflect_bundle.py <mixture_run_dir>

Publishes LASR-Callum/2026-08-06-qwen36-table2-80-selfreflect-20-10k-train with the exact
files scripts/gpu/runpod_train.py expects (code.tar.gz + mixture.jsonl), plus the mixture
stats/meta and a card carrying the repo's required dataset fields.
"""

import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

REPO = "LASR-Callum/2026-08-06-qwen36-table2-80-selfreflect-20-10k-train"
TRAIN_CONFIG = "configs/train/lora_qwen36_table2_selfreflect_r64.yaml"
MIX_CONFIG = "configs/data/mixture/qwen36_table2_selfreflect_20_80.yaml"
# Everything `python3 scripts/train/train_lora.py` imports, plus the configs.
CODE = [
    "pyproject.toml",
    "scripts/train/train_lora.py",
    "src/__init__.py",
    "src/utils.py",
    "src/train/__init__.py",
    "src/train/train_lora.py",
    "src/train/masking.py",
    "src/train/mask_gate.py",
    TRAIN_CONFIG,
    MIX_CONFIG,
]

CARD = """---
license: apache-2.0
task_categories: [text-generation]
tags: [sft, qwen3.6, lora, alignment, self-reflection, assistant-only-loss]
---

# Qwen3.6 Table2 80% + SynthDoc self-reflection 20% — 10k-example training bundle

| field | value |
|---|---|
| `experiment` | One-epoch Qwen3.6-27B assistant-only LoRA SFT (r64): Matthew's exact 7,999 Table-2 rows + 2,000 first-person self-reflection records — the self-reflection twin of `LASR-Callum/qwen3.6-27b-lora-table2-synthdoc-r64`, differing ONLY in the 20% slice (difficult-advice -> self-reflection). |
| `date_generated` | 2026-08-06 (mixture; Table-2 rows verbatim from the 2026-08-04 arm, self-reflection corpus 2026-08-03 + 2026-08-06 top-up) |
| `constitution` | `constitutions/claude_distilled_12_principles_mid/constitution.md` in `source_repo` (the self-reflection corpus's target; since 2026-08-05 byte-identical to the 9-principle generation-time snapshot). Table-2 rows connect to none. |
| `source_repo` | `teaching_claude_why_replication` @ `{git_sha}` (branch `model-eval-model-data-gen`, uncommitted mixture/train configs included in `code.tar.gz`) |
| `models` | Base `Qwen/Qwen3.6-27B`. Corpus generators: `anthropic/claude-haiku-4.5` + `anthropic/claude-sonnet-5` via OpenRouter (see corpus repo). |
| `generation_config` | No generation in this step: deterministic seed-0 shuffle and exact example-count fill (7,999 + 2,000). |
| `schema` | `mixture.jsonl`: `text` = Qwen3.6 ChatML with think blocks preserved; `source` = `table2` or `self_reflection`; `n_tokens`. |
| `provenance` | `uv run python scripts/data/mixture/build_mixture.py --config {mix_config}`; Table-2 side extracted verbatim by `scratch/prep_table2_from_matthew.py` from `LASR-Callum/2026-08-04-table2-synthdoc-h200x4-train` (minus its 2,203 difficult-advice rows); self-reflection side is `data/self_reflection_sft_all.jsonl` = `LASR-Callum/2026-08-03-synthdoc-self-reflection` (592) + pilot run 20260806_114324 (18) + top-up run {topup_run} (~1,410). |

## Mixture

{stats_block}

Mixture SHA-256: `{mix_sha}`.

## Training

`code.tar.gz` + `{train_config}`: BF16 LoRA r=64/alpha=128, 1 epoch, batch 1 x grad-accum 16
(global 16), lr 1e-4 cosine, 5% warmup, weight decay 0.01, max_seq_len 8192, assistant-only
loss with the generation-boundary think rule (empty markers masked, real traces supervised).
Launched by `scripts/gpu/runpod_train.py` on a credential-free pod; adapter pushed to
`LASR-Callum/qwen3.6-27b-lora-table2-selfreflect-r64` from the driver machine.
"""


def main(mix_run_dir: str, topup_run: str = "TBD") -> None:
    run = Path(mix_run_dir)
    mixture = run / "mixture.jsonl"
    stats = json.loads((run / "mixture_stats.json").read_text())
    assert mixture.exists()

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    mix_sha = hashlib.sha256(mixture.read_bytes()).hexdigest()

    api = HfApi()
    api.create_repo(REPO, repo_type="dataset", exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        tar_path = Path(td) / "code.tar.gz"
        with tarfile.open(tar_path, "w:gz") as t:
            for f in CODE:
                t.add(f, arcname=f)
        api.upload_file(path_or_fileobj=str(tar_path), path_in_repo="code.tar.gz",
                        repo_id=REPO, repo_type="dataset")

    api.upload_file(path_or_fileobj=str(mixture), path_in_repo="mixture.jsonl",
                    repo_id=REPO, repo_type="dataset")
    for name in ("mixture_stats.json", "run_meta.json"):
        p = run / name
        if p.exists():
            api.upload_file(path_or_fileobj=str(p), path_in_repo=name,
                            repo_id=REPO, repo_type="dataset")

    card = CARD.format(git_sha=sha, mix_sha=mix_sha, mix_config=MIX_CONFIG,
                       train_config=TRAIN_CONFIG, topup_run=topup_run,
                       stats_block="```json\n" + json.dumps(stats, indent=2) + "\n```")
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                    repo_id=REPO, repo_type="dataset")
    print(f"published https://huggingface.co/datasets/{REPO}")
    print(f"mixture sha256={mix_sha}")


if __name__ == "__main__":
    main(*sys.argv[1:])
