# ABOUTME: Package a training bundle (code.tar.gz + mixture jsonl + card) and push it to
# ABOUTME: HF, so scripts/gpu/runpod_train.py can train from a pod holding no credentials.

"""Publish one arm's training bundle.

Run: uv run python scratch/publish_train_bundle.py \
       --repo LASR-Callum/<date>-<arm>-bundle \
       --mixture data/<arm>.jsonl \
       --train_config configs/train/<arm>.yaml \
       --experiment "one sentence" [--extra_note "..."]

Generalises scratch/publish_selfreflect_bundle.py, which hardcoded one arm. The pod
carries no token: it downloads this repo anonymously, so the bundle MUST be public and
must contain everything `scripts/train/train_lora.py` imports. That import list is
derived here from the trainer's actual imports rather than remembered:

    train_lora -> src.train.{dynamic_batching, mask_gate, masking}
               -> src.model_profile
               -> src.huggingface (resolve_dataset, at run time)
               -> src.utils

Getting it wrong costs a pod boot, not a test failure — the pod installs deps for ~20
minutes and only then discovers the missing module.
"""

import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

import fire
from dotenv import load_dotenv
from huggingface_hub import HfApi

# Everything the trainer imports, plus the config it is pointed at.
CODE = [
    "pyproject.toml",
    "scripts/train/train_lora.py",
    "src/__init__.py",
    "src/utils.py",
    "src/huggingface.py",
    "src/model_profile.py",
    "src/train/__init__.py",
    "src/train/train_lora.py",
    "src/train/masking.py",
    "src/train/mask_gate.py",
    "src/train/dynamic_batching.py",
]


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip() or "unknown"


def main(repo: str, mixture: str, train_config: str, experiment: str,
         extra_note: str = "", private: bool = False) -> None:
    """Build and push a bundle.

    Args:
        repo: Target HF dataset repo id (dated, per the naming rule).
        mixture: Local path to the rendered mixture jsonl.
        train_config: Config the pod will train with; included in the tarball.
        experiment: One-sentence description for the card.
        extra_note: Appended to the card's notes row.
        private: MUST stay False for the credential-free pod to read it.
    """
    load_dotenv()
    mix = Path(mixture)
    cfg = Path(train_config)
    assert mix.exists(), f"missing mixture {mix}"
    assert cfg.exists(), f"missing config {cfg}"
    assert not private, ("the pod downloads this bundle anonymously — a private repo "
                         "would 401 after ~20 minutes of setup")

    files = [*CODE, str(cfg)]
    missing = [f for f in files if not Path(f).exists()]
    assert not missing, f"bundle would omit files the trainer imports: {missing}"

    sha = _git_sha()
    with tempfile.TemporaryDirectory() as td:
        tar_path = Path(td) / "code.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            for f in files:
                tar.add(f, arcname=f)
        api = HfApi()
        api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)

        stats = Path(str(mix) + ".stats.json")
        card = (
            "---\nlicense: apache-2.0\ntask_categories: [text-generation]\n"
            "tags: [sft, qwen3.6, lora, generator-ablation]\n---\n\n"
            f"# {repo.split('/')[-1]}\n\n"
            "| field | value |\n| --- | --- |\n"
            f"| `experiment` | {experiment} |\n"
            f"| `source_repo` | Matthew-Bozoukov/Lessons_from_constituitional_AFT @ {sha} |\n"
            f"| `train_config` | `{cfg}` (inside code.tar.gz) |\n"
            f"| `mixture` | `{mix.name}` |\n"
            f"| `provenance` | uv run python scripts/gpu/runpod_train.py up "
            f"--bundle {repo} --train_config {cfg} --mixture {mix.name} "
            f"--gpu 'NVIDIA H200' --gpu_count 2 |\n"
            f"| `notes` | Training bundle, not a corpus: code.tar.gz + the rendered "
            f"mixture. Public because the training pod holds no credentials and reads it "
            f"anonymously. {extra_note} |\n")
        Path(td, "README.md").write_text(card)

        uploads = [(tar_path, "code.tar.gz"), (Path(td, "README.md"), "README.md"),
                   (mix, mix.name)]
        if stats.exists():
            uploads.append((stats, stats.name))
        for local, name in uploads:
            api.upload_file(path_or_fileobj=str(local), path_in_repo=name,
                            repo_id=repo, repo_type="dataset")
            print(f"  uploaded {name}")
    print(f"bundle -> https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    fire.Fire(main)
