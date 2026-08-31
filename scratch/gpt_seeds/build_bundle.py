# ABOUTME: Build + publish the GPT seed-replicate training bundle: seed 0's own code.tar.gz with the
# ABOUTME: two seed configs added, beside the byte-identical mixture, in a new public HF dataset repo.
# Run: uv run python scratch/gpt_seeds/build_bundle.py   (after scratch/gpt_seeds/fetch_seed0_bundle.py)
"""Why not scratch/publish_train_bundle.py: that tars the WORKING TREE's trainer. The seed
replicates must run the code seed 0 ran, so this takes seed 0's tarball verbatim (pulled at
the revision its stamp pins) and only appends the two seed configs. The mixture is
re-uploaded unchanged (sha256 asserted) because scripts/gpu/runpod_train.py reads code and
data from ONE repo.
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

import fire
import yaml
from dotenv import load_dotenv
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
SEED0_CFG = (
    "configs/train/2026-08-24_lora_qwen36_table2_9284_gpt_responder_685_paired.yaml"
)
# NOT PRESENT ON THIS BRANCH. The GPT seed-replicate run finished on 2026-08-29 and its two
# seed configs were never merged to main; they live on `worktree-gpt-seeds`. This driver is
# kept as the record of how that bundle was built -- check the configs out from that branch
# before re-running it, or it will fail at the assert below rather than silently skip them.
SEED_CFGS = [
    "configs/train/lora_qwen36_t2_9284_gptresp685_paired_s42_2xh200.yaml",
    "configs/train/lora_qwen36_t2_9284_gptresp685_paired_s69_2xh200.yaml",
]
MIXTURE = "t2_9284_gptresp685_10k.jsonl"
REPO = "LASR-Callum/2026-08-28-gpt-responder-685-seeds-bundle"
SEED0_BUNDLE = "LASR-Callum/2026-08-25-gpt-responder-685-paired-bundle"
SEED0_ADAPTER = "LASR-Callum/qwen3.6-27b-lora-t2-9284-gptresp685-paired-r64"


def _assert_only_seed_differs(seed0: Path, cfg: Path) -> None:
    """The replicate may differ from seed 0 in seed / output_dir / hub_model_id, nothing else."""
    a = yaml.safe_load(seed0.read_text())
    b = yaml.safe_load(cfg.read_text())
    for k in ("output_dir",):
        a.pop(k), b.pop(k)
    a["train"].pop("hub_model_id"), b["train"].pop("hub_model_id")
    sa, sb = a.pop("seed"), b.pop("seed")
    assert sa == 0 and sb in (42, 69), (sa, sb)
    assert a == b, f"{cfg.name} differs from seed 0 beyond seed/output_dir/hub_model_id"


def main(
    src: str = "/Users/kunwar/.claude/jobs/7e2b4dd7/tmp/seed0_bundle", repo: str = REPO
) -> None:
    load_dotenv(ROOT / ".env")
    src_dir = Path(src)
    hashes = json.loads((src_dir / "hashes.json").read_text())
    seed0_cfg = ROOT / SEED0_CFG
    for c in SEED_CFGS:
        _assert_only_seed_differs(seed0_cfg, ROOT / c)
    print("seed configs differ from seed 0 only in seed/output_dir/hub_model_id: OK")

    # Seed 0's tarball, verbatim, plus the seed configs. The bundled seed-0 config must equal
    # the working tree's -- otherwise "same code as seed 0" is not what the tree documents.
    with tarfile.open(src_dir / "code.tar.gz") as t:
        members = [
            (m, t.extractfile(m).read() if m.isfile() else None) for m in t.getmembers()
        ]
    bundled_cfg = next(b for m, b in members if m.name == SEED0_CFG)
    assert bundled_cfg == seed0_cfg.read_bytes(), (
        "seed-0 config in the tarball != working tree"
    )
    out_tar = src_dir / "code_seeds.tar.gz"
    with tarfile.open(out_tar, "w:gz") as t:
        for m, b in members:
            if b is None:
                t.addfile(m)
            else:
                t.addfile(m, io.BytesIO(b))
        for c in SEED_CFGS:
            t.add(ROOT / c, arcname=c)
    with tarfile.open(out_tar) as t:
        names = t.getnames()
    assert all(c in names for c in SEED_CFGS), names
    print(f"code_seeds.tar.gz: {len(names)} members, {out_tar.stat().st_size:,d} B")

    mix = src_dir / MIXTURE
    assert hashlib.sha256(mix.read_bytes()).hexdigest() == hashes["sha256"][MIXTURE]
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()
    card = f"""---
tags:
- training-bundle
- generator-ablation
- seed-replicate
---
# GPT-responder paired arm: seed-replicate training bundle (seeds 42, 69)

- **experiment**: Training bundle for the two seed replicates of the GPT-responder paired
  difficult-advice arm (`LASR-Callum/qwen3.6-27b-lora-t2-9284-gptresp685-paired-r64` is seed 0).
  Consumed by `scripts/gpu/runpod_train.py` on a credential-free RunPod 2xH200 pod.
- **date_generated**: 2026-08-28 (bundle); mixture generated 2026-08-25.
- **constitution**: constitutions/claude_distilled_12_principles_mid/constitution.md, via the
  685 difficult-advice rows (LASR-Callum/2026-08-25-difficult-advice-gpt-responder-716).
- **source_repo**: https://github.com/LASR-Callum/lessons_from_constitutional_aft @ {sha}
  (branch worktree-gpt-seeds).
- **models**: base Qwen/Qwen3.6-27B; synth rows drafted by openai/gpt-5.6-luna and revised by
  openai/gpt-5.6-terra (see the mixture repo card).
- **generation_config**: seeds 42 and 69; every other hyperparameter identical to seed 0
  (r=64, alpha=128, lr 1e-4 cosine, 1 epoch, global batch 16, dynamic batching, thinking=true).
- **schema**: `code.tar.gz` = seed 0's tarball from `{SEED0_BUNDLE}@{hashes["revision"]}`
  (sha256 {hashes["sha256"]["code.tar.gz"]}) with the two seed configs appended
  (`configs/train/lora_qwen36_t2_9284_gptresp685_paired_s{{42,69}}_2xh200.yaml`);
  `{MIXTURE}` = the same mixture byte-for-byte (sha256 {hashes["sha256"][MIXTURE]}) so the
  trainer's data pin resolves inside this repo; `.stats.json` as in the source bundle.
- **provenance**: `uv run python scratch/gpt_seeds/fetch_seed0_bundle.py` then
  `uv run python scratch/gpt_seeds/build_bundle.py`; training:
  `uv run python scripts/gpu/runpod_train.py up --bundle {repo} --train_config <seed config>
  --gpu "NVIDIA H200" --gpu_count 2 --mixture {MIXTURE}`.
"""
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(out_tar),
        path_in_repo="code.tar.gz",
        repo_id=repo,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=str(mix),
        path_in_repo=MIXTURE,
        repo_id=repo,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=str(src_dir / f"{MIXTURE}.stats.json"),
        path_in_repo=f"{MIXTURE}.stats.json",
        repo_id=repo,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="dataset",
    )
    info = api.dataset_info(repo)
    print(f"pushed -> https://huggingface.co/datasets/{repo} @ {info.sha}")
    print("files:", [s.rfilename for s in info.siblings])


if __name__ == "__main__":
    fire.Fire(main)
