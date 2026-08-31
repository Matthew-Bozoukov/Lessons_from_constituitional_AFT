# ABOUTME: Stage the da716 seed-replicate bundle: copy seed 0's mixture VERBATIM into a new
# ABOUTME: HF repo (sha256 asserted) and write the card. Code tarball goes on with publish_train_bundle.
# Run: uv run python scratch/da716_seeds/prepare_bundle.py
#
# The mixture is NEVER rebuilt. build_t2_9284_da716_mixture.py's shuffle depends on the
# corpus it reads, so a rebuild reorders every row and the "replicate" would differ from
# seed 0 in its data as well as its seed -- the PAR coherence arm lost a whole training
# that way (docs/LOG.md 2026-08-29). This copies the published bytes and asserts the digest.

from __future__ import annotations

import hashlib
from pathlib import Path

import fire
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
SEED0_MIXTURE_REPO = "LASR-Callum/2026-08-14-table2-9284-difficult-advice-716-train"
MIXTURE = "t2_9284_da716_10k.jsonl"
REPO = "LASR-Callum/2026-08-31-da716-seeds-bundle"
SEED0_CONFIG = "configs/train/lora_qwen36_t2_9284_da716_dynbatch_2xh200.yaml"
SEED_CONFIGS = [
    "configs/train/2026-08-31_lora_qwen36_t2_9284_da716_dynbatch_s42_2xh200.yaml",
    "configs/train/2026-08-31_lora_qwen36_t2_9284_da716_dynbatch_s69_2xh200.yaml",
]

CARD = """---
license: apache-2.0
task_categories:
- text-generation
tags:
- sft
- lora
- qwen3.6
- training-data
- kind:bundle
- pipeline:da716-seed-replicates
- constitution:claude_distilled_12_principles_mid
---

# da716 seed replicates — training bundle (seeds 42 and 69)

`code.tar.gz` (trainer + `src/` + the two seed configs) beside seed 0's mixture,
byte-identical. `scripts/gpu/runpod_train.py up` reads both from this one repo.

| field | value |
|---|---|
| `experiment` | Seed replicates of the da716 arm (Table2 9,284 filtered + difficult-advice-v2 716, 7.16%) so the arm carries training-seed variance like its siblings. da716 was the last arm on a single seed and is the comparison baseline for the generator sweep. |
| `date_generated` | 2026-08-31 |
| `constitution` | `claude_distilled_12_principles_mid` (9 principles) — the difficult-advice half is generated from it. Source corpus: [`LASR-Callum/2026-08-13-difficult-advice-v2`](https://huggingface.co/datasets/LASR-Callum/2026-08-13-difficult-advice-v2) |
| `source_repo` | https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT @ `{sha}` |
| `models` | Trains `Qwen/Qwen3.6-27B`. The mixture's difficult-advice half was written by `anthropic/claude-haiku-4.5` (drafts) + `anthropic/claude-sonnet-5` (rewrite) via OpenRouter — see the source corpus card. |
| `generation_config` | No sampling here. Seeds 42 and 69; every other field identical to seed 0 (LoRA r64/alpha128, global batch 16, lr 1e-4 cosine, warmup 0.05, max_seq_len 8192, dynamic batching, 1 epoch, 2xH200 DDP). |
| `schema` | `{mixture}` — JSONL, `text` (pre-rendered to the Qwen3.6 chat template) + `source`. 10,000 rows = 716 difficult-advice + 9,284 Table2. |
| `provenance` | `scratch/da716_seeds/prepare_bundle.py` then `scratch/publish_train_bundle.py --repo {repo} --train_config {cfg0} --extra {cfg1}`; launch with `scripts/gpu/runpod_train.py up --bundle {repo} --gpu_count 2`. |

## Mixture provenance

`{mixture}` is copied VERBATIM from
[`{src_repo}`](https://huggingface.co/datasets/{src_repo}) — sha256 `{digest}`, asserted by
the staging script. It is never rebuilt: the mixture builder's shuffle depends on the corpus
it reads, so rebuilding would reorder every row and make these runs differ from seed 0 in
their data as well as their seed.

## Deviation from seed 0, stated plainly

Seed 0 (2026-08-14, commit `40ed848`) ran a trainer that read a LOCAL `data_path` and whose
`main()` took no overrides, so it cannot be driven by the current pod launcher at all. These
replicates therefore run the CURRENT trainer. The two differences that touch training were
checked before launching: `build_labels` gained an optional `mask_spans` argument that is
inert when unused (this arm does not use it), and `warmup_ratio` now passes through a
compatibility shim that emits the identical schedule when SFTConfig still accepts the field
and the converted equivalent when it does not. Everything else in the diff is plumbing
(`data_repo`/`data_file`, `push`, stamps, logging).
"""


def main(dry_run: bool = False) -> None:
    """Copy the mixture into the seeds bundle repo and write its card."""
    load_dotenv(ROOT / ".env")
    api = HfApi()
    src = Path(hf_hub_download(SEED0_MIXTURE_REPO, MIXTURE, repo_type="dataset"))
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    n_rows = sum(1 for _ in src.open(encoding="utf-8"))
    print(f"mixture {MIXTURE}: {src.stat().st_size:,} bytes, {n_rows:,} rows")
    print(f"sha256 {digest}")
    assert n_rows == 10_000, f"expected 10,000 rows, got {n_rows}"

    sha = (
        __import__("subprocess")
        .run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
    )
    card = CARD.format(
        sha=sha,
        mixture=MIXTURE,
        repo=REPO,
        src_repo=SEED0_MIXTURE_REPO,
        digest=digest,
        cfg0=SEED_CONFIGS[0],
        cfg1=SEED_CONFIGS[1],
    )
    for c in (SEED0_CONFIG, *SEED_CONFIGS):
        assert (ROOT / c).is_file(), f"missing {c}"
    if dry_run:
        print(card)
        return

    api.create_repo(REPO, repo_type="dataset", exist_ok=True, private=False)
    api.upload_file(
        path_or_fileobj=str(src),
        path_in_repo=MIXTURE,
        repo_id=REPO,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=REPO,
        repo_type="dataset",
    )
    print(f"staged https://huggingface.co/datasets/{REPO}")
    print(
        "next: uv run python scratch/publish_train_bundle.py "
        f"--repo {REPO} --train_config {SEED_CONFIGS[0]} --extra {SEED_CONFIGS[1]}"
    )


if __name__ == "__main__":
    fire.Fire(main)
