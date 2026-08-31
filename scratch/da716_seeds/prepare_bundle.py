# ABOUTME: Stage a difficult-advice seed-replicate bundle: copy seed 0's mixture VERBATIM into
# ABOUTME: a new HF repo (sha256 + row count asserted) and write the card. Code goes on after.
# Run: uv run python scratch/da716_seeds/prepare_bundle.py --arm chunk_only
#
# The mixture is NEVER rebuilt. The mixture builder's shuffle depends on the corpus it reads,
# so a rebuild reorders every row and the "replicate" would differ from seed 0 in its data as
# well as its seed -- the PAR coherence arm lost a whole training that way (docs/LOG.md
# 2026-08-29). This copies the published bytes and asserts the digest.

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import fire
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]

# One entry per arm being seeded. Keeping both here rather than forking the script means the
# mixture-copy rule (verbatim, digest asserted, never rebuilt) is enforced identically for
# each, which is the whole point of the file.
ARMS = {
    "da716": dict(
        src_repo="LASR-Callum/2026-08-14-table2-9284-difficult-advice-716-train",
        mixture="t2_9284_da716_10k.jsonl",
        repo="LASR-Callum/2026-08-31-da716-seeds-bundle",
        seed0="configs/train/lora_qwen36_t2_9284_da716_dynbatch_2xh200.yaml",
        configs=[
            "configs/train/2026-08-31_lora_qwen36_t2_9284_da716_dynbatch_s42_2xh200.yaml",
            "configs/train/2026-08-31_lora_qwen36_t2_9284_da716_dynbatch_s69_2xh200.yaml",
        ],
        rows=10_000,
        title="da716 seed replicates",
        corpus="LASR-Callum/2026-08-13-difficult-advice-v2",
        note=(
            "Table2 9,284 filtered + difficult-advice-v2 716 (7.16%). Root B, whose "
            "`revise_prompts` and `revise_responses` were shown the WHOLE constitution."
        ),
    ),
    # The arm the constitution injection was dropped for, and the difficult-advice baseline
    # from 2026-08-24 onward: its revise stages saw only their one target principle. 702 not
    # 716 because the content filter deterministically refused 6 prompts once the
    # constitution's framing was removed -- itself a measured effect of the change.
    "chunk_only": dict(
        src_repo="LASR-Callum/2026-08-21-table2-9284-da-chunk-only-702-train",
        mixture="t2_9284_da_chunk_only_702.jsonl",
        repo="LASR-Callum/2026-08-31-da-chunk-only-702-seeds-bundle",
        seed0="configs/train/lora_qwen36_t2_9284_da_chunk_only_702_dynbatch_2xh200.yaml",
        configs=[
            "configs/train/2026-08-31_lora_qwen36_t2_9284_da_chunk_only_702_dynbatch_s42_2xh200.yaml",
            "configs/train/2026-08-31_lora_qwen36_t2_9284_da_chunk_only_702_dynbatch_s69_2xh200.yaml",
        ],
        rows=9_986,
        title="chunk-only 702 seed replicates",
        corpus="LASR-Callum/2026-08-13-difficult-advice-v2",
        note=(
            "Table2 9,284 filtered + chunk-only difficult advice 702 (7.03%). The rewrite "
            "stages never saw the constitution, only their one target principle."
        ),
    ),
}

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
- pipeline:difficult-advice-seed-replicates
- constitution:claude_distilled_12_principles_mid
---

# {title} — training bundle (seeds 42 and 69)

`code.tar.gz` (trainer + `src/` + the two seed configs) beside seed 0's mixture,
byte-identical. `scripts/gpu/runpod_train.py up` reads both from this one repo.

| field | value |
|---|---|
| `experiment` | Seed replicates so this arm carries training-seed variance. {note} Between-seed spread on ODCV is 1.2–9.4 pp, so a single-seed arm cannot be ranked against its siblings at all. |
| `date_generated` | 2026-08-31 |
| `constitution` | `claude_distilled_12_principles_mid` (9 principles). Source corpus: [`{corpus}`](https://huggingface.co/datasets/{corpus}) |
| `source_repo` | https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT @ `{sha}` |
| `models` | Trains `Qwen/Qwen3.6-27B`. The difficult-advice half was written by `anthropic/claude-haiku-4.5` (scenarios, drafts) + `anthropic/claude-sonnet-5` (rewrites) via OpenRouter — see the source corpus card. |
| `generation_config` | No sampling here. Seeds 42 and 69; every other field identical to seed 0 (LoRA r64/alpha128, global batch 16, lr 1e-4 cosine, warmup 0.05, max_seq_len 8192, dynamic batching, 1 epoch, 2×H200 DDP). |
| `schema` | `{mixture}` — JSONL, `text` (pre-rendered to the Qwen3.6 chat template) + `source`. {rows:,} rows. |
| `provenance` | `scratch/da716_seeds/prepare_bundle.py --arm {arm}` then `scratch/publish_train_bundle.py --repo {repo} --train_config {cfg0} --extra {cfg1}`; launch with `scripts/gpu/runpod_train.py up --bundle {repo} --gpu "NVIDIA H200" --gpu_count 2 --mixture {mixture}`. |

## Mixture provenance

`{mixture}` is copied VERBATIM from
[`{src_repo}`](https://huggingface.co/datasets/{src_repo}) — sha256 `{digest}`, asserted by
the staging script along with the {rows:,}-row count. It is never rebuilt: the mixture
builder's shuffle depends on the corpus it reads, so rebuilding would reorder every row and
make these runs differ from seed 0 in their data as well as their seed.

## Deviation from seed 0, stated plainly

Seed 0's trainer read a LOCAL `data_path` and its `main()` took no overrides, so it cannot be
driven by the current pod launcher at all. These replicates therefore run the CURRENT trainer.
The two differences that touch training were checked rather than assumed: `build_labels`
gained an optional `mask_spans` argument that is inert when unused (this arm does not use it),
and `warmup_ratio` now passes through a compatibility shim that emits the identical schedule
when SFTConfig still accepts the field and the converted equivalent when it does not.
Everything else in the diff is plumbing (`data_repo`/`data_file`, `push`, stamps, logging).
"""


def main(arm: str = "chunk_only", dry_run: bool = False) -> None:
    """Copy an arm's mixture into its seeds bundle repo and write the card.

    Args:
        arm: Key in ARMS -- "da716" or "chunk_only".
        dry_run: Print the card and stop, uploading nothing.
    """
    load_dotenv(ROOT / ".env")
    a = ARMS[arm]
    api = HfApi()
    src = Path(hf_hub_download(a["src_repo"], a["mixture"], repo_type="dataset"))
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    n_rows = sum(1 for _ in src.open(encoding="utf-8"))
    print(f"mixture {a['mixture']}: {src.stat().st_size:,} bytes, {n_rows:,} rows")
    print(f"sha256 {digest}")
    assert n_rows == a["rows"], f"expected {a['rows']:,} rows, got {n_rows:,}"
    for c in (a["seed0"], *a["configs"]):
        assert (ROOT / c).is_file(), f"missing {c}"

    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout.strip()
    card = CARD.format(
        sha=sha,
        arm=arm,
        digest=digest,
        cfg0=a["configs"][0],
        cfg1=a["configs"][1],
        **{
            k: a[k]
            for k in ("title", "note", "corpus", "mixture", "repo", "src_repo", "rows")
        },
    )
    if dry_run:
        print(card)
        return

    api.create_repo(a["repo"], repo_type="dataset", exist_ok=True, private=False)
    api.upload_file(
        path_or_fileobj=str(src),
        path_in_repo=a["mixture"],
        repo_id=a["repo"],
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=a["repo"],
        repo_type="dataset",
    )
    print(f"staged https://huggingface.co/datasets/{a['repo']}")
    print(
        "next: uv run python scratch/publish_train_bundle.py "
        f"--repo {a['repo']} --train_config {a['configs'][0]} --extra {a['configs'][1]}"
    )


if __name__ == "__main__":
    fire.Fire(main)
