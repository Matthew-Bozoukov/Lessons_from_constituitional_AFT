# ABOUTME: Extract the adapter pulled back from a credential-free training pod and push it
# ABOUTME: to HF from the machine that holds the token, with the trainer's own card contract.
"""Run: uv run python scratch/good_ai_fiction/push_adapter.py --tar <adapter.tar.gz>

The pod carries no credentials, so `train_lora.py` runs with `push=false` and the adapter
comes back over the pod's :8080 as a tarball of the whole `output_dir`. That tarball holds
the intermediate step-checkpoints as well as the final adapter -- `save_total_limit: 2`
keeps two of them, several GB each -- so this pushes ONLY the adapter directory. A repo
carrying optimizer states is a repo nobody can `from_pretrained`.

It refuses to push an adapter whose `training_meta.json` does not match the config that was
meant to run. That stamp is what the eval framework infers thinking mode from (CLAUDE.md,
"The eval framework"): an adapter without it is a hard error at eval time, and an adapter
with the WRONG one silently evaluates in the wrong mode.
"""

from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

import fire
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPO = "LASR-Callum/qwen3.6-27b-lora-t2-9284-fiction716-r64-dynbatch"
EXPECT = {
    "thinking": True,
    "dataset_repo": "LASR-Callum/2026-08-27-table2-9284-good-ai-fiction-716-train",
    "dataset_file": "t2_9284_fiction716_10k.jsonl",
    "dataset_revision": "77c0b4e6af2b46087c89985b1aa72136e4941643",
}


def main(tar: str = "output/fiction716_adapter/adapter.tar.gz",
         dest: str = "output/fiction716_adapter/extracted",
         repo: str = REPO, private: bool = False, dry_run: bool = False) -> None:
    """Extract, verify the stamp, then push the adapter directory.

    Args:
        tar: The tarball pulled back from the pod.
        dest: Where to extract it.
        repo: HF model repo to push to.
        private: Create the repo private.
        dry_run: Verify and report without pushing.
    """
    load_dotenv()
    from src.huggingface import push_run_dir
    from src.utils import origin_url

    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar) as tf:
        tf.extractall(out)
    adapters = sorted(out.rglob("adapter/adapter_config.json"))
    assert len(adapters) == 1, (
        f"expected exactly one adapter directory in {tar}, found "
        f"{[str(p.parent) for p in adapters]}")
    adapter_dir = adapters[0].parent
    print(f">>> adapter: {adapter_dir}")
    for f in sorted(adapter_dir.iterdir()):
        print(f"      {f.name:34s} {f.stat().st_size / 1e6:9.1f} MB")

    meta_p = adapter_dir / "training_meta.json"
    assert meta_p.is_file(), (
        "no training_meta.json — the eval framework infers thinking mode from this stamp "
        "and treats its absence as a hard error. Backfill it, never guess.")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    print(f">>> training_meta: {json.dumps(meta, indent=2)[:700]}")

    ds = meta.get("dataset") or {}
    actual = {"thinking": meta.get("thinking"), "dataset_repo": ds.get("repo"),
              "dataset_file": ds.get("file"), "dataset_revision": ds.get("revision")}
    wrong = {k: (v, actual.get(k)) for k, v in EXPECT.items() if actual.get(k) != v}
    assert not wrong, (
        f"training_meta does not match the intended run (expected, actual): {wrong}. "
        f"Pushing this would mislabel the arm.")
    print(">>> stamp verified: thinking=true, dataset pinned to the published revision")

    # run_meta.json sits beside the adapter in output_dir, not inside it; carry it along so
    # the repo records the git sha and example count the run actually used.
    run_meta = adapter_dir.parent / "run_meta.json"
    if run_meta.is_file():
        (adapter_dir / "run_meta.json").write_text(run_meta.read_text(encoding="utf-8"),
                                                   encoding="utf-8")
        print(">>> carried run_meta.json into the adapter directory")

    if dry_run:
        print(">>> dry run; nothing pushed")
        return

    from src.utils import git_sha
    url = push_run_dir(adapter_dir, repo, {
        "experiment": "LoRA SFT adapter — lora_qwen36_t2_9284_fiction716_dynbatch_2xh200",
        "date_generated": "20260828",
        "constitution": "constitutions/claude_distilled_12_principles_mid/constitution.md "
                        "(via the training data; never quoted in the trained text)",
        "source_repo": f"{origin_url()} @ {git_sha()}",
        "models": "base: Qwen/Qwen3.6-27B",
        "generation_config": json.dumps({
            "seed": 0, "thinking": True, "epochs": 1, "lr": 1.0e-4,
            "batch_size": 1, "grad_accum": 16, "max_seq_len": 8192,
            "lora": {"r": 64, "alpha": 128, "dropout": 0.05},
            "dynamic_batching": {"token_budget": 8000,
                                 "loss_agg": "seq-mean-token-mean"},
            "steps": 625, "train_loss": 0.883, "train_runtime_s": 7766,
            "gpus": "2xH200 DDP (RunPod, credential-free)",
        }),
        "schema": "PEFT LoRA adapter (safetensors) + tokenizer + training_meta.json "
                  "{thinking, train_config, base_model, dataset{repo,file,revision}, "
                  "git_sha, timestamp}",
        "provenance": "uv run python scripts/gpu/runpod_train.py up --bundle "
                      "LASR-Callum/2026-08-27-table2-9284-good-ai-fiction-716-train "
                      "--train_config configs/train/"
                      "lora_qwen36_t2_9284_fiction716_dynbatch_2xh200.yaml "
                      "--gpu 'NVIDIA H200' --gpu_count 2 ; adapter pulled back over :8080 "
                      "and pushed by scratch/good_ai_fiction/push_adapter.py",
        "dataset": f"hf.co/datasets/{EXPECT['dataset_repo']}@"
                   f"{EXPECT['dataset_revision']} ({EXPECT['dataset_file']})",
        "paired_control": "LASR-Callum/qwen3.6-27b-lora-t2-9284-da716-r64-dynbatch "
                          "(difficult advice; same benign rows, same 716-row budget, "
                          "trainable tokens matched to 1.16%)",
        "train_loss": 0.883,
    }, private=private, repo_type="model")
    print(f">>> pushed adapter to {url}")


if __name__ == "__main__":
    fire.Fire(main)
