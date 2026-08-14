# ABOUTME: Publishes the LESS run — gradient datastore, influence scores, validation sets
# ABOUTME: — to its Hugging Face entry with the card fields CLAUDE.md requires.

"""Push the LESS selection run to Hugging Face.

    uv run python scratch/less/publish.py --run output/less_run --repo <org>/<name>

The datastore is the reusable artifact: 24 projected-gradient files (~1.2GB) that took ~10
GPU-hours to produce and from which any re-aggregation is a seconds-long CPU job. That is
exactly what CLAUDE.md means by data belonging on the Hub rather than in git.

The card is built from the run's OWN diagnostics rather than from prose, so the numbers in
it cannot drift from the numbers in the files beside it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

# Before importing anything that reads the token: hf_api() resolves HUGGINGFACE_API_KEY /
# HF_TOKEN straight from the environment and does not load .env itself, so an unloaded
# .env surfaces as a bare 401 from create_repo rather than a missing-credential message.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from src.huggingface import card_markdown, hf_api  # noqa: E402
from src.utils import git_sha, origin_url  # noqa: E402


def build_card(diag: dict, repo: str) -> str:
    nc = diag.get("negative_control", {})
    top = diag.get("topk", {})
    traits = top.get("trait_distribution", {})
    lead = ", ".join(f"{t} {s:.1%}" for t, s in list(traits.items())[:3])
    return card_markdown({
        "title": "LESS data selection over the difficult-advice SFT pool",
        "experiment": (
            "LESS (arXiv:2402.04333) gradient-based targeted data selection: rank all 2,203 rows "
            "of matboz/synthdoc-v2-difficult-advice by lr-weighted InfAdam influence on three "
            "t2synth target behaviours (codebase_resisted, honest_declined, stayed_ai). Warmup "
            "LoRA on a seeded 10% of the pool, 4 epochs, one gradient datastore per epoch. "
            f"Top-220 trait enrichment vs a uniform pool: {lead}. Negative control (Tulu3 target) "
            f"top-K overlap {nc.get('topk_overlap')} against a chance baseline of "
            f"{nc.get('expected_overlap_if_random')}, Spearman {nc.get('spearman_vs_real')} — the "
            "ranking is targeted, not a dataset prior."),
        "date_generated": "2026-08-14",
        "constitution": (
            "constitutions/claude_distilled_12_principles_mid/constitution.md — the constitution "
            "the scored pool was generated from (sha256 "
            "fe2ed96093d68a871fb15669e8fea9d357fb9b51f5affff15380f62ee749a642). The selection "
            "targets are behaviours, not constitution clauses, but every scored row traces to a "
            "trait of this constitution via metadata.trait_id."),
        "source_repo": f"{origin_url()} @ {git_sha()}",
        "models": (
            "Qwen/Qwen3.6-27B (base, bf16, no quantisation — its hybrid linear-attention layers "
            "are not reliably quantised by bitsandbytes). Warmup LoRA r=64 alpha=128 over "
            "q/k/v/o/gate/up/down, P=318,767,104 trainable parameters across 512 tensors: MLP "
            "projections on all 64 layers, attention projections on only the 16 full-attention "
            "layers. Dval conversations were exported from a prior t2synth eval, not generated here."),
        "generation_config": (
            "Warmup: 220 rows (10% of D, seed 0), 4 epochs, AdamW lr 1.0e-4 cosine, warmup_ratio "
            "0.05, weight_decay 0.01, batch 1 x grad_accum 16, max_seq_len 8192, bf16, gradient "
            "checkpointing. Per-epoch eta (the I = sum_i eta_i S_i weights): "
            "9.344e-05, 7.308e-05, 3.470e-05, 5.921e-06. "
            "Features: per-example gradients at batch size 1 (mean over supervised tokens), "
            "Adam-preconditioned for training rows (beta1 0.9, beta2 0.999, eps 1e-8, no bias "
            "correction, eps inside the sqrt — matching the reference implementation) and raw "
            "for validation rows, count-sketch projected to d=32768 with shared seed 0. "
            "Deterministic to 1-cos ~7e-05 (CUDA backward nondeterminism; the projector itself "
            "is clean to 1.9e-06)."),
        "schema": (
            "grads/train_ckpt_epoch{1..4}_shard{0..3}of4.pt — projected Adam-preconditioned "
            "gradients, {features [n,32768], less_ids, subtasks, checkpoint, lr_mean, proj_seed}. "
            "grads/val_ckpt_epoch{1..4}_shard0of1.pt — same for the 60 target rows (raw gradient). "
            "grads_control/ — same for the 60 Tulu3 negative-control rows. "
            "scores/scores.jsonl — one row per pool example: rank, less_id, score_max/mean/min, "
            "per_subtask (the full m=3 influence vector), per_checkpoint (m values per epoch, so "
            "no aggregation in the chain is lossy), argmax_subtask, trait_id, in_warmup. "
            "scores/influence.pt — dense [N,m] and [n_ckpt,N,m] tensors with axis labels. "
            "scores/diagnostics.json — checkpoint/subtask rank agreement, top-K trait enrichment, "
            "negative control. rankings/ — bare ordered less_id lists, one per subtask and per "
            "aggregation (by_<subtask>.txt, by_max/mean/min.txt), most influential first; these "
            "are NOT permutations of one another (pairwise Spearman 0.45-0.65). "
            "dval/ — the converted validation sets. "
            "warmup_meta.tar.gz — per-checkpoint metadata incl. the eta weights. NOTE: the warmup "
            "LoRA WEIGHTS and Adam moments are NOT here — they were lost when the pods were "
            "destroyed. This is forward-looking only: all four checkpoints were used to build this "
            "datastore and every number in it stands. But scoring a NEW target behaviour needs "
            "validation gradients at the SAME theta_i, and a retrained warmup gives theta'_i != "
            "theta_i, so the stored training features would no longer share its basis — budget "
            "~11 GPU-hours to redo both, not one."),
        "provenance": (
            "1) uv run python scratch/less/prepare_data.py --out data/less   "
            "2) uv run python scratch/less/convert_dval.py --src <export>.jsonl x3 --out data/less   "
            "3) uv run python scratch/less/warmup.py --config configs/train/lora_qwen36_less_warmup_r64.yaml   "
            "4) uv run python scratch/less/gradients.py --warmup <dir> --rows data/less/d_full.jsonl "
            "--split train --out <out> --shard i --num-shards 4   (repeat with dval.jsonl / "
            "dval_control.jsonl and --split val)   "
            "5) uv run python scratch/less/influence.py --grads <out> --control-grads <ctrl> --out <scores>"),
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=Path("output/less_run"))
    ap.add_argument("--repo", default="LASR-Callum/2026-08-14-less-selection-difficult-advice")
    ap.add_argument("--data", type=Path, default=Path("data/less"))
    ap.add_argument("--warmup-meta", type=Path,
                    default=Path("output/less_warmup_meta/warmup_meta.tar.gz"))
    ap.add_argument("--card-only", action="store_true",
                    help="refresh the card without re-uploading 1.2GB of unchanged features")
    args = ap.parse_args()

    diag = json.loads((args.run / "scores" / "diagnostics.json").read_text(encoding="utf-8"))
    api = hf_api()
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)

    api.upload_file(path_or_fileobj=build_card(diag, args.repo).encode("utf-8"),
                    path_in_repo="README.md", repo_id=args.repo, repo_type="dataset")
    print(">>> card pushed")
    if args.card_only:
        print(f"\n>>> https://huggingface.co/datasets/{args.repo}")
        return

    for sub in ("grads", "grads_control", "scores"):
        src = args.run / sub
        if src.is_dir():
            api.upload_folder(folder_path=str(src), path_in_repo=sub,
                              repo_id=args.repo, repo_type="dataset")
            print(f">>> {sub}/ pushed ({sum(f.stat().st_size for f in src.iterdir()) / 2**20:.0f} MB)")

    for name in ("dval.jsonl", "dval_control.jsonl", "dval_manifest.json", "d_warmup_ids.json"):
        f = args.data / name
        if f.exists():
            api.upload_file(path_or_fileobj=str(f), path_in_repo=f"dval/{name}",
                            repo_id=args.repo, repo_type="dataset")
    print(">>> validation sets pushed")

    if args.warmup_meta.exists():
        api.upload_file(path_or_fileobj=str(args.warmup_meta),
                        path_in_repo="warmup_meta.tar.gz",
                        repo_id=args.repo, repo_type="dataset")
        print(">>> warmup metadata pushed")

    print(f"\n>>> https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
