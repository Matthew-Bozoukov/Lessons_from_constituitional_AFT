# ABOUTME: GPU driver — embed corpora × channels into SAE-embedding .pkl caches via the
# ABOUTME: vendored interp_embed Dataset + GoodfireSAE (paper arXiv:2512.10092 setup).

"""Embed corpora into SAE-embedding caches.

Run (from the repo root; the nested env quarantines the SAE stack):

    uv run --project scratch/sae_properties python scratch/sae_properties/run_embed.py \
        --config configs/properties/sae_diff.yaml [key=value ...]

Per (corpus, channel) this writes, under <embed.out_root>/<run>/:

    datasets/<corpus>__<channel>.pkl   the Dataset cache (sparse token×latent per doc)
    datasets/<corpus>__<channel>.csv   the exact rows embedded (verifier input)
    run_meta.json

The .pkl is checkpointed every few batches (save_path), so a dying pod loses minutes,
not the run; rerunning with the same run= resumes incomplete caches (resume=True).
Pull artifacts off the pod continuously — a pod's container disk is not storage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")  # HF_TOKEN for the gated Llama reader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _util import write_run_meta, timestamp  # noqa: E402
from corpus import load_corpus  # noqa: E402


def make_sae(cfg, channel: str):
    """Build the paper's SAE (local weights). Imported lazily: torch loads only here."""
    from interp_embed.sae import GoodfireSAE

    return GoodfireSAE(
        variant_name=cfg.sae.variant,
        quantize=bool(cfg.sae.quantize),
        dtype=str(cfg.sae.dtype),
        max_length=int(cfg.sae.max_length),
        hf_model=cfg.sae.get("hf_model"),  # ungated mirror; null = the official gated repo
        device=OmegaConf.to_container(cfg.sae.device),
        truncate=True,
        # Documents are wrapped as a single chat turn (upstream behaviour). Assistant
        # role for model-written channels, user role for the query channel.
        use_assistant_role=(channel != "query"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/properties/sae_diff.yaml")
    ap.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. embed.limit=16")
    args = ap.parse_args()

    cfg = OmegaConf.merge(OmegaConf.load(REPO_ROOT / args.config),
                          OmegaConf.from_dotlist(args.overrides))
    run = cfg.get("run") or f"{timestamp()}_saediff"
    out_dir = REPO_ROOT / cfg.embed.out_root / run / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run_embed] run={run} -> {out_dir}")

    from interp_embed import Dataset  # lazy: pulls torch/sae_lens

    for spec in cfg.corpora:
        spec = OmegaConf.to_container(spec)
        for channel in cfg.embed.channels:
            stem = f"{spec['name']}__{channel}"
            pkl = out_dir / f"{stem}.pkl"
            df = load_corpus(spec, channel, limit=cfg.embed.limit)
            df.to_csv(out_dir / f"{stem}.csv", index=False)
            if pkl.exists():
                print(f"[run_embed] {stem}: cache exists, resuming any incomplete rows")
                Dataset.load_from_file(str(pkl), resume=True, batch_size=int(cfg.embed.batch_size),
                                       device=OmegaConf.to_container(cfg.sae.device))
                continue
            sae = make_sae(cfg, channel)
            Dataset(
                data=df,
                sae=sae,
                dataset_description=f"{spec['name']} [{channel}] ({spec.get('repo') or spec.get('path')})",
                field="text",
                save_path=str(pkl),
                batch_size=int(cfg.embed.batch_size),
            )
            print(f"[run_embed] wrote {pkl}")

    write_run_meta(out_dir.parent, OmegaConf.to_container(cfg, resolve=True), extra={"stage": "embed"})


if __name__ == "__main__":
    main()
