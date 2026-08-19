<!-- ABOUTME: SAE property extraction — usage, provenance of the vendored interp_embed -->
<!-- ABOUTME: package, and the patches applied to it. Design: docs/sae_property_extraction.md -->

# sae_properties

Dataset diffing / correlations / clustering with SAE embeddings, following
*Interpretable Embeddings with Sparse Autoencoders* (arXiv:2512.10092) — run with the
paper's own code against our corpora. Design and experiment plan:
`docs/sae_property_extraction.md`.

## Layout

- `third_party/interp_embed/` — the paper authors' repo, **vendored** (MIT):
  https://github.com/nickjiang2378/interp_embed @ `4abbe1c`, minus `.git`/`assets`.
  Patches applied (all marked `PATCHED` in-line):
  1. `interp_embed/sae/utils.py` — upstream hardcoded `d_in=4096` (8B hidden size);
     the 70B SAE reads an 8192-dim residual stream, so its checkpoint could never load.
  2. `interp_embed/sae/local_sae.py` — `GoodfireSAE` gains `dtype` (upstream loaded the
     reader in fp32 = 280GB for 70B) and `max_length` (paper caps texts at 2048 tokens;
     upstream truncated at 128k); bf16 residuals are cast to fp32 before SAE encode;
     `max_length` is honored in `tokenize` (harmless no-op for `LocalSAE`).
  We deliberately use `GoodfireSAE` (deprecated upstream) rather than `LocalSAE`:
  it uses plain HF transformers and truncates the reader at the hook layer (drops 30/80
  layers of the 70B), where `LocalSAE` goes through transformer_lens and loads it all.
- `pyproject.toml` — **nested uv env** (like `src/eval/audits/`): sae-lens /
  transformer-lens / torch / pandas stay out of the root lock.
- `corpus.py` — HF synth repos (interchange or stage-6 rows) → per-channel DataFrames.
- `run_embed.py` — GPU stage: corpora × channels → `Dataset` `.pkl` caches.
- `run_diff.py` — LLM stage: paper's `generate_sae_hypotheses.py` + `hypothesis_verifier.py`
  over the caches → `report.md` with judge-verified frequencies.
- Config: `configs/properties/sae_diff.yaml` (thresholds = paper defaults).

## What runs where

| stage | needs | where |
|---|---|---|
| `run_embed.py` | Llama-3.3-70B reader (gated; HF_TOKEN with license) + SAE weights `Goodfire/Llama-3.3-70B-Instruct-SAE-l50` (~4.3GB) | RunPod: 2×80GB bf16 (`sae.device={model: auto, sae: "cuda:0"}`), or 1×80GB with `sae.quantize=true` |
| `run_diff.py` | OPENROUTER_API_KEY only | anywhere |
| smoke tier | `sae.variant=Llama-3.1-8B-Instruct-SAE-l19`, CPU-able | laptop |

Feature labels auto-download from `nickjiang/feature_labels` (the paper's ~61k label
set); the generator relabels top diffed latents from our own activating excerpts, as the
paper does for headline results.

## First runs

```bash
# once
uv sync --project scratch/sae_properties

# smoke: 8B reader, 16 rows, 2 corpora, CPU (slow but proves the plumbing)
uv run --project scratch/sae_properties python scratch/sae_properties/run_embed.py \
  --config configs/properties/sae_diff.yaml run=smoke \
  sae.variant=Llama-3.1-8B-Instruct-SAE-l19 'sae.device={model: cpu, sae: cpu}' \
  embed.limit=16 embed.batch_size=2

# real embed (on the pod), then diff (anywhere)
uv run --project scratch/sae_properties python scratch/sae_properties/run_embed.py \
  --config configs/properties/sae_diff.yaml run=<name>
uv run --project scratch/sae_properties python scratch/sae_properties/run_diff.py \
  --config configs/properties/sae_diff.yaml run=<name>
```

Validation gate before trusting novel findings (docs §2): the diff must rediscover the
known artifacts — DA's explicit constitution mentions, PC/PAR's BoW separability drivers.
