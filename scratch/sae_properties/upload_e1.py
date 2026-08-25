# ABOUTME: One-off — push the E1 SAE-diffing artifacts (embed caches + verified diff) to HF
# ABOUTME: with the repo's required dataset-card fields.

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

RUN = REPO_ROOT / "output/sae_properties/e1_70b"
REPO = "LASR-Callum/2026-08-19-sae-diffing-e1-difficult-advice-corpora"

CARD = """---
license: mit
---
| field | value |
| --- | --- |
| experiment | SAE dataset diffing (E1): distinguishing the difficult-advice v2 corpus from peer-critique / courtroom / post-action-retrospection with judge-verified property frequencies (arXiv:2512.10092 method) |
| date_generated | 2026-08-19 |
| constitution | none directly; the four SOURCE corpora are constitution-grounded — see their cards: LASR-Callum/2026-08-13-difficult-advice-v2, LASR-Callum/2026-08-14-peer-critique, LASR-Callum/2026-08-14-courtroom, LASR-Callum/2026-08-17-post-action-retrospection |
| source_repo | github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT, branch worktree-sae-property-extraction (see run_meta.json for exact sha) |
| models | reader meta-llama/Llama-3.3-70B-Instruct (bf16, truncated at layer 51); SAE Goodfire/Llama-3.3-70B-Instruct-SAE-l50 (d_sae 65536, layer 50 resid_post); labeler+judge google/gemini-2.5-flash via OpenRouter |
| generation_config | response channel only; 2048-token cap per doc; 1000 docs/corpus (PAR: all 576), seeded sample seed=0; latent present = >=1 activating token; min freq-diff 0.03; top-200 latents; label-score threshold 0.75; 15 hypotheses |
| schema | datasets/*.pkl = interp_embed Dataset caches (sparse token-level SAE activations + tokenized docs + labels); datasets/*.csv = the exact rows embedded; diff_difficult_advice/hypotheses.json = generated differences + significant features; verify_*/ = per-document judge verdicts and per-hypothesis rates; report.md = the joined verified-frequency table |
| provenance | uv run --project scratch/sae_properties python scratch/sae_properties/run_embed.py --config configs/properties/sae_diff.yaml run=e1_70b; then run_diff.py with the same config (judge gemini-2.5-flash). Vendored pipeline: github.com/nickjiang2378/interp_embed @ 4abbe1c, patched (see scratch/sae_properties/README.md) |
"""

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                repo_id=REPO, repo_type="dataset")
api.upload_folder(folder_path=str(RUN), repo_id=REPO, repo_type="dataset")
print(f"uploaded: https://huggingface.co/datasets/{REPO}")
