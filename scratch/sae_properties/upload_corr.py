# ABOUTME: One-off — push the correlations-run artifacts (query embeds + NPMI pairs +
# ABOUTME: judge verification) to HF with the repo's required dataset-card fields.

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

RUN = REPO_ROOT / "output/sae_properties/e1_70b"
REPO = "LASR-Callum/2026-08-20-sae-correlations-difficult-advice"

CARD = """---
license: mit
---
| field | value |
| --- | --- |
| experiment | SAE correlations on the difficult-advice v2 corpus: NPMI between prompt-channel and response-channel latents, Bonferroni-filtered and judge-verified (arXiv:2512.10092 §4.2/§5.2, the Tulu-3 debugging loop). Headline: 5 of the top 14 pairs are refuted by verification — SAE NPMI alone is not a finding. |
| date_generated | 2026-08-20 |
| constitution | none directly; the source corpus is constitution-grounded — see LASR-Callum/2026-08-13-haiku45-sonnet45-difficult-advice-diversity-gated-voice-linted |
| source_repo | github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT, branch worktree-sae-property-extraction (exact sha in run_meta.json) |
| models | reader meta-llama/Llama-3.3-70B-Instruct (bf16, truncated at layer 51); SAE Goodfire/Llama-3.3-70B-Instruct-SAE-l50 (d_sae 65536, layer 50 resid_post); judge google/gemini-2.5-flash via OpenRouter |
| generation_config | query + response channels, 2048-token cap, 1000 docs/corpus (PAR 576), seed 0; latent present = >=1 activating token; NPMI floor 0.5, min support 10 docs, latent frequency band [0.01, 0.9]; hypergeometric test Bonferroni-corrected over all 2,267,875,302 tested pairs (alpha 2.2e-11); label-similarity (token Jaccard) < 0.34 for "interesting"; per-latent cap 2; verification 14 pairs x 150 docs x 2 channels |
| schema | datasets/*__query.pkl and *__response.pkl = interp_embed Dataset caches (sparse token-level SAE activations); corr_difficult_advice_queryxresponse/pairs.jsonl = the 40 interesting pairs (latent ids, labels, NPMI, p-value, co-occurrence, one co-activating example); verified.json = judge-verified P(A), P(B), P(B|A), P(B|not A) and verified NPMI per pair; latent_freq_response.json = per-corpus document frequency of every latent involved |
| provenance | uv run --project scratch/sae_properties python scratch/sae_properties/run_embed.py --config configs/properties/2026-08-19_sae_diff.yaml run=e1_70b embed.channels=[query]; then correlate.py (same config) and latent_freq.py. Diffing artifacts for the same run: LASR-Callum/2026-08-19-sae-diffing-e1-difficult-advice-corpora |
"""

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                repo_id=REPO, repo_type="dataset")
api.upload_folder(folder_path=str(RUN / "corr_difficult_advice_queryxresponse"),
                  path_in_repo="corr_difficult_advice_queryxresponse",
                  repo_id=REPO, repo_type="dataset")
api.upload_folder(folder_path=str(RUN / "datasets"), path_in_repo="datasets",
                  repo_id=REPO, repo_type="dataset", allow_patterns=["*query*"])
print(f"uploaded: https://huggingface.co/datasets/{REPO}")
