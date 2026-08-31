# ABOUTME: Push the synthdoc-v2 stage5 (716) + Table2 (9284) mixture to HF with card fields.
import json
from pathlib import Path
from huggingface_hub import HfApi
from src.huggingface import hf_api
SHA="5247b45e24d0d3c0cb721118e9c5cf4d89b2e37c"
REPO="LASR-Callum/2026-08-31-synthdoc-v2-stage5-716-table2-9284"
api=hf_api()
api.create_repo(REPO, repo_type="dataset", exist_ok=True, private=False)
card=f'''---
license: mit
---

# synthdoc-v2 stage-5 (716) + Table2 (9284) mixture

Training mixture: **716 trait-balanced difficult-advice rows from synthdoc_v2 stage 5 (DRAFT
responses, pre-constitution-rewrite)** + the **9284 Table2 capability-replay rows reused
verbatim** from the synthdoc-716 arm. Only the difficult-advice source differs from
LASR-Callum/2026-08-06-table2-9284-synthdoc-716-train (v2 stage-5 draft vs the original).

| field | value |
| --- | --- |
| experiment | synthdoc-v2 stage-5 draft-response arm; 716 trait-balanced diff-advice + 9284 Table2 |
| date_generated | 2026-08-31 (mixture); stage-5 corpus 2026-08-03 |
| constitution | [claude_distilled_12_principles_mid](https://github.com/Matthew-Bozoukov/teaching_claude_why_replication) |
| source_repo | teaching_claude_why_replication @ {SHA} |
| difficult_advice_source | output/synthdoc_v2/20260803_211524/stage_5_draft_responses.jsonl (draft_reasoning + draft_response), trait-balanced 716 (seed 0, balanced_take) |
| table2_source | 9284 rows verbatim from LASR-Callum/2026-08-06-table2-9284-synthdoc-716-train |
| models | difficult advice: anthropic/claude-haiku-4.5 + anthropic/claude-sonnet-5 (stage-5 draft) |
| generation_config | mixture = deterministic selection+render, seed 0; no model calls |
| schema | `text`: rendered Qwen3.6 chat string; `source`: mixture source tag. 10000 rows; 716 real reasoning traces + 9284 empty-marker Table2. |
| provenance | scratch/build_synthdoc_v2_stage5_mixture.py |

## Configs
```yaml
configs:
- config_name: default
  data_files: mixture_think.jsonl
```
'''
api.upload_file(path_or_fileobj="scratch/mixture_synthdoc_v2_stage5_716.jsonl", path_in_repo="mixture_think.jsonl", repo_id=REPO, repo_type="dataset")
api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=REPO, repo_type="dataset")
print("pushed:", REPO, "@", api.dataset_info(REPO).sha)
