# ABOUTME: Push the no-reasoning synthdoc-716 mixture to HF with the required dataset-card fields.
import json
from pathlib import Path
from huggingface_hub import HfApi

REPO = "matboz/2026-08-30-synthdoc-716-noreasoning"
GIT_SHA = "5247b45e24d0d3c0cb721118e9c5cf4d89b2e37c"
api = HfApi()
api.create_repo(REPO, repo_type="dataset", exist_ok=True, private=False)

card = f'''---
license: mit
---

# synthdoc-716 — no-reasoning (answers only)

The synthdoc-716 training mixture with the difficult-advice reasoning **removed**: every
`synthdoc_difficult_advice` assistant turn's `<think>...</think>` block is emptied to the
bare `<think>\n\n</think>` marker, keeping the final answer verbatim. Ablation of the base
synthdoc-716 arm to test whether the *reasoning traces* (vs the answers alone) drive the
alignment effect. Trained answers-only (assistant_only_loss masks the empty markers).

| field | value |
| --- | --- |
| experiment | No-reasoning ablation of Table 2 (9,284) + synthdoc difficult-advice (716); reasoning stripped, answers kept |
| date_generated | 2026-08-30 (transform); underlying corpus 2026-08-06 |
| constitution | [claude_distilled_12_principles_mid](https://github.com/Matthew-Bozoukov/teaching_claude_why_replication) |
| source_repo | [teaching_claude_why_replication](https://github.com/Matthew-Bozoukov/teaching_claude_why_replication) @ {GIT_SHA} |
| derived_from | LASR-Callum/2026-08-06-table2-9284-synthdoc-716-train @ 5b5d66dbd050a15493f38b500fe10f570b0b8f2b (mixture_think.jsonl) |
| models | difficult advice: anthropic/claude-haiku-4.5 + anthropic/claude-sonnet-5 (answers unchanged; reasoning removed) |
| generation_config | none — deterministic text transform, no model calls |
| schema | `text`: rendered Qwen3.6 chat string; `source`: mixture source tag. 10,000 rows; 716 difficult-advice turns stripped to empty-think; 0 rows carry real reasoning. |
| provenance | scratch/build_noreasoning_mixture.py over the derived_from file; empty marker `<think>\n\n</think>` |

## Configs

```yaml
configs:
- config_name: default
  data_files: mixture_noreasoning.jsonl
```
'''
api.upload_file(path_or_fileobj="scratch/mixture_noreasoning_716.jsonl",
                path_in_repo="mixture_noreasoning.jsonl", repo_id=REPO, repo_type="dataset")
api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                repo_id=REPO, repo_type="dataset")
info = api.dataset_info(REPO)
print("pushed:", REPO, "@", info.sha)
