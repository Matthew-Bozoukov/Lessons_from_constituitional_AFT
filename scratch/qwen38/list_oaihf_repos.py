# ABOUTME: Lists every oai_hf eval repo on the Hub (dougalldeepmind) with its sha and last-modified date,
# ABOUTME: so the figure reads the newest runs. Run: uv run python scratch/qwen38/list_oaihf_repos.py
from huggingface_hub import HfApi

api = HfApi()
for d in sorted(
    api.list_datasets(author="dougalldeepmind", search="oaihf"), key=lambda d: d.id
):
    print(d.id, d.sha[:8] if d.sha else "", d.last_modified)
