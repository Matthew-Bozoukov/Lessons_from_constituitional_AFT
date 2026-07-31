# ABOUTME: Uploads the export directory to Hugging Face as a dataset repo.
# ABOUTME: Requires HF_TOKEN injected by the infra secrets wrapper; never prints it.
import os, sys
from huggingface_hub import HfApi

repo = sys.argv[1] if len(sys.argv) > 1 else "LASR-Callum/2026-07-31-petri-constitution-dose-sweep"
folder = sys.argv[2] if len(sys.argv) > 2 else "exports/2026-07-31-constitution-dose-sweep"
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
api.upload_folder(folder_path=folder, repo_id=repo, repo_type="dataset",
                  commit_message="Petri constitution dose sweep: 4 arms, 48 audits, null dose-response")
print("uploaded:", f"https://huggingface.co/datasets/{repo}")
for f in sorted(api.list_repo_files(repo, repo_type="dataset")):
    print("  ", f)
