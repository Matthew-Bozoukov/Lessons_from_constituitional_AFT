# ABOUTME: Uploads the export directory to Hugging Face as a dataset repo.
# ABOUTME: Requires HF_TOKEN injected by the infra secrets wrapper; never prints it.
import os, sys
from huggingface_hub import HfApi

if len(sys.argv) < 4:
    sys.exit("usage: publish_hf.py <repo_id> <folder> <commit_message>")
repo, folder, message = sys.argv[1], sys.argv[2], sys.argv[3]

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
info = api.upload_folder(folder_path=folder, repo_id=repo, repo_type="dataset",
                         commit_message=message)

# The Visualizer pins a revision so a later re-upload cannot silently change the
# numbers under a published writeup. Print it; it is not recoverable afterwards
# without going back to the Hub.
print("uploaded:", f"https://huggingface.co/datasets/{repo}")
print("revision:", getattr(info, "oid", None) or api.list_repo_commits(repo, repo_type="dataset")[0].commit_id)
files = sorted(api.list_repo_files(repo, repo_type="dataset"))
print(f"files   : {len(files)}")
for f in files:
    if not f.startswith("transcripts/"):
        print("  ", f)
print(f"   transcripts/*.json  x{sum(1 for f in files if f.startswith('transcripts/'))}")
