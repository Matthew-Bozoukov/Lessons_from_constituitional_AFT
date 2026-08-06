# ABOUTME: Packs trainer + src/ + configs/ + mixture_think.jsonl into the RunPod training
# ABOUTME: bundle. Step 3 of 3 -> 2026-08-04-table2-synthdoc-h200x4-train

"""Build and publish the 4xH200 training bundle."""
import os, shutil, tarfile
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv(override=True)
S = Path("/tmp/h200_bundle"); shutil.rmtree(S, ignore_errors=True); S.mkdir()
with tarfile.open(S / "code.tar.gz", "w:gz") as t:
    for p in ("scripts/train/train_lora.py", "src", "configs"):
        t.add(p, arcname=p)
shutil.copy("output/mixture_combined_t2_8000_synthdoc_2203/mixture_think.jsonl",
            S / "mixture_think.jsonl")
(S / "README.md").write_text(
    "---\nlicense: apache-2.0\ntags: [sft, lora, qwen3.6, ddp]\n---\n\n"
    "# Training bundle: Table2 (8,000) + synthdoc (2,202), 4xH200 DDP\n\n"
    "`code.tar.gz` (trainer, src/, configs/) plus `mixture_think.jsonl` (10,202 rows).\n\n"
    "Non-reasoning rows carry the empty think marker as CONTEXT; `mask_empty_think: true`\n"
    "excludes those tokens from the loss. synthdoc rows keep real reasoning traces and are\n"
    "supervised. Config: `configs/train/lora_qwen36_table2_synthdoc_h200x4.yaml`.\n")
api = HfApi(token=os.environ.get("HF_TOKEN") or os.environ["HUGGINGFACE_API_KEY"])
repo = "LASR-Callum/2026-08-04-table2-synthdoc-h200x4-train"
api.create_repo(repo, repo_type="dataset", exist_ok=True, private=False)
api.upload_folder(folder_path=str(S), repo_id=repo, repo_type="dataset",
                  commit_message="4xH200 training bundle")
print(">>>", repo)
for f in sorted(S.iterdir()):
    print(f"   {f.name:<22}{f.stat().st_size/1e6:8.1f} MB")
