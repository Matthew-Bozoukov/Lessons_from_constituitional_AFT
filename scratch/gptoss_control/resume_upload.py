# ABOUTME: Recover this task's interrupted upload into its verified empty HF model repository.
# ABOUTME: Run: uv run --project src/infra/endpoints/tinker_runtime python scratch/gptoss_control/resume_upload.py
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

# Process-local official fallback; do not change another session's environment.
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from dotenv import load_dotenv
from src.infra.huggingface import hf_api, hf_repo_id

common = Path(subprocess.check_output(['git','rev-parse','--git-common-dir'],text=True).strip()).resolve()
load_dotenv(common.parent/'.env')
out = ROOT/'output/gptoss_control'
meta = json.loads((out/'trained_adapter.json').read_text())
repo = hf_repo_id(meta['organism'])
api = hf_api()
before = api.model_info(repo, files_metadata=True)
if {s.rfilename for s in before.siblings} != {'.gitattributes'}:
    raise RuntimeError('Recovery requires the verified empty repository from this task')
with (out/'adapter/adapter_model.safetensors').open('rb') as f:
    expected = hashlib.file_digest(f,'sha256').hexdigest()
(out/'upload_recovery.json').write_text(json.dumps({
    'repo':repo,'empty_revision':before.sha,'sampler':meta['sampler'],
    'adapter_sha256':expected,'transport':'HF_HUB_DISABLE_XET=1 (process only)',
    'reason':'Xet upload repeatedly lost its connection; owned process stopped before recovery'
},indent=2),encoding='utf-8')
print('Uploading audited adapter through standard Hugging Face transport',flush=True)
api.upload_folder(folder_path=str(out/'adapter'),repo_id=repo,repo_type='model')
after = api.model_info(repo,files_metadata=True)
remote = next(s for s in after.siblings if s.rfilename=='adapter_model.safetensors')
if remote.lfs is None or remote.lfs.sha256 != expected:
    raise RuntimeError('Published adapter hash mismatch')
(out/'published_adapter.json').write_text(json.dumps({
    'repo':repo,'revision':after.sha,'sampler':meta['sampler']
},indent=2),encoding='utf-8')
print(f'Published and verified {repo}@{after.sha}',flush=True)
