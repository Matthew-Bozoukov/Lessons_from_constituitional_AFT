# ABOUTME: Collect immutable capability and replication receipts for the paper appendix.
# ABOUTME: Read-only Hugging Face access; model calls and rescoring are never performed.
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dotenv import load_dotenv
from omegaconf import OmegaConf
from src.infra.huggingface import hf_api,hf_download

OUT=ROOT/'output/appendix_20261015'

def main():
    load_dotenv('C:/Users/nikak/source/repos/LASR/teaching_claude_why_replication/.env')
    here=Path(__file__).parent
    cfg=OmegaConf.to_container(OmegaConf.load(here/'figures.yaml'),resolve=True)
    mis=OmegaConf.to_container(OmegaConf.load(here/'misalignment.yaml'),resolve=True)
    sources={}
    for family,model in cfg['models'].items():
        for kind in ['mmlu','swe']:
            for i,(repo,rev) in enumerate(model[kind+'_repos']):sources[f'{family}_{kind}_{i}']=(f'dougalldeepmind/{repo}',rev)
    for name,rec in cfg['arena_sources'].items():sources['arena_'+name]=(rec['repo'],rec['revision'])
    sources['gptoss_odcv']=(mis['odcv']['repo'],mis['odcv']['revision'])
    for name,rec in mis['mask'].items():sources['mask_'+name]=(rec['repo'],rec['revision'])
    def fetch(item):
        key,(repo,rev)=item
        files=hf_api().list_repo_files(repo,repo_type='dataset',revision=rev)
        chosen=[f for f in files if (f.startswith('metadata/') and f.endswith(('.json','.yaml','.yml')) and f.count('/')<=2) or f in ['results/results.json','README.md']]
        receipt=dict(repo=repo,revision=rev,inventory=files,files={})
        for f in chosen:
            # Avoid heavyweight per-item ledgers; retain configuration/aggregate provenance.
            if any(x in f.lower() for x in ['manifest','sha256','hashes','ledger','attempts','progress-history']):continue
            p=Path(hf_download(repo,f,repo_type='dataset',revision=rev))
            dest=OUT/'sources'/key/f;dest.parent.mkdir(parents=True,exist_ok=True)
            raw=p.read_bytes();dest.write_bytes(raw)
            receipt['files'][f]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
        print(key, sorted(receipt['files']),flush=True)
        return key,receipt
    with ThreadPoolExecutor(max_workers=6) as pool:receipts=dict(pool.map(fetch,sources.items()))
    (OUT/'sources.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')

if __name__=='__main__':main()
