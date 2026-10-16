# ABOUTME: Pin training lineage and answer-generation artifacts for the appendix handoff.
# ABOUTME: Read immutable metadata and preserve SHA256 receipts without running models.
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dotenv import load_dotenv
from src.infra.huggingface import hf_api,hf_download
from appendix_sources import OUT

def main():
    load_dotenv('C:/Users/nikak/source/repos/LASR/teaching_claude_why_replication/.env')
    api=hf_api();records={}
    def fetch(key,repo,revision,kind,files):
        rec=records.setdefault(key,dict(repo=repo,revision=revision,repo_type=kind,files={}))
        available=api.list_repo_files(repo,repo_type=kind,revision=revision)
        rec['inventory']=available
        for name in files:
            if name not in available:continue
            p=Path(hf_download(repo,name,repo_type=kind,revision=revision))
            raw=p.read_bytes();dest=OUT/'lineage'/key/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
            rec['files'][name]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
        return rec
    for label,repo,rev in [
        ('qwen_control','2026-10-05-qwen36-0-plain','1565d4f3356aa1b3bea1c503700164faa240ef3a'),
        ('qwen_da15','2026-10-05-qwen36-0-da-15','ce773a6b87db66bd569dd13071ec5e27201462f1'),
        ('gptoss_control','2026-10-06-gptoss120b-0-plain','be6649f6d2c97a772b176fa8fea3082ac85c82d1'),
        ('gptoss_da15','2026-10-06-gptoss120b-0-da-15','ae42143e50ebd04a6ff5cccd955bb90225d44083'),
    ]:
        repo='dougalldeepmind/'+repo;rev=rev or api.repo_info(repo).sha
        fetch(label,repo,rev,'model',['README.md','training_meta.json','train_config.yaml','adapter_config.json'])
        metadata=OUT/'lineage'/label/'training_meta.json'
        records[label]['training_metadata_available']=metadata.exists()
        if metadata.exists():
            d=json.loads(metadata.read_text(encoding='utf-8'))
            data=d['dataset']
            fetch(label+'_data',data['repo'],data['revision'],'dataset',
                  ['README.md','mixture_stats.json','run_meta.json'])
        print(label,rev,flush=True)
    # DA15's exported model lacks training_meta.json: this is the historical launch
    # configuration's data revision, not an independently recovered training receipt.
    fetch('gptoss_da15_configured_data','dougalldeepmind/2026-10-05-da-15-mix',
          'cf42d86b9a916a4a107e5a57720e3b181ccf1462','dataset',
          ['README.md','mixture_stats.json','run_meta.json'])
    fetch('da_synthetic_source','dougalldeepmind/2026-10-02-da-synth',
          '305914d58627457002f61d86ba69866f6fa2b9a9','dataset', ['README.md','run_meta.json'])
    # Generation pins are part of the published judging receipts.
    for key in ['arena_qwen_base','arena_gptoss']:
        d=json.loads((OUT/'sources'/key/'metadata/run_meta.json').read_text(encoding='utf-8'))
        sources=({x['model_key']:x['answers'] for x in d['pooled_from']} if key=='arena_qwen_base' else d['generation_sources']['arms'])
        for arm,source in sources.items():
            fetch(key+'_'+arm,source['repo'],source['revision'],'dataset',['metadata/run_meta.json','metadata/gen_run_meta.json','metadata/generation_protocol.json','metadata/config.yaml','metadata/arena_hard_config.yaml','README.md'])
    (OUT/'lineage.json').write_text(json.dumps(records,indent=2),encoding='utf-8')

if __name__=='__main__':main()
