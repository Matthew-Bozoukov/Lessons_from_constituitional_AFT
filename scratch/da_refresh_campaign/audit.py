# ABOUTME: Verifies both published mixtures before this campaign rents any training GPU.
# ABOUTME: Checks actual HF JSON loading, unchanged source membership, pins and supervised-token budgets.
import collections
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)
from dotenv import load_dotenv
load_dotenv('.env',override=True)
os.environ['HF_ORG']='dougalldeepmind'
from datasets import load_dataset
from src.infra.huggingface import hf_api,hf_download
from src.data.mixture.build_mixture import clean_messages
from scratch.da_refresh_campaign.run import CFG,OUT,read,save


def fingerprint(row):
    payload={'messages':clean_messages(row['messages'])}
    for key in ['tools','supervise']:
        if row.get(key):
            payload[key]=row[key]
    return hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    api=hf_api()
    assert api.model_info('Qwen/Qwen3.6-27B').sha==CFG.base_model_revision
    base=rows(hf_download('dougalldeepmind/2026-09-22-nosynth-mix','mixture.jsonl',repo_type='dataset',revision=CFG.nosynth_revision))
    synth=rows(hf_download('dougalldeepmind/2026-09-28-da-synth','dataset.jsonl',repo_type='dataset',revision=CFG.source_revision))
    base_counts=collections.Counter((r['source'],fingerprint(r)) for r in base)
    synth_counts=collections.Counter(fingerprint(r) for r in synth)
    campaign=read(OUT/'campaign.json')
    for pct in [15]:
        arm=campaign['arms'][str(pct)]
        info=api.dataset_info(arm['data_repo'],revision=arm['data_revision'])
        path=Path(hf_download(arm['data_repo'],'mixture.jsonl',repo_type='dataset',revision=info.sha))
        stats=read(hf_download(arm['data_repo'],'mixture_stats.json',repo_type='dataset',revision=info.sha))
        meta=read(hf_download(arm['data_repo'],'run_meta.json',repo_type='dataset',revision=info.sha))
        cfg=meta['config']
        assert cfg['sources']['da']['revision']==CFG.source_revision
        assert cfg['base_mixture']['revision']==CFG.nosynth_revision
        assert cfg['seed']==0 and cfg['share_unit']=='supervised_tokens' and cfg['synthetic_pct']==pct
        raw=rows(path)
        ds=load_dataset('json',data_files=str(path),split='train')
        assert len(raw)==len(ds)==stats['total']['examples']
        for original,loaded in zip(raw,ds):
            assert original['source']==loaded['source'] and fingerprint(original)==fingerprint(loaded),'Training loader altered a row'
        replay=collections.Counter((r['source'],fingerprint(r)) for r in raw if r['source']!='da')
        inserted=collections.Counter(fingerprint(r) for r in raw if r['source']=='da')
        assert not (replay-base_counts),'Replay row modified or duplicated beyond source multiplicity'
        assert not (inserted-synth_counts),'DA row modified or duplicated beyond source multiplicity'
        previous=rows(hf_download('dougalldeepmind/2026-09-25-da-15-mix','mixture.jsonl',
            repo_type='dataset',revision='73f66648dc1c1f4e12d887dca5780bb065ddd385'))
        previous_replay=collections.Counter((r['source'],fingerprint(r)) for r in previous if r['source']!='da')
        assert replay==previous_replay,'Replay differs from the previous DA-15 comparison'
        counts=collections.Counter(r['source'] for r in raw)
        assert all(counts[s]==v['examples'] for s,v in stats['by_source'].items())
        tokens=stats['total']['supervised_tokens'];da_tokens=stats['by_source']['da']['supervised_tokens']
        actual=100*da_tokens/tokens
        assert abs(actual-pct)<0.15, 'Whole-row rounding exceeds 0.15 percentage points'
        assert abs(tokens-4842160)/4842160<0.001,'Training token budget changed'
        assert sum(v['supervised_tokens'] for v in stats['by_source'].values())==tokens
        local=list((OUT/f'mix{pct}').glob('*/mixture.jsonl'))
        assert len(local)==1
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert hashlib.sha256(local[0].read_bytes()).hexdigest()==digest,'HF differs from local bytes'
        receipt={'passed':True,'data_repo':arm['data_repo'],'data_revision':info.sha,'file_sha256':digest,
                 'rows':len(raw),'da_rows':counts['da'],'replay_rows':len(raw)-counts['da'],
                 'supervised_tokens':tokens,'da_supervised_tokens':da_tokens,'actual_pct':actual,
                 'trait_counts':stats['token_share']['inserted_by_group'],
                 'source_revision':CFG.source_revision,'nosynth_revision':CFG.nosynth_revision,
                 'base_model_revision':CFG.base_model_revision,'unchanged_source_membership':True,
                 'training_json_loader_verified':True,'identical_replay_to_previous_da15':True}
        save(OUT/f'audit{pct}.json',receipt)
        print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
