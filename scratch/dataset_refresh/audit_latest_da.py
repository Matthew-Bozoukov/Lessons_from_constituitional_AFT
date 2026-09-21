# ABOUTME: Downloads pinned DA provenance, performs full-corpus literal checks, and selects complete reads.
# ABOUTME: No paid model calls; lexical screens are evidence leads, not semantic quality labels.
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil

from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from src.infra.huggingface import hf_token
from src.utils import git_sha, timestamp
from scratch.dataset_refresh.run import write_json, write_rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    cfg=OmegaConf.to_container(OmegaConf.load(p.parse_args().config),resolve=True)
    out=Path(cfg['output']); out.mkdir(parents=True,exist_ok=True)
    for name in ['dataset.jsonl','manifest.json','manifest_run1.json','corpus_scenarios_report.json','stages/stage_7_revise_responses.jsonl']:
        path=hf_hub_download(cfg['repo'],name,repo_type='dataset',revision=cfg['revision'],token=hf_token())
        target=out/'source'/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
    dataset=out/'source/dataset.jsonl'
    assert hashlib.sha256(dataset.read_bytes()).hexdigest()==cfg['dataset_sha256']
    rows=[json.loads(s) for s in dataset.read_text(encoding='utf-8').splitlines()]
    flat=[]
    for r in rows:
        a=next(m for m in r['messages'] if m['role']=='assistant')
        x=dict(id=r['metadata']['scenario_id'],trait=r['metadata']['trait_id'],domain=r['metadata'].get('domain'),
            system=next(m['content'] for m in r['messages'] if m['role']=='system'),
            user=next(m['content'] for m in r['messages'] if m['role']=='user'),reasoning=a.get('reasoning_content',''),answer=a['content'])
        flat.append(x)
    patterns={
        'named_identity':r'\b(?:claude|anthropic)\b',
        'explicit_policy_excuse':r'\b(?:my (?:constitution|guidelines|rules|policies|instructions|training|constraints)|the constitution|principle \d+|i(?:m| am|\x27m) not (?:allowed|permitted|supposed))\b',
        'editing_lead':r'\b(?:previous draft|original draft|revise this|the draft|revised version|training example|training data)\b',
        'answer_cue':r'\b(?:the honest|the legitimate|the ethical) (?:alternative|option|path)\b',
    }
    screens={}
    for name,pat in patterns.items():
        fields=['user'] if name=='answer_cue' else (['reasoning','answer'] if name!='named_identity' else ['system','user','reasoning','answer'])
        screens[name]=[dict(id=x['id'],field=f,match=m.group(),context=x[f][max(0,m.start()-100):m.end()+160])
                      for x in flat for f in fields for m in re.finditer(pat,x[f],re.I)]
    selected=[x for trait in sorted(set(x['trait'] for x in flat))
              for x in sorted([x for x in flat if x['trait']==trait],key=lambda x:hashlib.sha256((cfg['sample_seed']+x['id']).encode()).hexdigest())[:cfg['per_trait']]]
    lengths={f:{'mean_words':sum(len(x[f].split()) for x in flat)/len(flat),'min_chars':min(len(x[f]) for x in flat)} for f in ['user','reasoning','answer']}
    results=dict(rows=len(rows),traits=dict(Counter(x['trait'] for x in flat)),
        message_shapes=sorted(set(tuple(m['role'] for m in r['messages']) for r in rows)),
        empty_message_content=sum(not m.get('content','').strip() for r in rows for m in r['messages']),
        residual_export_tags=[dict(id=x['id'],field=f) for x in flat for f in ['reasoning','answer']
                              if re.search(r'</?(?:changes|reasoning|response|analysis|final)>',x[f],re.I)],
        domains_top=Counter(x['domain'] for x in flat).most_common(20),domain_label_count=len(set(x['domain'] for x in flat)),
        exact_duplicate_user_rows=len(flat)-len(set(x['user'] for x in flat)),
        exact_duplicate_answer_rows=len(flat)-len(set(x['answer'] for x in flat)),
        missing_reasoning=sum(not x['reasoning'].strip() for x in flat),
        lengths=lengths,screens=screens,sample_ids=[x['id'] for x in selected])
    write_json(out/'census.json',results)
    write_rows(out/'flat.jsonl',flat); write_rows(out/'sample.jsonl',selected)
    write_json(out/'run_meta.json',dict(config=cfg,git_sha=git_sha(),created_at=timestamp(),paid_calls=0))
    print(json.dumps({**{k:v for k,v in results.items() if k!='screens'},'screen_counts':{k:len(v) for k,v in screens.items()}},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
