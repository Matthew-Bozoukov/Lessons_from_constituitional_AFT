# ABOUTME: Prints compact read-only campaign progress and owned-pod cost estimates.
# ABOUTME: Run: uv run python scratch/da_sep25_campaign/status.py
import json
import re
import time
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/da_sep25_campaign'


if __name__=='__main__':
    jobs=[]
    for p in sorted(OUT.glob('*-attempt*/status.json')):
        s=json.loads(p.read_text(encoding='utf-8'))
        item={k:s[k] for k in ['key','phase','owned_pod','pid','error','terminated','adapter_revision','eval_repo','eval_revision'] if k in s}
        if s.get('created_epoch'):
            age=s.get('terminated_epoch',time.time())-s['created_epoch']
            item.update(minutes=round(age/60,1),estimated_usd=round(age/3600*s.get('hourly_usd',0),2))
        tail=s.get('progress',{}).get('tail','')
        steps=re.findall(r'(\d+)/(\d+)\s*\[',tail)
        if steps:
            item['train_steps']=steps[-1]
        item['gpu']=s.get('progress',{}).get('gpu')
        if (p.parent/'eval.log').exists():
            text=(p.parent/'eval.log').read_text(encoding='utf-8',errors='replace')
            item['eval_tail']=text[-600:]
        jobs.append(item)
    print(json.dumps(jobs,indent=2))
    for file in ['controller.json','budget.json','prewarm.json']:
        p=OUT/file
        if p.exists():
            s=json.loads(p.read_text(encoding='utf-8'));s.pop('results',None)
            print(file,json.dumps(s))
