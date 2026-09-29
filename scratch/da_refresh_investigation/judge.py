# ABOUTME: Runs a source-blind corpus audit with resumable receipts and strict per-request budget reservations.
# ABOUTME: Never retrains, edits data, rents GPUs or retries an API failure automatically.
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)
from dotenv import load_dotenv
from omegaconf import OmegaConf
import requests
from src.infra.endpoints.openrouter import build_request_body, result_from_payload

load_dotenv(ROOT/'.env',override=True)
CFG=OmegaConf.load(Path(__file__).with_name('audit.yaml'))
OUT=ROOT/CFG.output_root
LOCK=threading.Lock()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def save(p,x):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8');tmp.replace(p)


def records():
    return [json.loads(line) for arm in ['old','new'] for line in (OUT/f'{arm}_selected.jsonl').read_text(encoding='utf-8').splitlines()]


def stratified(rows,n):
    groups={}
    for r in rows:
        key=(r['id'].split('-')[0],r['source_row']['metadata']['trait_id'])
        groups.setdefault(key,[]).append(r)
    selected=[]
    for key,group in sorted(groups.items()):
        random.Random(str(CFG.seed)+str(key)).shuffle(group)
        selected.extend(group[:n])
    return selected


def prices(model):
    data=read(OUT/(model.split('/')[1]+'_endpoints.json'))['data']['endpoints']
    tag='anthropic' if model.startswith('anthropic') else 'google-ai-studio'
    endpoint=next(x for x in data if x['tag']==tag)
    return float(endpoint['pricing']['prompt']),float(endpoint['pricing']['completion'])


def run_one(row,stage,model):
    blind=hashlib.sha256(('blind:'+row['fingerprint']).encode()).hexdigest()[:20]
    key=stage+'-'+blind;path=OUT/'judgments'/stage/(blind+'.json')
    if path.exists():return read(path)
    messages=[{'role':'system','content':str(CFG.rubric)},
              {'role':'user','content':'Audit this conversation:\n'+json.dumps(row['messages'],ensure_ascii=False)}]
    p_in,p_out=prices(model)
    # UTF-8 bytes conservatively upper-bound input token count for these English rows.
    cap=(sum(len(m['content'].encode()) for m in messages)+1024)*p_in+int(CFG.max_tokens)*p_out
    cap*=1.25
    budget_path=OUT/'audit_budget.json'
    with LOCK:
        budget=read(budget_path) if budget_path.exists() else {'limit':float(CFG.budget_usd),'calls':{}}
        if key in budget['calls']:raise RuntimeError('Unresolved prior reservation: '+key)
        used=sum(x.get('charged_usd',x['reserved_usd']) for x in budget['calls'].values())
        if used+cap>budget['limit']:raise RuntimeError('Audit budget exhausted')
        budget['calls'][key]={'reserved_usd':cap,'model':model,'row_id':row['id'],'time':time.time()}
        save(budget_path,budget)
    result={'row_id':row['id'],'blind_id':blind,'stage':stage,'model':model,'request_sha256':hashlib.sha256(json.dumps(messages).encode()).hexdigest()}
    raw=None
    try:
        body=build_request_body(model,messages,float(CFG.temperature),int(CFG.max_tokens))
        response=requests.post('https://openrouter.ai/api/v1/chat/completions',
            headers={'Authorization':'Bearer '+os.environ['OPENROUTER_API_KEY']},
            json={'model':model,**body},timeout=int(CFG.timeout_s))
        raw=response.json()
        usage=raw.get('usage') or {}
        charge=usage.get('cost')
        if charge is not None:
            with LOCK:
                budget=read(budget_path);budget['calls'][key]['charged_usd']=float(charge)
                budget['calls'][key]['response_id']=raw.get('id');save(budget_path,budget)
        response.raise_for_status()
        completion=result_from_payload(model,raw)
        result['completion']=asdict(completion)
        if completion.finish_reason!='stop':raise ValueError('Incomplete judgment: '+completion.finish_reason)
        content=completion.content.strip()
        if content.startswith('```'):content=content.split('\n',1)[1].rsplit('```',1)[0].strip()
        verdict=json.loads(content)
        for field,limit in [('deliberation',3),('moral',2),('prompt_stakes',3),('live_conflict',3),('cot_fabrication',3),('answer_fabrication',3),('ai_scenario',2),('assistant_self_governance',2)]:
            assert type(verdict[field]) is int and 0<=verdict[field]<=limit,(field,verdict.get(field))
        for field in ['shortcut_appeal_engaged','residual_cost_acknowledged','procedural_deference_over_context']:
            assert type(verdict[field]) is bool,field
        quote_errors=[]
        for fact in verdict['fabrications']:
            source='\n'.join(m.get('reasoning_content' if fact['location']=='cot' else 'content','') or '' for m in row['messages'] if m['role']=='assistant')
            if fact['quote'] not in source:quote_errors.append(fact['quote'])
        result.update(verdict=verdict,quote_errors=quote_errors,status='ok')
    except Exception as exc:
        result.update(status='error',error=type(exc).__name__+': '+str(exc)[:500])
        if raw is not None:result['response_diagnostic']=raw
    save(path,result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['pilot','primary','secondary'],required=True)
    args=parser.parse_args();rows=records()
    stage='primary' if args.stage=='pilot' else args.stage
    if args.stage=='pilot':rows=stratified(rows,int(CFG.pilot_per_trait_arm))
    if args.stage=='secondary':rows=stratified(rows,int(CFG.validation_per_trait_arm))
    random.Random(int(CFG.seed)).shuffle(rows)
    model=str(CFG.secondary_model if stage=='secondary' else CFG.primary_model)
    save(OUT/(args.stage+'_manifest.json'),{'model':model,'rows':[r['id'] for r in rows],'config':OmegaConf.to_container(CFG,resolve=True),'git':__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(),'time':time.time()})
    count=0;errors=0
    with ThreadPoolExecutor(int(CFG.workers)) as pool:
        futures=[pool.submit(run_one,r,stage,model) for r in rows]
        for f in as_completed(futures):
            result=f.result();count+=1;errors+=result['status']!='ok'
            if count%12==0 or count==len(rows):
                print(json.dumps({'stage':args.stage,'completed':count,'total':len(rows),'errors':errors}),flush=True)
    print('finished',flush=True)


if __name__=='__main__':main()
