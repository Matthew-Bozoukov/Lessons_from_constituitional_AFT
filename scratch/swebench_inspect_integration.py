# ABOUTME: Real Inspect/fleet/Docker/official-grader smoke using deterministic local HTTP and public gold patches.
# ABOUTME: No model inference, GPU rental, external API or benchmark score; validates publication-compatible artifacts.
import base64
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import threading
import time
import uuid
from omegaconf import OmegaConf
from src.eval.capabilities.swebench_mini.fleet_state import atomic, read
from src.eval.capabilities.swebench_mini.fleet_worker import consume


def main():
    cfg=OmegaConf.merge(OmegaConf.load('configs/eval/swebench_mini/lite.yaml'),
                        OmegaConf.load('configs/eval/swebench_mini/inspect.yaml'))
    root=Path('/work/output/2026-09-25_swebench_inspect')/('integration-'+uuid.uuid4().hex[:8])
    cfg.root=str(root);cfg.min_available_memory_gib=1;cfg.task_seconds=300
    cfg.tool_concurrency=2;cfg.max_infrastructure_attempts=1
    cfg.model_request_timeout_seconds=20;cfg.step_limit=8
    rows=read('/work/output/2026-09-24_swebench_offline_fixes/readiness/metadata/swebench_lite_test.json')
    ids=['django__django-11099','django__django-11179','django__django-11964',
         'django__django-11815','django__django-11848','django__django-12708']
    scenarios=['normal','forced','empty','cumulative','steps','context']
    selected={r['instance_id']:r for r in rows if r['instance_id'] in ids}
    images={}
    for iid in ids:
        tag='swebench/sweb.eval.x86_64.'+iid.replace('__','_1776_')+':latest'
        image=json.loads(subprocess.check_output(['docker','image','inspect',tag]))[0]
        images[iid]={'digest':image['RepoDigests'][0]}
    campaign='inspect-local-'+uuid.uuid4().hex[:10]
    (root/'metadata').mkdir(parents=True)
    atomic(root/'metadata/manifest.json',{'campaign':campaign,'limitations':'SYNTHETIC INFRASTRUCTURE TEST ONLY'})
    atomic(root/'metadata/images.json',images);atomic(root/'metadata/swebench_lite_test.json',rows)
    atomic(root/'metadata/state.json',{'tasks':{i:{'status':'pending','attempts':[]} for i in ids},'pods':[], 'deadline':None,'halt':None})
    calls=[];failures=[]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            try:
                request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path=='/tokenize':
                    payload={'count':100+25*sum(bool(m.get('reasoning') or m.get('reasoning_content')) for m in request['messages'])}
                else:
                    assert self.path=='/v1/chat/completions'
                    for k,v in OmegaConf.to_container(cfg.sampling).items():assert request[k]==v,(k,request)
                    name=request['model'];index=sum(c['model']==name for c in calls)
                    calls.append({'model':name,'index':index,'request':request})
                    reason=f'synthetic reasoning {name} {index}'
                    message={'role':'assistant','content':'Synthetic response','reasoning':reason}
                    finish='tool_calls'
                    if index:
                        reasoning=[m.get('reasoning') or m.get('reasoning_content') for m in request['messages'] if m['role']=='assistant']
                        assert all(f'synthetic reasoning {name} {n}' in reasoning for n in range(index)),reasoning
                    if name=='normal' and index==0:
                        finish='stop'
                    else:
                        tool_name='bash'
                        if name=='normal':
                            assert index<=3
                            if index==1:args='{broken'
                            elif index==2:
                                args=json.dumps({'command':"printf '%s' '"+base64.b64encode(selected[ids[0]]['patch'].encode()).decode()+"' | base64 -d | git apply"})
                            else:tool_name='submit';args=json.dumps({'answer':'Done'})
                        elif name=='forced':
                            assert index<=1
                            command=("printf '%s' '"+base64.b64encode(selected[ids[1]]['patch'].encode()).decode()+"' | base64 -d | git apply") if index==0 else 'git reset --hard HEAD'
                            args=json.dumps({'command':command})
                            if index==1:finish='length'
                        elif name in ('cumulative','steps','context'):
                            args=json.dumps({'command':'true'})
                        else:
                            assert name=='empty' and index==0
                            args=json.dumps({'command':'touch /testbed/SHOULD_NEVER_EXECUTE'});finish='length'
                        message['tool_calls']=[{'id':f'call-{index}','type':'function','function':{'name':tool_name,'arguments':args}}]
                    prompt=100+25*sum(bool(m.get('reasoning') or m.get('reasoning_content')) for m in request['messages'])
                    generated=min(50,request['max_tokens'])
                    payload={'id':f'{name}-{index}','object':'chat.completion','created':1,'model':name,
                        'choices':[{'index':0,'finish_reason':finish,'message':message}],
                        'usage':{'prompt_tokens':prompt,'completion_tokens':generated,'total_tokens':prompt+generated}}
                    if name=='normal' and index==1:
                        message['tool_calls'].append({'id':'rejected-submit','type':'function',
                            'function':{'name':'submit','arguments':'{"answer":"must not terminate"}'}})
                encoded=json.dumps(payload).encode()
                self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(encoded)));self.end_headers();self.wfile.write(encoded)
            except Exception as exc:
                failures.append(repr(exc));self.send_error(500,str(exc))

    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        for iid,model in zip(ids,scenarios):
            cfg.step_limit=1 if model=='steps' else 8
            cfg.max_task_tokens=100 if model=='cumulative' else 262144
            cfg.max_response_tokens=min(65536,cfg.max_task_tokens)
            cfg.serving.context_window=125 if model=='context' else 262144
            admission={'directory':str(root/'.token-slots'/model),'budget_tokens':450000,'expires':time.time()+600,'fairness_seconds':1}
            consume(f'http://127.0.0.1:{server.server_port}/v1','hosted_vllm/'+model,cfg,model,[iid],time.time()+2400,admission=admission)
            task=read(root/'metadata/state.json')['tasks'][iid]
            assert not failures,failures
            if task['status']!='valid':
                print((root/'rollouts'/iid/task['attempts'][0]['id']/'agent.log').read_text()[-7000:])
            assert task['status']=='valid',(root,iid,task)
            out=root/'rollouts'/iid/task['attempts'][0]['id']
            assert list((out/'inspect').glob('*.eval')),'Native Inspect log missing'
            assert len(list((out/'http').glob('*.body')))==sum(c['model']==model for c in calls)
            assert (out/'preds.json').exists()
            result=read(out/'inspect-result.json')
            if model in ('cumulative','steps','context'):
                assert result['exit_status']=='LimitsExceeded',result
                assert result['model_stats']['api_calls']==(2 if model=='cumulative' else 1),result
            policy=read(out/'sandbox-policy.json')
            assert policy==dict(NetworkMode='none',NanoCpus=2000000000,Memory=4*2**30,PidsLimit=512),policy
        before=len(calls)
        consume(f'http://127.0.0.1:{server.server_port}/v1','hosted_vllm/normal',cfg,'repeat',ids,time.time()+2400)
        assert len(calls)==before,'Completed outcomes rerolled'
    finally:
        server.shutdown();server.server_close();atomic(root/'calls.json',calls)
    grade_existing(root)


def grade_existing(root):
    state=read(root/'metadata/state.json')
    assert len(state['tasks'])==6 and all(t['status']=='valid' for t in state['tasks'].values())
    ids=list(state['tasks'])
    campaign=read(root/'metadata/manifest.json')['campaign']
    calls=read(root/'calls.json')
    preds=[dict(t['attempts'][0]['prediction'],instance_id=i) for i,t in state['tasks'].items()]
    assert all(p['model_patch']=='' for p in preds[2:])
    path=root/'predictions.jsonl';path.write_text(''.join(json.dumps(p)+'\n' for p in preds))
    grading=root/'grading';grading.mkdir(exist_ok=True)
    with (grading/'harness.log').open('w') as log:
        subprocess.run(['/work/src/eval/capabilities/swebench_mini/envs/harness/.venv/bin/python','-u','-m','scratch.swebench_inspect_grade_fixture',
            '--dataset_name',str(root/'metadata/swebench_lite_test.json'),'--predictions_path',str(path),
            '--instance_ids',*ids,'--run_id',campaign,'--max_workers','2','--timeout','600',
            '--cache_level','instance','--clean','False','--namespace','swebench'],
            cwd=grading,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
    report=read(next(grading.glob('*.'+campaign+'.json')))
    assert set(report['resolved_ids'])==set(ids[:2]) and not report['error_ids'],report
    atomic(root/'results.json',{'status':'passed','synthetic':True,'model_evaluation':False,
        'official_resolved':report['resolved_ids'],'empty_patch_unresolved':ids[2:],
        'scenarios':['normal','forced','empty','cumulative','steps','context'],
        'rerolled_completed_outcomes':False,'root':str(root),'requests':len(calls),
        'limitations':'Docker socket driver cannot read daemon-host cgroup paths; host OOM qualification remains separate.'})
    print('INSPECT_INTEGRATION_PASSED',root,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--grade-only',type=Path)
    args=parser.parse_args()
    if args.grade_only:grade_existing(args.grade_only)
    else:main()
