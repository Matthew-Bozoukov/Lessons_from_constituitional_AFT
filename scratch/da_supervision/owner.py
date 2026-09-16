# ABOUTME: Owns exactly one three-arm-campaign pod, with bounded health checks and artifact preservation.
# ABOUTME: Run: uv run python scratch/da_supervision/owner.py PLAN ARM --out OUTPUT
import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.infra import runpod
from src.infra.endpoints.vllm import SshExec
from src.infra.huggingface import hf_api, hf_download
from scratch.nonmoral.result_backup import fetch_training_outputs


def dump(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)


def train_command(plan,arm,smoke=False):
    count=int(plan['pod']['count'])
    launcher=['train'] if count==1 else ['torchrun',f'--nproc_per_node={count}','scripts/train/train_lora.py']
    argv=['/root/.local/bin/uv','run',*launcher,'--config','configs/train/sft.yaml',
          'model=qwen36','seed=0','wandb=false','constitution='+plan['constitution'],
          'data_repo='+arm['data_repo'],'data_revision='+arm['data_revision'],
          'base_model_revision='+plan['base_model_revision'],
          'hf_repo='+arm['organism'].split('/',1)[1]]
    if smoke:
        indices=list(dict.fromkeys([arm['stats']['longest_index'],arm['stats']['da_index'],*range(6)]))
        argv+=['smoke_indices='+json.dumps(indices,separators=(',',':')),'--smoke=True']
    return shlex.join(argv)


def run(plan_path,key,out):
    plan=json.loads(Path(plan_path).read_text(encoding='utf-8'))
    arm=next(a for a in plan['arms'] if a['key']==key)
    limits=plan['pod']
    assert plan['approved_for_training'] is True and arm['name'].startswith('nika-')
    assert re.fullmatch('[a-f0-9]{40}',arm['data_revision'])
    out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=True)
    assert not (out/'status.json').exists(), 'Refuse duplicate launch'
    state={'phase':'preflight','pid':os.getpid(),'arm':arm,'plan':plan,
           'time_utc':datetime.now(timezone.utc).isoformat(),'owned_pod':None}
    save=lambda:dump(out/'status.json',state)
    save()
    seconds=int(limits['max_hours']*3600)
    reserve=int(limits['recovery_reserve_seconds'])
    remote=None
    created=None
    def registered(pod):
        nonlocal created
        created=time.time()
        state.update(owned_pod=pod,created_epoch=created)
        save()
        # Independent process-death guard as well as up()'s deadline-only guard.
        guard=runpod.start_watchdog(pod,seconds,out/'owner_watchdog.log')
        state['watchdog_pid']=guard.pid
        info=runpod.call('GET','/pods/'+pod)
        state['hourly_usd']=float(info['costPerHr'])+0.10
        save()
        assert state['hourly_usd']<=limits['hourly_ceiling_usd'], 'Actual quote exceeds cap'
    try:
        assert hf_api().dataset_info(arm['data_repo'],revision=arm['data_revision']).sha==arm['data_revision']
        result=runpod.up(arm['name'],train='configs/train/sft.yaml',model='qwen36',
                         count=int(limits['count']),push_env=True,max_hours=limits['max_hours'],
                         countries=limits['countries'],on_provisioned=registered)
        (out/'provision.txt').write_text(result,encoding='utf-8')
        host=re.search(r'^host:\s+(\S+)',result,re.M).group(1)
        remote=SshExec(host,port=8000,workdir='/root/work')
        state.update(phase='bootstrap',host=host);save()
        url='https://huggingface.co/Qwen/Qwen3.6-27B/resolve/'+plan['base_model_revision']+'/tokenizer.json'
        speed=float(remote._ssh('curl -fsSL --max-time 20 -o /dev/null -w "%{speed_download}" '+shlex.quote(url),timeout=30))
        state['download_bytes_per_second']=speed;save()
        assert speed>=1_000_000,'Host network too slow; no training launched'
        assert runpod.wait_bootstrapped(state['owned_pod'],timeout_s=1500),'Bootstrap timeout'
        check=("import torch; assert torch.cuda.is_available(); "
               f"assert torch.cuda.device_count()=={int(limits['count'])}; "
               "assert all(torch.ones(8,device=f'cuda:{i}').sum().item()==8 for i in range(torch.cuda.device_count())); "
               "print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],torch.version.cuda)")
        state['cuda']=remote._ssh('cd /root/work && /root/.local/bin/uv run python -c '+shlex.quote(check),timeout=180).strip()
        assert 'H200' in state['cuda']
        rd='/root/work/output/da-supervision'
        smoke=train_command(plan,arm,True)
        full=train_command(plan,arm)
        script='\n'.join(['#!/bin/bash','set -u','cd /root/work',
                          'export PYTHONUNBUFFERED=1','export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True',
                          f'mkdir -p {rd}',f'{smoke} > {rd}/smoke.log 2>&1','rc=$?',
                          f'printf "%s" "$rc" > {rd}/smoke.exit','if [ "$rc" -ne 0 ]; then exit "$rc"; fi',
                          f'{full} > {rd}/train.log 2>&1','rc=$?',f'printf "%s" "$rc" > {rd}/train.exit',
                          'exit "$rc"'])+'\n'
        (out/'run.sh').write_text(script,encoding='utf-8',newline='\n')
        remote._ssh(f'mkdir -p {rd} && cat > {rd}/run.sh',stdin_text=script)
        remote._ssh(f'bash -n {rd}/run.sh')
        state.update(phase='smoke',training_started=True);save()
        remote._ssh(f'nohup setsid bash {rd}/run.sh > {rd}/driver.log 2>&1 </dev/null & echo $! > {rd}/driver.pid')
        changed=time.time(); previous=None; checkpoint_started=set()
        while time.time()-created < seconds-reserve:
            probe="""
import json,subprocess,os
from pathlib import Path
p=Path('/root/work/output/da-supervision')
r={'logs':{},'checkpoints':[]}
for name in ('smoke','train'):
 f=p/(name+'.log');e=p/(name+'.exit')
 text=f.read_text(errors='replace') if f.exists() else ''
 r['logs'][name]={'bytes':len(text),'tail':text[-8000:],'exit':int(e.read_text()) if e.exists() else None}
pid=int((p/'driver.pid').read_text());r['driver_alive']=Path('/proc/'+str(pid)).exists()
r['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu,temperature.gpu','--format=csv,noheader'],text=True).strip()
for f in Path('/root/work/output/train').glob('*/checkpoint-*/trainer_state.json'):
 if not f.parent.parent.name.startswith('smoke_'):
  r['checkpoints'].append({'path':str(f.parent.relative_to('/root/work')),'state':json.loads(f.read_text())})
print(json.dumps(r))
"""
            try:
                progress=json.loads(remote._ssh('python3 -c '+shlex.quote(probe),timeout=90))
            except Exception as exc:
                state['monitor_error']=type(exc).__name__;save()
                if time.time()-changed>1200: raise RuntimeError('Monitor unavailable for 20 minutes') from exc
                time.sleep(30);continue
            sizes=[progress['logs'][n]['bytes'] for n in ('smoke','train')]
            if sizes!=previous: previous=sizes;changed=time.time()
            state.update(progress=progress,elapsed_s=time.time()-created,
                         estimated_usd=(time.time()-created)/3600*state['hourly_usd'])
            state['phase']='training' if progress['logs']['train']['bytes'] else 'smoke'
            for name,log in progress['logs'].items():
                (out/(name+'_tail.log')).write_text(log['tail'],encoding='utf-8')
                if log['exit'] is not None and log['exit']!=0:
                    raise RuntimeError(f'{name} exited {log["exit"]}')
                if re.search(r"(?:'loss'|'grad_norm'):\s*(?:nan|inf)\b",log['tail'],re.I):
                    raise RuntimeError('Non-finite training metric')
            for checkpoint in progress['checkpoints']:
                path=checkpoint['path']
                for entry in checkpoint['state'].get('log_history',[]):
                    assert all(math.isfinite(float(entry[k])) for k in ('loss','grad_norm') if k in entry)
                # Preserve the first completed checkpoint independently while training continues.
                if not checkpoint_started:
                    cpout=out/'checkpoint_backup';cpout.mkdir(exist_ok=True)
                    log=open(cpout/'worker.log','a',encoding='utf-8')
                    flags={'creationflags':subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP} if os.name=='nt' else {'start_new_session':True}
                    worker=subprocess.Popen([sys.executable,'scratch/nonmoral/result_backup.py','--host',host,
                                             '--checkpoint',path,'--out',str(cpout/'files')],cwd=ROOT,
                                             stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,**flags)
                    log.close();checkpoint_started.add(path);state['checkpoint_backup_pid']=worker.pid
            save()
            if progress['logs']['train']['exit']==0:
                info=hf_api().model_info(arm['organism'])
                meta=json.loads(Path(hf_download(arm['organism'],'training_meta.json',revision=info.sha)).read_text())
                assert meta['dataset']['revision']==arm['data_revision']
                assert meta['base_model_revision']==plan['base_model_revision'] and meta['thinking'] is True
                assert any(s.rfilename.endswith('.safetensors') for s in info.siblings)
                state.update(phase='published',adapter_revision=info.sha,training_complete=True);save();break
            if not progress['driver_alive']: raise RuntimeError('Driver exited without successful train receipt')
            if time.time()-changed>1200: raise RuntimeError('Training made no log progress for 20 minutes')
            print(json.dumps({'phase':state['phase'],'pod':state['owned_pod'],'elapsed_s':round(state['elapsed_s']),
                              'usd':round(state['estimated_usd'],2),'gpu':progress['gpu']}),flush=True)
            time.sleep(30)
        else:
            raise RuntimeError('Training window exhausted; recovery reserve begins')
    except BaseException as exc:
        state.update(phase='failed',failure=f'{type(exc).__name__}: {exc}');save()
        response=getattr(exc,'response',None)
        if response is not None:
            try:
                payload=response.json()
                state['provider_error']={k:payload[k] for k in ('error','message','statusCode') if k in payload}
                save();print(json.dumps(state['provider_error']),flush=True)
            except (ValueError,TypeError):
                pass
        print(state['failure'],flush=True)
    finally:
        if state['owned_pod']:
            if remote and state.get('training_started'):
                deadline=min(created+seconds-60,time.time()+reserve)
                try:
                    # The process group belongs to this fresh pod and was recorded at launch.
                    remote._ssh('p=/root/work/output/da-supervision/driver.pid; g=$(cat "$p"); '
                                'case "$g" in ""|*[!0-9]*) exit 1;; esac; '
                                'if kill -0 -- -"$g" 2>/dev/null; then kill -STOP -- -"$g"; fi',timeout=30)
                    logs=remote._ssh('python3 -c '+shlex.quote("import json; from pathlib import Path; p=Path('/root/work/output/da-supervision'); print(json.dumps({f.name:f.read_text(errors='replace') for f in p.glob('*.log')}))"),timeout=60)
                    dump(out/'complete_logs.json',json.loads(logs))
                    expected=[dict(arm,base_model_revision=plan['base_model_revision'])] if state.get('training_complete') else []
                    state['phase']='preserving';save()
                    state['local_backup']=fetch_training_outputs(remote,out,expected,
                        timeout=max(60,int(deadline-time.time())),include_roots=['output/train'],archive_name='da-supervision-backup.tar')
                    dump(out/'local_backup.json',state['local_backup'])
                except Exception as exc:
                    state['backup_error']=f'{type(exc).__name__}: {exc}';save()
            # Verified HF adapter is durable even if optional local checkpoint backup fails.
            safe=not state.get('training_started') or state.get('local_backup',{}).get('verified') or state.get('adapter_revision')
            if safe:
                runpod.teardown(state['owned_pod'])
                state.update(terminated=True,phase='complete' if state.get('training_complete') else 'failed_terminated')
            else:
                state['phase']='recovery_required';save()
                # Keep the owner alive under the unchanged hard deadline for manual recovery.
                while time.time()<created+seconds: time.sleep(min(30,created+seconds-time.time()))
            save()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('plan');parser.add_argument('arm');parser.add_argument('--out',required=True)
    args=parser.parse_args();run(args.plan,args.arm,args.out)
