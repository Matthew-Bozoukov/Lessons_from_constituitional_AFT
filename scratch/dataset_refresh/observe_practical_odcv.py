# ABOUTME: Read-only status and health observer for the practical low-stakes ODCV owner.
# ABOUTME: Reports native transcript counts, local Docker usage and serving health without restarting work.
import hashlib
import json
import subprocess
import time
from pathlib import Path
import requests
from src.infra.endpoints.vllm import SshExec
from src.infra.runpod import default_keypair, _parent_alive

ROOT = Path('output/odcv_practical_20260922')
STATE = ROOT/'practical/broader_eval_status.json'

def main():
    s = json.loads(STATE.read_text(encoding='utf-8'))
    previous = json.loads((ROOT/'observer.json').read_text()) if (ROOT/'observer.json').exists() else {}
    roots = list(Path('C:/odcv-practical').glob('*_odcv_refresh_practical_20260922_*'))
    if len(roots)>1:
        raise RuntimeError('Multiple run roots require inspection')
    records = []
    judged = 0
    if roots:
        root=roots[0]
        published=list((root/'rollouts').rglob('messages_record.txt'))
        records=published or [p for p in root.rglob('messages_record.txt') if 'agent_logs' in p.parts and 'combined' not in str(p)]
        ledger=ROOT/'practical/judge_ledger.json'
        if ledger.exists():
            judged=sum(e.get('status')=='settled' for e in json.loads(ledger.read_text()))
    out={k:s.get(k) for k in ('phase','pod_id','host','estimated_gpu_and_storage_usd','judge_charged_or_reserved_usd','termination_verified','error','gpu_preflight')}
    out.update(owner_alive=_parent_alive(s['pid']),native_transcripts=len(records),judge_calls_settled=judged,run_root=str(roots[0]) if roots else None,observed_epoch=time.time())
    if s.get('phase')=='evaluating' and not s.get('termination_verified'):
        try:
            out['endpoint_healthy']=requests.get('http://127.0.0.1:18124/health',timeout=5).status_code==200
        except requests.RequestException:
            out['endpoint_healthy']=False
    if time.time()-previous.get('resource_check_epoch',0)>240:
        tag=hashlib.md5(b'qwen36_0_da_lowstakes_practical_7').hexdigest()[:6]
        r=subprocess.run(['docker','stats','--no-stream','--format','{{json .}}'],capture_output=True,text=True,timeout=30)
        stats=[json.loads(line) for line in r.stdout.splitlines() if line.strip()]
        out['docker_stats']=[x for x in stats if f'odcv-{tag}-' in x.get('Name','')]
        out['docker_containers_total']=len(stats)
        out['resource_check_epoch']=time.time()
        if s.get('host') and not s.get('termination_verified'):
            try:
                remote=SshExec(s['host'],port=18124,identity=default_keypair()[1])
                out['gpu']=remote._ssh('nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits',timeout=15).strip()
            except Exception as e:
                out['gpu_probe_error']=type(e).__name__
    else:
        for k in ('resource_check_epoch','docker_stats','docker_containers_total','gpu'):
            if k in previous: out[k]=previous[k]
    if s.get('termination_verified'):
        out['gpu'] = 'terminated'
    (ROOT/'observer.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({k:v for k,v in out.items() if k not in ('docker_stats','gpu_preflight')}))

if __name__=='__main__':
    main()
