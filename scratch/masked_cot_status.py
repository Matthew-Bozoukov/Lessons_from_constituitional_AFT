# ABOUTME: Read-only status for the October 5 full-CoT-masked training campaign.
# ABOUTME: Saves compact remote progress and checks checkpoint loss/gradient finiteness.
import json
import shlex
from pathlib import Path

from src.infra.endpoints.vllm import SshExec

manifest = json.loads(Path('output/masked_cot_campaign/campaign.json').read_text())
remote = r'''
from pathlib import Path
import json, math, subprocess
root=Path('/root/work/output/masked_cot_campaign')
log=(root/'training.log').read_text(errors='replace')
lines=log.replace('\r','\n').splitlines()
key=[x for x in lines if x.startswith('>>>')]
progress=[x for x in lines if '/608' in x]
states=list(Path('/root/work/output/train').glob('*/checkpoint-*/trainer_state.json'))
state={}; metrics=[]
if states:
 p=max(states,key=lambda p:int(p.parent.name.split('-')[-1]))
 state=json.loads(p.read_text()); metrics=state.get('log_history',[])
 assert all(math.isfinite(float(r[k])) for r in metrics for k in ['loss','grad_norm'] if k in r), 'nonfinite training metric'
out={'key_lines':key[-10:],'progress':progress[-1:] or lines[-3:],
     'exit_code':(root/'training.exit').read_text().strip() if (root/'training.exit').exists() else None,
     'checkpoint_step':state.get('global_step'),'latest_metrics':metrics[-1:],
     'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()}
print(json.dumps(out))
'''
ssh = SshExec(manifest['host'], port=8000, workdir='/root/work')
result = json.loads(ssh._ssh('/root/work/.venv/bin/python -c ' + shlex.quote(remote), timeout=45))
Path('output/masked_cot_campaign/latest_status.json').write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != 'key_lines'}, indent=2))
