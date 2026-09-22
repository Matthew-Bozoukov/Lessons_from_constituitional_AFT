# ABOUTME: Read-only live health snapshot with direct SSH fallback when owner polling is stale.
# ABOUTME: Run: uv run --no-sync python -m scratch.dataset_refresh.observe_practical_training
import json
import re
import shlex
import time
from pathlib import Path

from src.infra.endpoints.vllm import SshExec


def main():
    root = Path('output/lowstakes_practical_training')
    state = json.loads((root/'run/status.json').read_text())
    progress = state.get('progress', {})
    tail = progress.get('arms', [{}])[0].get('tail', '')
    gpu = progress.get('gpu')
    stale = time.time() - state.get('created_epoch', time.time()) - state.get('elapsed_s', 0)
    direct = False
    if stale > 90 and not state.get('terminated') and state.get('host'):
        remote = SshExec(state['host'], port=8000, workdir='/root/work')
        probe = """import json,subprocess
from pathlib import Path
p=Path('/root/work/output/nonmoral-paired-supervision/arm_0.log')
print(json.dumps({'tail':p.read_text(errors='replace')[-4500:],
'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu,temperature.gpu','--format=csv,noheader'],text=True).strip()}))
"""
        live = json.loads(remote._ssh('python3 -c '+shlex.quote(probe), timeout=20))
        tail, gpu, direct = live['tail'], live['gpu'], True
    steps = re.findall(r'(\d+)/625', tail)
    elapsed = time.time() - state['created_epoch']
    result = dict(phase=state['phase'], step=int(steps[-1]) if steps else None,
        gpu=gpu, minutes=round(elapsed/60, 1),
        estimated_usd=round(elapsed/3600*state.get('budget_hourly_usd', 5), 2),
        backups=len(state.get('checkpoint_backups', [])),
        terminated=state.get('terminated', False), failure=state.get('failure'),
        direct_health_check=direct, archive_published=(root/'final_release_receipt.json').exists())
    if re.search(r"'(?:loss|grad_norm)':\s*'?[-+]?(?:nan|inf)", tail, re.I):
        result['failure'] = 'Nonfinite optimization report requires inspection'
    (root/'live_observer_status.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
