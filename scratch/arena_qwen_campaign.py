# ABOUTME: Operate the approved October Qwen DA15/plain Arena campaign on one H200.
# ABOUTME: Uses the standard provisioner, eval entrypoint, pinned configs and explicit receipts.
import ctypes
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
from src.infra import runpod

ROOT = Path('output/arena_qwen_da15_20261012')
CONTROL = 'dougalldeepmind/2026-10-05-qwen36-0-plain'
DA = 'dougalldeepmind/2026-10-05-qwen36-0-da-15'


def save(name, data):
    temp = ROOT / (name + '.tmp')
    temp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temp.replace(ROOT / name)


def main(action):
    load_dotenv('.env')
    ROOT.mkdir(parents=True, exist_ok=True)
    if os.name == 'nt':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    if action == 'provision':
        if (ROOT / 'pod.json').exists():
            raise RuntimeError('Pod receipt already exists; inspect it before allocating again')
        def registered(pod_id):
            save('pod.json', {'id': pod_id, 'created_utc': datetime.now(timezone.utc).isoformat(),
                              'gpu': 'NVIDIA H200', 'max_hours': 6})
        try:
            result = runpod.up(name='nika-arena-qwen-da15-20261012', eval='arena_hard',
                               target=[CONTROL, DA], gpu='NVIDIA H200', count=1,
                               max_hours=6, on_provisioned=registered)
            print(result, flush=True)
            receipt = json.loads((ROOT / 'pod.json').read_text(encoding='utf-8'))
            receipt['host'] = re.search(r'^host:\s+(\S+)', result, re.M).group(1)
            info = runpod.call('GET', '/pods/' + receipt['id'])
            receipt['hourly_usd'] = info.get('costPerHr')
            save('pod.json', receipt)
        except BaseException:
            if (ROOT / 'pod.json').exists():
                receipt = json.loads((ROOT / 'pod.json').read_text(encoding='utf-8'))
                runpod.teardown(receipt['id'])
            raise
        return
    if action not in ('smoke', 'full'):
        raise ValueError(action)
    receipt = json.loads((ROOT / 'pod.json').read_text(encoding='utf-8'))
    command = [sys.executable, '-u', 'scripts/run_eval.py', '--name', 'arena_hard',
               '--config', str(ROOT / f'{action}.yaml'), '--target', DA, '--reference', CONTROL,
               '--server', receipt['host'], '--port', '18943']
    command += ['--no-push'] if action == 'smoke' else ['--terminate-pod']
    save(action + '_launch.json', {'command': command, 'started_utc': datetime.now(timezone.utc).isoformat()})
    code = 1
    try:
        code = subprocess.call(command, env=os.environ | {'PYTHONUTF8': '1', 'PYTHONUNBUFFERED': '1'})
    finally:
        if action == 'full' or code != 0:
            if any(p['id'] == receipt['id'] for p in runpod.active_pods()):
                runpod.teardown(receipt['id'])
        save(action + '_exit.json', {'exit_code': code, 'finished_utc': datetime.now(timezone.utc).isoformat()})
    raise SystemExit(code)


if __name__ == '__main__':
    main(sys.argv[1])
