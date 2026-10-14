# ABOUTME: Owned local Docker launcher and Windows resource/awake guard for GPT-OSS MMLU.
# ABOUTME: Run: uv run --project scratch/swebench_cpu_env python scratch/gptoss_mmlu/host.py start|guard
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path.cwd()
OUT = ROOT / 'output/gptoss_mmlu_20261014'
NAME = 'lasr-gptoss-mmlu-20261014'

def guard():
    import psutil
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        while not (OUT/'complete.json').exists() and not (OUT/'stopped.json').exists():
            value = dict(at=time.time(), available_ram=psutil.virtual_memory().available,
                         disk_free=psutil.disk_usage(str(ROOT)).free, pid=os.getpid())
            tmp = OUT/'host-resources.tmp'
            tmp.write_text(json.dumps(value))
            tmp.replace(OUT/'host-resources.json')
            subprocess.run(['docker','ps','--filter',f'name=^{NAME}$','--format','{{.Names}}'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
            time.sleep(10)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

def start():
    from dotenv import dotenv_values
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    source = dotenv_values(Path.home()/'source/repos/LASR/teaching_claude_why_replication/.env')
    for key in ('TINKER_API_KEY','HF_TOKEN','HUGGINGFACE_API_KEY'):
        env.pop(key, None)
        if source.get(key): env[key] = source[key]
    assert env.get('TINKER_API_KEY') and (env.get('HF_TOKEN') or env.get('HUGGINGFACE_API_KEY'))
    args = ['docker','run','-d','--name',NAME,'--label','lasr.campaign=gptoss-mmlu-20261014',
            '--cpus','6','--memory','4g','--memory-swap','4g','--pids-limit','1024',
            '-v',f'{ROOT}:/work','-v','lasr-gptoss-mmlu-cpu-env:/work/scratch/swebench_cpu_env/.venv',
            '-v','lasr-gptoss-mmlu-shim-env:/work/src/infra/endpoints/tinker_env/.venv',
            '-v','lasr-gptoss-hf-cache:/root/.cache/huggingface',
            '-v','lasr-gptoss-uv-cache:/root/.cache/uv',
            '-e','HF_ORG=dougalldeepmind','-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1',
            '-e','TOKENIZERS_PARALLELISM=false']
    for key in ('TINKER_API_KEY','HF_TOKEN','HUGGINGFACE_API_KEY'):
        if env.get(key): args += ['-e',key]
    args += ['lasr-gptoss-shared-smoke:20261008','sleep','infinity']
    cid = subprocess.check_output(args, env=env, text=True).strip()
    (OUT/'container.json').write_text(json.dumps(dict(name=NAME,id=cid,at=time.time())))
    print(json.dumps(dict(name=NAME,id=cid)))

if __name__ == '__main__':
    {'start':start,'guard':guard}[sys.argv[1]]()
