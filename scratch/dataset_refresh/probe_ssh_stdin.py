# ABOUTME: Diagnose Windows concurrent SSH stdin contention without touching the trainer.
# ABOUTME: Run hidden with python scratch/dataset_refresh/probe_ssh_stdin.py; only remote sleep/printf commands.
import json
import subprocess
import time
from pathlib import Path

from src.infra.endpoints.vllm import SshExec, ssh_argv

root = Path('output/lowstakes_practical_training')
state = json.loads((root/'run/status.json').read_text())
remote = SshExec(state['host'], port=8000, workdir='/root/work')
argv, target = ssh_argv(remote.host)
results = []
for detach_stdin in (False, True):
    with (root/'ssh_probe_transfer.log').open('wb') as stream:
        options = {'stdin': subprocess.DEVNULL} if detach_stdin else {}
        child = subprocess.Popen([*argv, target, 'sleep 10'], stdout=stream,
            stderr=subprocess.DEVNULL, **options)
        time.sleep(1)
        start = time.monotonic()
        try:
            result = remote._ssh('printf healthy', timeout=5)
        except Exception as exc:
            result = type(exc).__name__
        results.append(dict(detach_stdin=detach_stdin, seconds=time.monotonic()-start, result=result))
        child.wait(timeout=20)
(root/'ssh_stdin_probe.json').write_text(json.dumps(results, indent=2))
print(json.dumps(results))
