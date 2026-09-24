# ABOUTME: Submit to exactly one receipted Vast CPU; provider identity and existing expiry gate every operation.
# ABOUTME: No GPU creation, CPU creation, expiry extension, or credentials transfer happens here.
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def check_instance(receipt, instance, now=None):
    now = datetime.now(timezone.utc) if now is None else now
    assert instance and int(instance['id']) == int(receipt['instance_id']), 'Receipted CPU missing'
    assert instance['label'] == receipt['label'], 'CPU ownership changed'
    remaining = (datetime.fromisoformat(receipt['stop_at']) - now).total_seconds()
    # Fine-grained task/boot/grading admission uses the selected config remotely.
    assert remaining > 120, 'CPU lifetime expired or cannot accept a launch'
    assert instance['actual_status'] in ('running', 'stopped', 'loading', 'created'), 'Unexpected CPU status'


def submit(receipt_path, identity, command):
    from vastai import VastAI
    assert identity and Path(identity).is_file(), 'Use --cpu-key for the prepared CPU SSH identity'
    receipt_path = Path(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    client = VastAI(api_key=os.environ['VAST_API_KEY'].strip())
    instance = client.show_instance(int(receipt['instance_id']))
    check_instance(receipt, instance)
    if instance['actual_status'] == 'stopped':
        assert client.start_instance(instance['id']).get('success'), 'CPU resume rejected'
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        instance = client.show_instance(int(receipt['instance_id']))
        check_instance(receipt, instance)
        if instance['actual_status'] == 'running' and instance.get('ssh_host') and instance.get('ssh_port'):
            break
        time.sleep(10)
    else:
        raise TimeoutError('CPU did not become available; no inference GPUs were requested')
    ssh = ['ssh', '-i', str(Path(identity).resolve()), '-p', str(instance['ssh_port']),
           '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=yes',
           'root@' + instance['ssh_host']]
    # Known-host validation is deliberately retained when an address changes.
    expected = json.dumps({'instance_id': receipt['instance_id'], 'label': receipt['label'], 'stop_at': receipt['stop_at']})
    check = ('import json; from pathlib import Path; r=json.loads(Path("/srv/lasr/receipt.json").read_text()); '
             'e=json.loads(' + repr(expected) + '); assert all(r[k]==v for k,v in e.items()), "CPU receipt mismatch"')
    python = '/srv/lasr/repo/scratch/swebench_cpu_env/.venv/bin/python'
    subprocess.run(ssh + [shlex.join([python, '-c', check])], check=True, timeout=45)
    remote = 'cd /srv/lasr/repo && ' + shlex.join([python, '-m', 'src.eval.capabilities.swebench_mini.fleet', *command])
    subprocess.run(ssh + [remote], check=True, timeout=600)
    receipt.update(ssh_host=instance['ssh_host'], ssh_port=instance['ssh_port'])
    temporary = receipt_path.with_suffix(receipt_path.suffix + '.tmp')
    temporary.write_text(json.dumps(receipt, indent=2) + '\n')
    temporary.replace(receipt_path)
