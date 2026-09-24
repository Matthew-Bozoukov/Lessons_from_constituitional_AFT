# ABOUTME: Submit to one receipted Vast CPU with an explicit persistent or expiring lifetime.
# ABOUTME: Ownership and SSH identity gate submission; provisioning is documented in the repository skill.
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def receipt_deadline(receipt):
    """None is unlimited only with explicit persistent authorization, never a missing expiry."""
    if receipt.get('lifetime') == 'persistent':
        assert receipt.get('stop_at') is None, 'Persistent receipt must not contain an old stop deadline'
        assert receipt.get('lifetime_authorized_at'), 'Persistent CPU needs an authorization record'
        return None
    assert receipt.get('stop_at'), 'CPU receipt needs an explicit lifetime'
    expiry = datetime.fromisoformat(receipt['stop_at'])
    assert expiry.tzinfo is not None, 'CPU expiry must include a timezone'
    return expiry.timestamp()


def deadline_value(value):
    """Use infinity for arithmetic only; durable JSON stores unlimited lifetime as null."""
    return float('inf') if value is None else float(value)


def watchdog_deadline(receipt):
    deadline = receipt_deadline(receipt)
    if receipt.get('boot_deadline') and not receipt.get('ssh_verified'):
        return min(deadline_value(deadline), datetime.fromisoformat(receipt['boot_deadline']).timestamp())
    return deadline


def ssh_endpoint(instance):
    # VM proxy SSH can be refused even when its mapped direct port is healthy.
    ports = (instance.get('ports') or {}).get('22/tcp') or []
    if instance.get('public_ipaddr') and ports:
        return instance['public_ipaddr'], int(ports[0]['HostPort'])
    assert instance.get('ssh_host') and instance.get('ssh_port'), 'CPU SSH endpoint unavailable'
    return instance['ssh_host'], int(instance['ssh_port'])


def check_instance(receipt, instance, now=None):
    now = datetime.now(timezone.utc) if now is None else now
    assert instance and int(instance['id']) == int(receipt['instance_id']), 'Receipted CPU missing'
    assert instance['label'] == receipt['label'], 'CPU ownership changed'
    remaining = deadline_value(receipt_deadline(receipt)) - now.timestamp()
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
    host, port = ssh_endpoint(instance)
    ssh = ['ssh', '-i', str(Path(identity).resolve()), '-p', str(port),
           '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=yes',
           'root@' + host]
    # Known-host validation is deliberately retained when an address changes.
    expected = json.dumps({k: receipt.get(k) for k in ('instance_id', 'label', 'stop_at', 'lifetime', 'lifetime_authorized_at')})
    check = ('import json; from pathlib import Path; r=json.loads(Path("/srv/lasr/receipt.json").read_text()); '
             'e=json.loads(' + repr(expected) + '); assert all(r.get(k)==v for k,v in e.items()), "CPU receipt mismatch"')
    python = '/srv/lasr/repo/scratch/swebench_cpu_env/.venv/bin/python'
    subprocess.run(ssh + [shlex.join([python, '-c', check])], check=True, timeout=45)
    remote = 'cd /srv/lasr/repo && ' + shlex.join([python, '-m', 'src.eval.capabilities.swebench_mini.fleet', *command])
    subprocess.run(ssh + [remote], check=True, timeout=600)
    receipt.update(ssh_host=host, ssh_port=port)
    temporary = receipt_path.with_suffix(receipt_path.suffix + '.tmp')
    temporary.write_text(json.dumps(receipt, indent=2) + '\n')
    temporary.replace(receipt_path)
