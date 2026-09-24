# ABOUTME: Cross-checkout CPU registry and explicit persistent/expiring lifetime management.
# ABOUTME: Never creates or destroys a rental; the repository SWE-bench skill owns cold provisioning.
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess

from src.eval.capabilities.swebench_mini.fleet_host import receipt_deadline, ssh_endpoint


REGISTRY = Path.home() / '.lasr/swebench/cpu.json'


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def register(path, receipt_path, identity):
    receipt = json.loads(Path(receipt_path).read_text())
    receipt_deadline(receipt)
    assert Path(identity).is_file(), 'SSH identity missing'
    if Path(path).exists():
        old = json.loads(Path(path).read_text())
        assert old.get('instance_id') in (None, receipt['instance_id']), 'Another CPU is registered; reconcile it first'
    atomic(path, {'state': 'registered', 'instance_id': receipt['instance_id'],
                  'receipt': str(Path(receipt_path).resolve()), 'identity': str(Path(identity).resolve())})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['status', 'register', 'persistent', 'expire'])
    parser.add_argument('--registry', type=Path, default=REGISTRY)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--identity', type=Path)
    parser.add_argument('--stop-at', help='Timezone-aware ISO UTC deadline for expire')
    args = parser.parse_args()
    if args.action == 'register':
        assert args.receipt and args.identity
        register(args.registry, args.receipt, args.identity)
    if not args.registry.exists():
        print(json.dumps({'state': 'empty', 'registry': str(args.registry)}))
        return
    entry = json.loads(args.registry.read_text())
    if entry.get('state') == 'empty':
        assert args.action == 'status', 'Prepare a CPU first using the repository skill'
        print(json.dumps(entry))
        return
    from vastai import VastAI
    client = VastAI(api_key=os.environ['VAST_API_KEY'].strip())
    receipt_path = Path(entry['receipt'])
    receipt = json.loads(receipt_path.read_text())
    instance = client.show_instance(int(entry['instance_id']))
    assert instance and int(instance['id']) == int(receipt['instance_id']), 'Registered CPU missing; reconcile before replacement'
    assert instance['label'] == receipt['label'], 'Ownership mismatch'
    if args.action in ('persistent', 'expire'):
        assert instance['actual_status'] == 'running', 'Resume and verify SSH before changing lifetime'
        updated = dict(receipt)
        updated.update(lifetime='persistent' if args.action == 'persistent' else 'expiring',
                       stop_at=None if args.action == 'persistent' else args.stop_at,
                       lifetime_authorized_at=datetime.now(timezone.utc).isoformat())
        deadline = receipt_deadline(updated)
        if deadline is not None:
            assert deadline > datetime.now(timezone.utc).timestamp(), 'New expiry must be in the future'
        host, port = ssh_endpoint(instance)
        ssh = ['ssh', '-i', entry['identity'], '-p', str(port), '-o', 'ConnectTimeout=10',
               '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', 'root@' + host]
        # Refuse mutations during a model run. The updated watchdog must be deployed first.
        script = ('import json,sys,subprocess; from pathlib import Path; '
                  'from src.eval.capabilities.swebench_mini.fleet_host import receipt_deadline; '
                  'r=json.load(sys.stdin); receipt_deadline(r); '
                  'p=Path("/srv/lasr/receipt.json"); old=json.loads(p.read_text()); '
                  'assert old["instance_id"]==r["instance_id"] and old["label"]==r["label"]; '
                  'assert subprocess.run(["systemctl","is-active","--quiet","lasr-swebench-lite.service"]).returncode!=0; '
                  'archive=p.with_name("receipt-before-lifetime-"+str(__import__("time").time_ns())+".json"); '
                  'archive.write_text(p.read_text()); t=p.with_suffix(".tmp"); '
                  't.write_text(json.dumps(r,indent=2)); t.chmod(0o600); t.replace(p)')
        command = 'cd /srv/lasr/repo && ' + shlex.join(['scratch/swebench_cpu_env/.venv/bin/python', '-c', script])
        subprocess.run(ssh + [command], input=json.dumps(updated), text=True, check=True, timeout=45)
        atomic(receipt_path.with_name('receipt-before-lifetime-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '.json'), receipt)
        atomic(receipt_path, updated)
        receipt = updated
    print(json.dumps({'state': instance['actual_status'], 'instance_id': instance['id'],
                      'label': instance['label'], 'lifetime': receipt.get('lifetime', 'expiring'),
                      'stop_at': receipt['stop_at'], 'ssh_host': ssh_endpoint(instance)[0],
                      'ssh_port': ssh_endpoint(instance)[1], 'hourly_usd': instance.get('dph_total'),
                      'receipt': str(receipt_path)}, indent=2))


if __name__ == '__main__':
    main()
