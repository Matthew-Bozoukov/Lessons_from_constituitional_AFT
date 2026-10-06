# ABOUTME: Stage committed code and allowlisted credentials onto one already-provisioned receipted Vast CPU VM.
# ABOUTME: Strict SSH ownership/resource gates only; no creation, resume, bootstrap, Docker install or GPU rental.
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import ipaddress
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
REQUIRED = ('HF_TOKEN', 'HF_ORG', 'USER_PREFIX', 'VAST_API_KEY', 'RUNPOD_API_KEY',
            'DOCKERHUB_USERNAME', 'DOCKERHUB_TOKEN')


class StageError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise StageError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{time.time_ns()}.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def endpoint(receipt, instance):
    require(instance and int(instance['id']) == int(receipt['instance_id']), 'Receipted VM is missing')
    require(instance.get('label') == receipt['label'], 'Provider label does not match receipt')
    require(instance.get('actual_status') == 'running', 'Receipted VM must already be running')
    mappings = (instance.get('ports') or {}).get('22/tcp') or []
    ports = {int(mapping['HostPort']) for mapping in mappings}
    require(instance.get('public_ipaddr') and len(ports) == 1, 'Require one unambiguous direct public SSH port')
    host = str(ipaddress.ip_address(instance['public_ipaddr']))
    port = ports.pop()
    require(0 < port < 65536, 'Invalid direct SSH port')
    return host, port


def credentials(env_path):
    from dotenv import dotenv_values
    with redirect_stderr(io.StringIO()):
        values = dotenv_values(env_path, interpolate=False)
    missing = [key for key in REQUIRED if not (values.get(key) or '').strip()]
    require(not missing, 'Missing required env names: ' + ', '.join(missing))
    selected = {key: values[key].strip() for key in REQUIRED}
    require(all(not any(c in value for c in '\r\n\0') for value in selected.values()), 'Multiline/NUL credential values are unsupported')
    return selected


def committed_archive(path, code_sha):
    require(bool(re.fullmatch('[0-9a-f]{40}', code_sha)), 'Require full committed Git SHA')
    # Compare bytes with git's own archive; a manually modified tar or an untracked
    # .env addition cannot pass by merely copying the Git pax comment.
    proc = subprocess.Popen(['git', 'archive', '--format=tar', code_sha], cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    expected = hashlib.file_digest(proc.stdout, 'sha256').hexdigest()
    proc.stdout.close()
    require(proc.wait() == 0, 'Cannot archive the requested local commit')
    require(digest(path) == expected, 'Archive differs from git archive --format=tar of --code-sha')
    with tarfile.open(path, 'r:') as archive:
        for member in archive.getmembers():
            name = Path(member.name).name
            require(name not in ('.env', 'credentials.env') and not name.endswith(('.key', '.env')),
                    'Committed archive contains a secret-like filename; inspect it before staging')
    return expected


PROBE = r'''
import json,os,shutil,subprocess,sys
from pathlib import Path
expected=json.load(sys.stdin)
def check(ok,message):
 if not ok: raise RuntimeError(message)
check(Path('/proc/1/comm').read_text().strip()=='systemd','VM PID 1 is not systemd')
container=subprocess.run(['systemd-detect-virt','--container'],capture_output=True,text=True)
check(container.returncode==1 and container.stdout.strip()=='none','Container detected; require native Docker VM')
receipt=Path('/srv/lasr/receipt.json')
if receipt.exists():
 old=json.loads(receipt.read_text());check(all(old.get(k)==expected[k] for k in ('instance_id','label')),'Remote receipt ownership mismatch')
for service in ('lasr-swebench-lite.service','lasr-plain-swe-queue.service','lasr-swebench-plain-queue.service'):
 s=subprocess.run(['systemctl','show',service,'--property=ActiveState','--value'],capture_output=True,text=True)
 check(s.returncode in (0,1,4) and s.stdout.strip() in ('inactive','failed',''),'Inference or wave queue service is active')
check(shutil.which('docker') is not None,'Native Docker is absent; install/prepare it explicitly before staging')
docker=subprocess.run(['docker','--host','unix:///var/run/docker.sock','info','--format','{{json .}}'],capture_output=True,text=True,timeout=30)
check(docker.returncode==0,'Native Docker daemon is not usable; prepare it explicitly before staging')
info=json.loads(docker.stdout)
check(subprocess.run(['systemctl','is-active','--quiet','docker.service']).returncode==0,'Docker is not managed by the VM systemd')
cpus=float(len(os.sched_getaffinity(0)))
quota=Path('/sys/fs/cgroup/cpu.max')
if quota.exists():
 q,p=quota.read_text().split()
 if q!='max': cpus=min(cpus,int(q)/int(p))
else:
 q=Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us');p=Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us')
 if q.exists() and p.exists() and int(q.read_text())>0: cpus=min(cpus,int(q.read_text())/int(p.read_text()))
mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
disk=shutil.disk_usage('/srv')
check(cpus>=60,'Fewer than 60 effective CPUs')
check(mem['MemTotal']>=190*2**30,'Less than 190 GiB actual RAM')
check(disk.total>=480*2**30,'Less than 480 GiB staging filesystem')
check(disk.free>=50*2**30,'Less than 50 GiB free disk')
repo=Path('/srv/lasr/repo');deployment=Path('/srv/lasr/lite-deployment.json')
if repo.exists():
 check(deployment.exists() and json.loads(deployment.read_text()).get('code_sha')==expected['code_sha'],'Different or unreceipted deployment exists; use explicit quiescent redeployment')
print(json.dumps(dict(effective_cpus=cpus,memory_bytes=mem['MemTotal'],available_memory_bytes=mem['MemAvailable'],filesystem_bytes=disk.total,free_bytes=disk.free,systemd=True,native_docker=True,docker_version=info.get('ServerVersion'),docker_root=info.get('DockerRootDir'))))
'''


INSTALL = r'''
import hashlib,json,os,shlex,sys,tarfile,tempfile,time
from pathlib import Path,PurePosixPath
os.umask(0o077)
p=json.load(sys.stdin);base=Path('/srv/lasr');archive=base/'staging'/p['archive_name']
def check(ok,message):
 if not ok: raise RuntimeError(message)
with archive.open('rb') as f:
 h=hashlib.sha256()
 while chunk:=f.read(4*1024*1024):h.update(chunk)
check(h.hexdigest()==p['archive_sha256'],'Transferred archive hash mismatch')
repo=base/'repo';deployment=base/'lite-deployment.json'
if repo.exists():
 check(deployment.exists() and json.loads(deployment.read_text()).get('code_sha')==p['code_sha'],'Existing deployment changed')
else:
 stage=Path(tempfile.mkdtemp(prefix='code-',dir=base/'staging'))
 with tarfile.open(archive,'r:') as tar:
  members=tar.getmembers();links={m.name.rstrip('/') for m in members if m.issym()}
  for m in members:
   rel=PurePosixPath(m.name)
   check(not rel.is_absolute() and '..' not in rel.parts,'Unsafe archive member')
   check(not any(str(parent) in links for parent in rel.parents),'Archive member traverses symlink')
   check(m.isfile() or m.isdir() or m.issym(),'Unsupported archive member type')
   target=stage/m.name
   if m.issym():check((target.parent/m.linkname).resolve().is_relative_to(stage.resolve()),'Archive symlink escapes staging')
  # Member types, paths and every symlink ancestor have already been checked.
  tar.extractall(stage)
 stage.rename(repo)
def write(path,value):
 temp=path.with_name(path.name+'.'+str(time.time_ns())+'.tmp')
 with temp.open('w') as f:f.write(value);f.flush();os.fsync(f.fileno())
 temp.chmod(0o600);temp.replace(path)
old=base/'receipt.json'
if old.exists():
 previous=json.loads(old.read_text())
 check(all(previous.get(k)==p['receipt'][k] for k in ('instance_id','label')),'Remote receipt ownership changed')
 write(base/('receipt-before-stage-'+str(time.time_ns())+'.json'),old.read_text())
write(base/'credentials.env',''.join(k+'='+shlex.quote(v)+'\n' for k,v in p['credentials'].items()))
write(base/'resource-probe.json',json.dumps(p['probe'],indent=2)+'\n')
write(deployment,json.dumps(p['deployment'],indent=2)+'\n')
write(old,json.dumps(p['receipt'],indent=2)+'\n')
check(oct((base/'credentials.env').stat().st_mode & 0o777)=='0o600','Credential file permissions differ')
print(json.dumps(dict(staged=True,instance_id=p['receipt']['instance_id'],code_sha=p['code_sha'],archive_sha256=p['archive_sha256'],ssh_verified=p['receipt']['ssh_verified'])))
'''


def ssh_call(ssh, script, payload, *, timeout=120):
    result = subprocess.run([*ssh, 'python3 -c ' + shlex.quote(script)], input=json.dumps(payload).encode(),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if result.returncode:
        # Remote tracebacks may include payload values. Never relay raw stderr.
        for message in ('Native Docker is absent; install/prepare it explicitly before staging',
                        'Native Docker daemon is not usable; prepare it explicitly before staging',
                        'VM PID 1 is not systemd', 'Container detected; require native Docker VM',
                        'Fewer than 60 effective CPUs', 'Less than 190 GiB actual RAM',
                        'Less than 480 GiB staging filesystem', 'Less than 50 GiB free disk',
                        'Different or unreceipted deployment exists; use explicit quiescent redeployment'):
            if message.encode() in result.stderr:
                raise StageError(message)
        raise StageError('Remote staging/probe rejected; no raw output printed. Check VM systemd, native Docker, resources and existing deployment.')
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser()
    for key in ('receipt', 'identity', 'env', 'archive', 'code-sha'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    receipt_path = Path(args.receipt).resolve()
    receipt = json.loads(receipt_path.read_text())
    from src.eval.capabilities.swebench_mini.fleet_host import receipt_deadline
    deadline = receipt_deadline(receipt)
    require(deadline is None or deadline > time.time() + 120, 'CPU lifetime expired or too short')
    require(Path(args.identity).is_file(), 'SSH identity missing')
    values = credentials(args.env)
    archive_sha = committed_archive(args.archive, args.code_sha)
    from vastai import VastAI
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        try:
            instance = VastAI(api_key=values['VAST_API_KEY']).show_instance(int(receipt['instance_id']))
        except Exception:
            raise StageError('Provider read failed; no resources changed') from None
    host, port = endpoint(receipt, instance)
    ssh = ['ssh', '-i', str(Path(args.identity).resolve()), '-p', str(port), '-o', 'BatchMode=yes',
           '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15',
           '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3', 'root@' + host]
    probe = ssh_call(ssh, PROBE, {'instance_id': receipt['instance_id'], 'label': receipt['label'], 'code_sha': args.code_sha})
    stamp = datetime.now(timezone.utc).isoformat()
    probe.update(verified_at=stamp, instance_id=receipt['instance_id'])
    archive_name = 'source-' + archive_sha + '.tar'
    command = 'umask 077; mkdir -p /srv/lasr/staging && cat > ' + shlex.quote('/srv/lasr/staging/' + archive_name)
    with Path(args.archive).open('rb') as stream:
        transfer = subprocess.run([*ssh, command], stdin=stream, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    require(transfer.returncode == 0, 'Archive transfer failed; receipt remains unchanged')
    updated = dict(receipt, ssh_verified=True, ssh_verified_at=stamp, ssh_host=host, ssh_port=port,
                   code_sha=args.code_sha, resource_probe=probe)
    deployment = dict(code_sha=args.code_sha, git_commit=args.code_sha, archive_sha256=archive_sha, staged_at=stamp, instance_id=receipt['instance_id'])
    result = ssh_call(ssh, INSTALL, dict(archive_name=archive_name, archive_sha256=archive_sha, code_sha=args.code_sha,
        credentials=values, receipt=updated, probe=probe, deployment=deployment), timeout=180)
    require(result.get('staged') is True and result.get('code_sha') == args.code_sha and result.get('archive_sha256') == archive_sha,
            'Remote staging acknowledgement differs')
    backup = receipt_path.with_name(receipt_path.name + f'.before-stage-{time.time_ns()}')
    backup.write_bytes(receipt_path.read_bytes())
    atomic(receipt_path.with_name('cpu-resource-probe-' + str(time.time_ns()) + '.json'), probe)
    atomic(receipt_path, updated)
    print(json.dumps(result))


if __name__ == '__main__':
    try:
        main()
    except StageError as exc:
        raise SystemExit(str(exc)) from None
    except Exception:
        raise SystemExit('CPU staging failed; receipts were not certified unless staging finished. Inspect inputs/host locally; sensitive exception output suppressed.') from None
