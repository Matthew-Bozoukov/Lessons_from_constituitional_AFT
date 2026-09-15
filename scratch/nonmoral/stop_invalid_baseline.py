# ABOUTME: Preserve the invalid CRLF baseline's live Docker artifacts before exact-resource cleanup.
# ABOUTME: Only containers whose Compose working directory is inside this specific run are eligible.
import json
from pathlib import Path
import subprocess
import time

ROOT = Path('C:/nm-eval/2026-09-09_odcv_nonmoral_common_3x_021823')
OUT = Path('output/nonmoral_overnight/20260909')


def docker(*args):
    return subprocess.run(['docker', *args], capture_output=True, text=True,
                          encoding='utf-8', errors='replace', timeout=120)


ids = docker('ps', '-aq').stdout.split()
items = json.loads(docker('inspect', *ids).stdout) if ids else []
owned = []
for item in items:
    labels = item['Config'].get('Labels') or {}
    cwd = labels.get('com.docker.compose.project.working_dir', '')
    try:
        Path(cwd).resolve().relative_to(ROOT.resolve())
    except ValueError:
        continue
    assert labels['com.docker.compose.project'].startswith('odcv-29634b-')
    owned.append((item['Id'], labels))

records = []
for cid, labels in owned:
    dest = OUT / 'invalid_live_snapshots' / labels['com.docker.compose.project']
    dest.mkdir(parents=True, exist_ok=True)
    service = labels['com.docker.compose.service']
    stopped = docker('stop', '--time', '5', cid)
    logs = docker('logs', cid)
    (dest / f'{service}_docker.log').write_text(logs.stdout + logs.stderr, encoding='utf-8')
    copied = None
    if service == 'executor':
        copied = docker('cp', f'{cid}:/app/messages_record.txt', str(dest/'partial_messages_record.txt')).returncode
    removed = docker('rm', cid)
    records.append(dict(container_id=cid, project=labels['com.docker.compose.project'],
                        service=service, stopped=stopped.returncode, copied=copied, removed=removed.returncode))

network_records = []
for project in sorted({x[1]['com.docker.compose.project'] for x in owned}):
    for nid in docker('network','ls','-q','--filter',f'label=com.docker.compose.project={project}').stdout.split():
        result = docker('network','rm',nid)
        network_records.append(dict(network_id=nid, project=project, removed=result.returncode))

status_path=OUT/'baseline_status.json'
state=json.loads(status_path.read_text())
closed=time.time()
elapsed=closed-state['rented_at_unix']
state.update(phase='invalidated_crlf_stopped', termination_verified=True, terminated_at_unix=closed,
             elapsed_seconds=elapsed, estimated_gpu_and_storage_usd=elapsed/3600*(state['actual_gpu_hourly_usd']+.1),
             invalidation_reason='Windows CRLF shell scripts caused environment repair behavior; no scientific results accepted.',
             completed_transcripts_preserved=sum(p.stat().st_size>0 for p in ROOT.glob('**/agent_logs/**/messages_record.txt')),
             total_planned_rollouts=240, judge_calls=0)
status_path.write_text(json.dumps(state,indent=2),encoding='utf-8')
(OUT/'invalid_cleanup.json').write_text(json.dumps(dict(containers=records,networks=network_records),indent=2),encoding='utf-8')
print(json.dumps(dict(owned_containers=len(records), owned_networks=len(network_records),
                     completed_transcripts_preserved=state['completed_transcripts_preserved'],
                     estimated_gpu_and_storage_usd=state['estimated_gpu_and_storage_usd'])))
