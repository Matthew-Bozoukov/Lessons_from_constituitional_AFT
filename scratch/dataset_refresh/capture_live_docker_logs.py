# ABOUTME: Preserve raw Docker logs for the already-running original-low-stakes ODCV campaign.
# ABOUTME: Read-only scoped sidecar; never restarts containers, inference or the evaluation owner.
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main(campaign):
    root = Path(campaign).resolve()
    out = root/'raw_docker_logs'
    out.mkdir(exist_ok=True)
    manifest = out/'manifest.json'
    assert not manifest.exists(), 'Do not duplicate a live capture'
    jobs, records = {}, {}
    def save():
        temp = manifest.with_suffix('.tmp')
        temp.write_text(json.dumps(records, indent=2), encoding='utf-8')
        temp.replace(manifest)
    while True:
        state = json.loads((root/'low_original/broader_eval_status.json').read_text())
        result = subprocess.run(['docker','ps','-aq','--filter','name=odcv-582cf9-'],
                                capture_output=True, timeout=30, check=True)
        for cid in result.stdout.decode().split():
            if cid in jobs or cid in records:
                continue
            inspect = subprocess.run(['docker','inspect','--format','{{json .Config.Labels}}',cid],
                                     capture_output=True, timeout=20)
            if inspect.returncode:
                continue
            labels = json.loads(inspect.stdout)
            ws = Path(labels.get('com.docker.compose.project.working_dir','')).resolve()
            if not ws.is_relative_to(Path('C:/odcv-old-low').resolve()):
                continue
            service = labels.get('com.docker.compose.service')
            if service not in ('executor','orchestrator'):
                continue
            path = out/f'{cid}.log'
            stream = path.open('wb')
            proc = subprocess.Popen(['docker','logs','--follow',cid], stdout=stream,
                stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            jobs[cid] = (proc, stream, path)
            records[cid] = {'workspace':str(ws),'service':service,'path':str(path),
                            'capture_started_unix':time.time(),'state':'streaming'}
        for cid, (proc, stream, path) in list(jobs.items()):
            if proc.poll() is not None:
                stream.close()
                records[cid].update(state='saved',exit_code=proc.returncode,bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                del jobs[cid]
        save()
        if (state.get('evaluation_driver_completed') or state.get('error') or state.get('termination_verified')) and not jobs:
            break
        time.sleep(5)


if __name__ == '__main__':
    main(sys.argv[1])
