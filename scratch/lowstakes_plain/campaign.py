# ABOUTME: Train September practical low-stakes on plain replay, then run matched ODCV and MASK.
# ABOUTME: Reuses guarded owners and an explicitly recorded concurrent-campaign admission policy.
from pathlib import Path
import json
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scratch.nonmoral_plain.campaign import run
from src.infra import runpod


def reservation():
    """Release the existing $100 reservation only on completion and no owned GPUs."""
    override = ROOT / 'output/lowstakes_plain/admission_override.json'
    if override.exists():
        policy = json.loads(override.read_text())
        assert policy['approved'] is True and policy['policy'] == 'user_requested_concurrent_start'
        return 0
    script = """import json,pathlib
r=pathlib.Path('/srv/lasr/runs/base-qwen-20261007/metadata')
print(json.dumps({'state':json.loads((r/'state.json').read_text()),'supervisor':json.loads((r/'supervisor.json').read_text())}))
"""
    result = subprocess.run(['ssh', '-i', 'C:/Users/nikak/.ssh/msm_audit', '-p', '56859',
                             '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                             'root@77.104.167.149', 'python3 -c ' + shlex.quote(script)],
                            capture_output=True, text=True, timeout=40)
    if result.returncode:
        return 100
    data = json.loads(result.stdout)
    if (data['supervisor'].get('status') != 'complete'
            or not data['supervisor'].get('verified_hf_revision')):
        return 100
    owned = {p['id'] for p in data['state']['pods'] if p.get('id')}
    if any(p['id'] in owned or p.get('name', '').startswith('nika-swe-lite-d742aeb6-')
           for p in runpod.active_pods()):
        return 100
    return 0


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-pretraining', action='store_true')
    args = parser.parse_args()
    run(ROOT / 'output/lowstakes_plain', reservation=reservation, resume_pretraining=args.resume_pretraining)
