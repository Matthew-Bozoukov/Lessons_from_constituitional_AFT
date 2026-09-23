# ABOUTME: Feed one frozen instance to stock mini-SWE-agent; change only persistence and transport.
# ABOUTME: Each invocation owns a unique attempt directory and digest-pinned isolated Docker container.
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

from scratch.swebench_lite_state import atomic, read


def cleanup(label):
    ids = subprocess.check_output(['docker', 'ps', '-aq', '--filter', f'label=lasr_attempt={label}'], text=True).split()
    if ids:
        subprocess.run(['docker', 'rm', '-f', *ids], check=True, timeout=90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    request = read(parser.parse_args().request)
    # Imports happen after the caller installed the isolated global-config environment.
    from minisweagent.run.benchmarks import swebench as upstream
    import minisweagent
    import yaml
    assert importlib.metadata.version('mini-swe-agent') == '2.2.1'
    official = Path(minisweagent.__file__).parent / 'config/benchmarks/swebench.yaml'
    config = yaml.safe_load(official.read_text())
    out = Path(request['out'])
    out.mkdir(parents=True, exist_ok=True)
    config['model']['model_name'] = request['model']
    config['model']['model_kwargs']['api_base'] = request['endpoint']
    config['environment']['run_args'] = ['--rm', '--network', 'none', '--pull', 'never',
        '--cpus', str(request['cpus']), '--memory', request['memory'], '--pids-limit', str(request['pids']),
        '--label', 'lasr_campaign=' + request['campaign'], '--label', 'lasr_attempt=' + request['attempt']]
    config['agent']['output_path'] = str(out / 'checkpoint.traj.json')
    atomic(out / 'scaffold.json', {'version': '2.2.1', 'official_sha256': hashlib.sha256(official.read_bytes()).hexdigest(),
                                 'effective_config': config, 'effective_cost_limit': 'inert; step_limit=250'})

    class AtomicTrackingAgent(upstream.ProgressTrackingAgent):
        def save(self, path, *extra_dicts):
            data = super().save(None, *extra_dicts)
            if path:
                atomic(path, data)
            return data

    upstream.ProgressTrackingAgent = AtomicTrackingAgent
    row = request['instance']
    assert '@sha256:' in request['image']
    row['image_name'] = request['image']
    # Upstream receives only the public problem plus its environment; no gold patch.
    for key in ('patch', 'test_patch', 'hints_text', 'FAIL_TO_PASS', 'PASS_TO_PASS'):
        row.pop(key, None)
    try:
        upstream.process_instance(row, out, config, upstream.RunBatchProgressManager(1, out / 'exit_status.yaml'))
    finally:
        cleanup(request['attempt'])


if __name__ == '__main__':
    main()
