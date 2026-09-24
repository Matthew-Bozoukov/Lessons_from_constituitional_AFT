# ABOUTME: GPU-free Docker workload qualification with real repository tests and agent-process overhead.
# ABOUTME: Run on the prepared CPU: python -m scratch.swebench_cpu_load --output PATH [--baseline-only].
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import uuid

import psutil
from scratch.swebench_lite_state import atomic, read, tool_slot
from scratch.swebench_lite_task import container_resources, install_container_limits

WORKLOADS = {
    'django/django': 'python tests/runtests.py --settings=test_sqlite --parallel=2 expressions',
    'sympy/sympy': 'python -c "import sympy; assert sympy.test(\'core/tests/test_basic.py\', subprocess=False)"',
    'scikit-learn/scikit-learn': 'python -m pytest sklearn/metrics/tests/test_classification.py -q',
    'matplotlib/matplotlib': 'python -m pytest lib/matplotlib/tests/test_colors.py -q',
    'astropy/astropy': 'python -m pytest astropy/coordinates/tests/test_angles.py -q',
    'pytest-dev/pytest': 'python -m pytest testing/test_parseopt.py -q',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    nonce = 'capacity-' + uuid.uuid4().hex
    rows = read('/srv/lasr/runs/cpu-readiness/metadata/swebench_lite_test.json')
    images = read('/srv/lasr/runs/cpu-readiness/metadata/images.json')
    cases = [(next(r['instance_id'] for r in rows if r['repo'] == repo), cmd)
             for repo, cmd in WORKLOADS.items()]
    children = []
    stopped = threading.Event()
    record = {'kind': 'infrastructure-check', 'model_inference': False, 'nonce': nonce,
              'started': time.time(), 'cpu_count': psutil.cpu_count(),
              'memory_gib': psutil.virtual_memory().total / 2**30, 'phases': []}

    def terminate(*unused):
        stopped.set()
        ids = subprocess.check_output(['docker', 'ps', '-aq', '--filter', 'label=lasr_load=' + nonce], text=True).split()
        if ids:
            subprocess.run(['docker', 'rm', '-f', *ids], capture_output=True, timeout=90)
        for proc in children:
            if proc.poll() is None:
                proc.terminate()

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    def phase(count, limit, rounds, overhead):
        phase_dir = args.output / f'{count}-agents-{limit}-tools'
        phase_dir.mkdir()
        barrier = threading.Barrier(count)
        samples = []
        finished = threading.Event()
        def observe():
            while not finished.is_set():
                cpu = psutil.cpu_times_percent(interval=1)
                mem = psutil.virtual_memory()
                samples.append({'time': time.time(), 'cpu_busy': 100-cpu.idle,
                                'iowait': cpu.iowait, 'available_gib': mem.available/2**30})
                if mem.available < 32*2**30:
                    stopped.set()
                    terminate()
                    break
        watcher = threading.Thread(target=observe, daemon=True)
        watcher.start()
        started = time.monotonic()

        def task(index):
            iid, command = cases[index % len(cases)]
            name = nonce + '-' + str(index)
            proc = None
            data = {'instance_id': iid, 'command': command, 'commands': []}
            try:
                if overhead:
                    proc = subprocess.Popen(['src/eval/capabilities/swebench_mini/envs/agent/.venv/bin/python',
                        '-c', 'import litellm, minisweagent; print("ready", flush=True); import sys; sys.stdin.read()'],
                        env=os.environ | {'MSWEA_SILENT_STARTUP': '1'},
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                    children.append(proc)
                    while True:
                        line = proc.stdout.readline()
                        assert line, 'agent imports failed'
                        if line.strip() == 'ready':
                            break
                subprocess.run(['docker', 'run', '-d', '--name', name, '--label', 'lasr_load='+nonce,
                    '--network', 'none', '--pull', 'never', '--cpus', '2', '--memory', '4g', '--pids-limit', '512',
                    '-e', 'BASH_ENV=/root/.bashrc', '-w', '/testbed', '--entrypoint', '/bin/sleep',
                    images[iid]['digest'], 'infinity'], check=True, capture_output=True, timeout=120)
                install_container_limits(name, 2, 120)
                barrier.wait(timeout=240)
                for rep in range(rounds):
                    if stopped.is_set():
                        raise RuntimeError('memory guard or user stop')
                    queued = time.monotonic()
                    with tool_slot(args.output / '.tool-slots', limit, min_available_gib=32):
                        begin = time.monotonic()
                        result = subprocess.run(['docker', 'exec', name, 'bash', '-c', command],
                                                capture_output=True, text=True, timeout=135)
                    (phase_dir / f'{index}-{rep}.log').write_text(result.stdout + result.stderr)
                    data['commands'].append({'returncode': result.returncode,
                        'queue_seconds': begin-queued, 'execution_seconds': time.monotonic()-begin})
                data['resources'] = container_resources(name)
                if proc:
                    data['agent_pss_mib'] = psutil.Process(proc.pid).memory_full_info().pss/2**20
            except Exception as exc:
                barrier.abort()
                data['error'] = type(exc).__name__ + ': ' + str(exc)
            finally:
                subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=90)
                if proc and proc.poll() is None:
                    proc.communicate(input='', timeout=30)
            return data
        try:
            with ThreadPoolExecutor(max_workers=count) as pool:
                results = list(pool.map(task, range(count)))
        finally:
            finished.set()
            watcher.join(timeout=3)
        value = {'agents': count, 'tool_limit': limit, 'rounds': rounds, 'agent_overhead': overhead,
                 'seconds': time.monotonic()-started, 'tasks': results, 'samples': samples}
        atomic(phase_dir/'results.json', value)
        record['phases'].append(value)
        atomic(args.output/'results.json', record)
        errors = sum(bool(x.get('error')) for x in results)
        failures = sum(c['returncode'] != 0 for x in results for c in x['commands'])
        print(json.dumps({'agents': count, 'tools': limit, 'errors': errors, 'nonzero_commands': failures,
                          'seconds': value['seconds'], 'output': str(phase_dir)}), flush=True)
        return errors, failures

    try:
        errors, failures = phase(6, 1, 1, False)
        if errors or failures:
            raise RuntimeError('Baseline commands must pass before stress qualification')
        if not args.baseline_only:
            for n in [40, 60, 80]:
                errors, failures = phase(n, n, 2, True)
                if errors or stopped.is_set():
                    raise RuntimeError('Qualification stopped for infrastructure failure')
            phase(80, 32, 2, True)
        record['status'] = 'finished'
    finally:
        terminate()
        record['finished'] = time.time()
        atomic(args.output/'results.json', record)


if __name__ == '__main__':
    main()
