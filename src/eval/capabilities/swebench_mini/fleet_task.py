# ABOUTME: Feed one frozen instance to mini-SWE-agent with explicit, terminal generation budgets.
# ABOUTME: Each invocation owns a unique attempt directory and digest-pinned isolated Docker container.
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import time

from src.eval.capabilities.swebench_mini.fleet_state import atomic, read, tool_slot


def resource_environment(cpus):
    """Libraries often see host CPU count instead of the Docker quota."""
    assert math.isfinite(float(cpus)) and float(cpus) > 0
    threads = str(max(1, math.floor(float(cpus))))
    return {key: threads for key in ('DJANGO_TEST_PROCESSES', 'OMP_NUM_THREADS',
        'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
        'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'LOKY_MAX_CPU_COUNT')}


def resource_shell(cpus, timeout):
    assert math.isfinite(float(timeout)) and float(timeout) > 0
    exports = '\n'.join(f'export {key}={value}' for key, value in resource_environment(cpus).items())
    # docker exec's client timeout does not kill the command inside the container.
    # The inner shell inherits the sentinel; timeout owns its entire process group.
    return ('\n# LASR_RESOURCE_LIMITS_V1\n' + exports + '\n'
        'if [ "${LASR_BOUNDED_SHELL:-0}" != 1 ] && [ -n "${BASH_EXECUTION_STRING:-}" ]; then\n'
        '  export LASR_BOUNDED_SHELL=1\n'
        f'  exec /usr/bin/timeout --signal=TERM --kill-after=5s {float(timeout):g}s '
        '/bin/bash -c "$BASH_EXECUTION_STRING"\nfi\n')


def install_container_limits(container, cpus, timeout):
    """Apply before the first command, or between turns on a running attempt."""
    script = resource_shell(cpus, timeout)
    subprocess.run(['docker', 'exec', '-i', container, '/bin/sh', '-c',
        'test -x /usr/bin/timeout && '
        '(grep -q LASR_RESOURCE_LIMITS_V1 /root/.bashrc || cat >> /root/.bashrc)'],
        input=script, text=True, check=True, capture_output=True, timeout=15)


def container_resources(container):
    """Read cgroup evidence before Docker deletes it; never infer zero from failure."""
    try:
        info = json.loads(subprocess.check_output(['docker', 'inspect', container], text=True, timeout=10))[0]
        pid = info['State']['Pid']
        group = next(line.split(':', 2)[2] for line in Path(f'/proc/{pid}/cgroup').read_text().splitlines()
                     if line.startswith('0:'))
        base = Path('/sys/fs/cgroup') / group.lstrip('/')
        result = {'container': container, 'memory_peak_bytes': int((base / 'memory.peak').read_text()),
                  'memory_limit_bytes': info['HostConfig']['Memory']}
        for name in ('memory.events', 'cpu.stat'):
            result[name] = {k: int(v) for k, v in (line.split() for line in (base / name).read_text().splitlines())}
        return result
    except (OSError, ValueError, KeyError, StopIteration, subprocess.SubprocessError) as exc:
        return {'container': container, 'observation_error': type(exc).__name__}


def token_limit_reason(response, total, task_limit):
    """Truncated/model-budget outputs are terminal outcomes, not retryable failures."""
    if response['choices'][0].get('finish_reason') == 'length':
        return 'response_token_limit'
    if total >= task_limit:
        return 'task_token_limit'
    return None


def install_token_limits(model_class, limits_exceeded, response_limit, task_limit):
    original_query, original_parse = model_class._query, model_class._parse_actions

    def query(self, messages, **kwargs):
        remaining = task_limit - getattr(self, '_lite_tokens', 0)
        assert remaining > 0, 'Terminal budget outcome must stop the agent'
        kwargs['max_tokens'] = min(response_limit, remaining)
        return original_query(self, messages, **kwargs)

    def parse(self, response):
        raw = response.model_dump()
        used = raw['usage']['completion_tokens']
        assert isinstance(used, int) and used >= 0, 'Missing completion-token accounting'
        self._lite_tokens = getattr(self, '_lite_tokens', 0) + used
        reason = token_limit_reason(raw, self._lite_tokens, task_limit)
        if reason:
            # Preserve even the truncated response before the stock agent's exit.
            message = raw['choices'][0]['message'] | {'role': 'assistant',
                       'extra': {'response': raw, 'actions': [], 'limit_reason': reason}}
            raise limits_exceeded(message, {'role': 'exit', 'content': reason,
                    'extra': {'exit_status': 'LimitsExceeded', 'submission': '',
                              'limit_reason': reason, 'completion_tokens': self._lite_tokens}})
        return original_parse(self, response)

    model_class._query, model_class._parse_actions = query, parse


def configure_request_transport(config, request):
    # Explicit per-call timeout reaches the OpenAI client; changing a library's
    # global default is insufficient. Defaults also cover older queued requests.
    timeout = float(request.get('model_request_timeout_seconds', 1800))
    attempts = int(request.get('model_request_attempts', 2))
    assert math.isfinite(timeout) and timeout > 0 and attempts > 0
    config['model']['model_kwargs'].update(timeout=timeout, num_retries=0)
    # One retry layer only: do not multiply mini-swe-agent retries by SDK retries.
    os.environ['MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT'] = str(attempts)
    return {'timeout_seconds': timeout, 'max_attempts': attempts, 'sdk_retries': 0}


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
    from minisweagent.models.litellm_model import LitellmModel
    from minisweagent.exceptions import LimitsExceeded
    from minisweagent.environments.docker import DockerEnvironment
    import yaml
    assert importlib.metadata.version('mini-swe-agent') == '2.2.1'
    official = Path(minisweagent.__file__).parent / 'config/benchmarks/swebench.yaml'
    config = yaml.safe_load(official.read_text())
    out = Path(request['out'])
    out.mkdir(parents=True, exist_ok=True)
    config['model']['model_name'] = request['model']
    config['model']['model_kwargs']['api_base'] = request['endpoint']
    config['model']['model_kwargs']['max_tokens'] = request['max_response_tokens']
    transport = configure_request_transport(config, request)
    config['environment']['env'].update(request['environment'])
    config['environment']['env'].update(resource_environment(request['cpus']))
    assert config['environment']['env']['BASH_ENV'] == '/root/.bashrc'
    original_start = DockerEnvironment._start_container
    original_execute = DockerEnvironment.execute

    def admitted_execute(self, *args, **kwargs):
        with tool_slot(request.get('tool_slots_path', str(out.parent / '.tool-slots')),
                       request.get('tool_concurrency', 32),
                       wait_seconds=request.get('tool_queue_timeout_seconds', 600),
                       min_available_gib=request.get('min_available_memory_gib', 16)) as waited:
            started = time.monotonic()
            try:
                return original_execute(self, *args, **kwargs)
            finally:
                with (out / 'tool-timing.jsonl').open('a') as stream:
                    stream.write(json.dumps({'queue_seconds': waited,
                        'execution_seconds': time.monotonic()-started, 'time': time.time()}) + '\n')

    DockerEnvironment.execute = admitted_execute

    def bounded_start(self):
        original_start(self)
        install_container_limits(self.container_id, request['cpus'], self.config.timeout)

    DockerEnvironment._start_container = bounded_start
    assert 0 < request['max_response_tokens'] <= request['max_task_tokens']
    install_token_limits(LitellmModel, LimitsExceeded, request['max_response_tokens'], request['max_task_tokens'])
    config['environment']['run_args'] = ['--rm', '--network', 'none', '--pull', 'never',
        '--cpus', str(request['cpus']), '--memory', request['memory'], '--pids-limit', str(request['pids']),
        '--label', 'lasr_campaign=' + request['campaign'], '--label', 'lasr_attempt=' + request['attempt']]
    config['agent']['output_path'] = str(out / 'checkpoint.traj.json')
    atomic(out / 'scaffold.json', {'version': '2.2.1', 'official_sha256': hashlib.sha256(official.read_bytes()).hexdigest(),
                                 'effective_config': config, 'effective_cost_limit': 'inert; step_limit=250',
                                 'resource_policy': 'quota-derived-thread-limits-and-in-container-timeout-v1',
                                 'resource_shell': resource_shell(request['cpus'], config['environment']['timeout']),
                                 'tool_admission': {key: request.get(key) for key in
                                      ('tool_concurrency', 'tool_queue_timeout_seconds', 'min_available_memory_gib')},
                                 'task_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                 'max_task_tokens': request['max_task_tokens'],
                                 'token_limit_policy': 'terminal unresolved; never infrastructure retry',
                                 'request_transport': transport})

    class AtomicTrackingAgent(upstream.ProgressTrackingAgent):
        def save(self, path, *extra_dicts):
            data = super().save(None, *extra_dicts)
            if path:
                atomic(path, data)
            if any('exit_status' in extra.get('info', {}) for extra in extra_dicts):
                atomic(out / 'resources.json', container_resources(self.env.container_id))
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
