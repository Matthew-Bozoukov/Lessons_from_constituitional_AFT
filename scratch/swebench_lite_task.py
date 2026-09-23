# ABOUTME: Feed one frozen instance to mini-SWE-agent with explicit, terminal generation budgets.
# ABOUTME: Each invocation owns a unique attempt directory and digest-pinned isolated Docker container.
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

from scratch.swebench_lite_state import atomic, read


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
    import yaml
    assert importlib.metadata.version('mini-swe-agent') == '2.2.1'
    official = Path(minisweagent.__file__).parent / 'config/benchmarks/swebench.yaml'
    config = yaml.safe_load(official.read_text())
    out = Path(request['out'])
    out.mkdir(parents=True, exist_ok=True)
    config['model']['model_name'] = request['model']
    config['model']['model_kwargs']['api_base'] = request['endpoint']
    config['model']['model_kwargs']['max_tokens'] = request['max_response_tokens']
    config['environment']['env'].update(request['environment'])
    assert 0 < request['max_response_tokens'] <= request['max_task_tokens']
    install_token_limits(LitellmModel, LimitsExceeded, request['max_response_tokens'], request['max_task_tokens'])
    config['environment']['run_args'] = ['--rm', '--network', 'none', '--pull', 'never',
        '--cpus', str(request['cpus']), '--memory', request['memory'], '--pids-limit', str(request['pids']),
        '--label', 'lasr_campaign=' + request['campaign'], '--label', 'lasr_attempt=' + request['attempt']]
    config['agent']['output_path'] = str(out / 'checkpoint.traj.json')
    atomic(out / 'scaffold.json', {'version': '2.2.1', 'official_sha256': hashlib.sha256(official.read_bytes()).hexdigest(),
                                 'effective_config': config, 'effective_cost_limit': 'inert; step_limit=250',
                                 'max_task_tokens': request['max_task_tokens'],
                                 'token_limit_policy': 'terminal unresolved; never infrastructure retry'})

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
