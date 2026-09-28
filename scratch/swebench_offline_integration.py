# ABOUTME: Local real-agent/Docker/official-grader qualification with synthetic responses and public gold patches.
# ABOUTME: Uses only existing task images and never contacts a model, rents infrastructure, or publishes a model score.
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import threading
import time
import uuid
from omegaconf import OmegaConf
from src.eval.capabilities.swebench_mini.fleet_state import State, atomic, read
from src.eval.capabilities.swebench_mini.fleet_worker import consume


def main():
    cfg = OmegaConf.load('configs/eval/swebench_mini/lite.yaml')
    root = Path('/work/output/2026-09-24_swebench_offline_fixes') / ('integration-' + uuid.uuid4().hex[:8])
    cfg.root = str(root)
    # This laptop qualification uses sequential containers, not a Vast capacity test.
    cfg.min_available_memory_gib = 1
    cfg.task_seconds = 300
    cfg.tool_concurrency = 2
    cfg.grading_workers = 2
    cfg.max_infrastructure_attempts = 1
    cfg.model_request_timeout_seconds = 20
    rows = read('/work/output/2026-09-24_swebench_offline_fixes/readiness/metadata/swebench_lite_test.json')
    ids = ['django__django-11099', 'django__django-11179', 'django__django-11964']
    selected = {r['instance_id']: r for r in rows if r['instance_id'] in ids}
    assert len(selected) == 3
    images = {}
    for iid in ids:
        tag = 'swebench/sweb.eval.x86_64.' + iid.replace('__', '_1776_') + ':latest'
        info = json.loads(subprocess.check_output(['docker', 'image', 'inspect', tag]))[0]
        images[iid] = {'digest': info['RepoDigests'][0]}
    campaign = 'offline-' + uuid.uuid4().hex[:10]
    (root/'metadata').mkdir(parents=True)
    atomic(root/'metadata/manifest.json', {'campaign': campaign, 'limitations': 'SYNTHETIC INFRASTRUCTURE TEST ONLY'})
    atomic(root/'metadata/images.json', images)
    atomic(root/'metadata/swebench_lite_test.json', rows)
    atomic(root/'metadata/state.json', {'tasks': {i: {'status': 'pending', 'attempts': []} for i in ids},
           'pods': [], 'deadline': None, 'halt': None})
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path == '/tokenize':
                count = 100 + 25 * sum(bool(m.get('reasoning')) for m in request['messages'])
                payload = json.dumps({'count': count}).encode()
            else:
                for key, value in OmegaConf.to_container(cfg.sampling).items():
                    assert request[key] == value, key
                task = request['model']
                index = sum(m['role'] == 'assistant' for m in request['messages'])
                calls.append([task, index])
                message = {'role': 'assistant', 'content': 'Synthetic qualification response',
                           'reasoning': f'Synthetic reasoning {task} {index}'}
                finish = 'tool_calls'
                if task == 'normal' and index == 0:
                    pass  # No-tool error: the next request must retain this response.
                else:
                    if task == 'normal':
                        assert index <= 3
                        assert request['messages'][2]['reasoning_content'] == 'Synthetic reasoning normal 0'
                        command = ("printf '%s' '" + base64.b64encode(selected[ids[0]]['patch'].encode()).decode()
                                   + "' | base64 -d > /tmp/audit.patch") if index == 2 else 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat /tmp/audit.patch'
                        arguments = '{broken' if index == 1 else json.dumps({'command': command})
                    elif task == 'forced':
                        assert index <= 1
                        command = ("printf '%s' '" + base64.b64encode(selected[ids[1]]['patch'].encode()).decode()
                                   + "' | base64 -d | git apply") if index == 0 else 'git reset --hard HEAD'
                        arguments = json.dumps({'command': command})
                        if index == 1: finish = 'length'
                    else:
                        assert task == 'empty' and index == 0
                        arguments = json.dumps({'command': 'touch SHOULD_NEVER_EXECUTE'})
                        finish = 'length'
                    message['tool_calls'] = [{'id': f'call-{index}', 'type': 'function',
                                             'function': {'name': 'bash', 'arguments': arguments}}]
                prompt = 100 + 25 * sum(bool(m.get('reasoning_content')) for m in request['messages'])
                payload = json.dumps({'id': f'{task}-{index}', 'object': 'chat.completion', 'created': 1, 'model': task,
                    'choices': [{'index': 0, 'finish_reason': finish, 'message': message}],
                    'usage': {'prompt_tokens': prompt, 'completion_tokens': 50, 'total_tokens': prompt+50}}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers(); self.wfile.write(payload)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        for iid, model in zip(ids, ('normal', 'forced', 'empty')):
            admission = {'directory': str(root/'.token-slots'/model), 'budget_tokens': 450000,
                         'expires': time.time()+600, 'fairness_seconds': 1}
            endpoint = f'http://127.0.0.1:{server.server_port}/v1'
            consume(endpoint, 'hosted_vllm/'+model, cfg, model, [iid], time.time()+2400, admission=admission)
            task = read(root/'metadata/state.json')['tasks'][iid]
            assert task['status'] == 'valid', (root, iid, task)
            attempt = root/'rollouts'/iid/task['attempts'][0]['id']
            assert len(list((attempt/'http').glob('*.body'))) == sum(c[0] == model for c in calls)
            assert read(attempt/'diagnostics.json')['discarded_response_corrections'] == 0
        before = list(calls)
        consume(endpoint, 'hosted_vllm/normal', cfg, 'repeat', ids, time.time()+2400)
        assert calls == before, 'Completed outcomes must never reroll'
    finally:
        server.shutdown(); server.server_close()
    state = read(root/'metadata/state.json')
    predictions = [dict(t['attempts'][0]['prediction'], instance_id=i) for i, t in state['tasks'].items()]
    assert predictions[2]['model_patch'] == ''
    predictions_path = root/'predictions.jsonl'
    predictions_path.write_text(''.join(json.dumps(p)+'\n' for p in predictions))
    grading = root/'grading'; grading.mkdir()
    harness = Path('/work/src/eval/capabilities/swebench_mini/envs/harness/.venv/bin/python')
    with (grading/'harness.log').open('w') as log:
        subprocess.run([str(harness), '-m', 'swebench.harness.run_evaluation',
            '--dataset_name', str(root/'metadata/swebench_lite_test.json'), '--predictions_path', str(predictions_path),
            '--instance_ids', *ids, '--run_id', campaign, '--max_workers', '2', '--timeout', '600',
            '--cache_level', 'instance', '--clean', 'False', '--namespace', 'swebench'],
            cwd=grading, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
    reports = list(grading.glob('*.'+campaign+'.json'))
    assert len(reports) == 1, (root, reports)
    report = read(reports[0])
    assert set(report['resolved_ids']) == set(ids[:2]), (root, report)
    assert not report['error_ids'], (root, report['error_ids'])
    atomic(root/'results.json', {'status': 'passed', 'synthetic': True, 'model_evaluation': False,
           'calls': calls, 'official_resolved': report['resolved_ids'], 'empty_patch_unresolved': ids[2],
           'rerolled_completed_outcomes': False, 'root': str(root)})
    print('OFFLINE_INTEGRATION_PASSED', root, flush=True)


if __name__ == '__main__': main()
