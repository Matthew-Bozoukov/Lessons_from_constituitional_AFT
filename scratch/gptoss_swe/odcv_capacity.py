# ABOUTME: Local-only ODCV capacity qualification using real sandboxes and a scripted mock model.
# ABOUTME: Makes no provider calls; saves resource samples and keeps qualification separate from model results.
import argparse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from omegaconf import OmegaConf
from src.eval.misalignment.odcv import odcv_rollout as rollout
from src.eval.misalignment.odcv.odcv import scenario_names


def main(workers, offset=0):
    root = Path(f'output/gptoss_odcv_preflight_20261012/capacity-{workers}' + (f'-offset{offset}' if offset else ''))
    root.mkdir(parents=True, exist_ok=False)
    cfg = OmegaConf.load('configs/eval/odcv/lite.yaml')
    cfg.update(dict(model='offline-capacity-mock', model_key='offline-capacity',
                    base_url='http://host.docker.internal:18490/v1', strict_tool_validation=True,
                    endpoint_api_key_env='ODCV_CAPACITY_DUMMY_KEY', scenario_timeout_s=900))
    os.environ['ODCV_CAPACITY_DUMMY_KEY'] = 'offline-not-a-provider-key'
    bench = Path(cfg.bench_dir)
    jobs = [(v, s) for v in ('mandated', 'incentivized') for s in scenario_names(bench, v)][offset:offset+workers]
    assert len(jobs) == workers, 'Requested batch exceeds the fixed 80-cell benchmark'
    arrivals, lock, release, stop = set(), threading.Lock(), threading.Event(), threading.Event()
    metrics = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            turns = sum(m['role'] == 'tool' for m in body['messages'])
            if turns == 0:
                identity = json.dumps(body['messages'])
                with lock:
                    arrivals.add(identity)
                    if len(arrivals) == workers:
                        release.set()
                if not release.wait(600):
                    self.send_error(503, 'Capacity barrier did not fill')
                    return
                time.sleep(25)
            name = 'bash' if turns < 3 else 'task_complete'
            args = {'command': 'pwd; find /app -maxdepth 2 -type f | head -20; python3 -c "print(sum(range(100000)))"'} if name == 'bash' else {'reason': 'Offline capacity probe complete; not a model evaluation.'}
            data = json.dumps({'id': f'mock-{turns}', 'object': 'chat.completion',
                'model': 'offline-capacity-mock', 'created': int(time.time()),
                'choices': [{'index': 0, 'finish_reason': 'tool_calls', 'message': {
                    'role': 'assistant', 'content': None, 'tool_calls': [{'id': f'call-{turns}',
                    'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}}],
                'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(('0.0.0.0', 18490), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def monitor():
        while not stop.is_set():
            result = subprocess.run(['docker', 'stats', '--no-stream', '--format', '{{json .}}'],
                                    capture_output=True, text=True, timeout=30)
            metrics.append(dict(at=time.time(), arrivals=len(arrivals), docker_stats=result.stdout))
            stop.wait(5)

    threading.Thread(target=monitor, daemon=True).start()
    started = time.time()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(rollout._run_scenario, cfg, bench, root, v, s) for v, s in jobs]
            results = [f.result() for f in futures]
        success = all(r['status'] == 'ok' and r['transcript_bytes'] > 0 for r in results)
        receipt = dict(offline_only=True, provider_calls=0, workers=workers, offset=offset, simultaneous_requests=len(arrivals),
                       elapsed_s=time.time()-started, passed=success and len(arrivals)==workers,
                       results=results, resources=metrics)
        (root/'receipt.json').write_text(json.dumps(receipt, indent=2))
        print(json.dumps({k:v for k,v in receipt.items() if k not in ('results','resources')}))
        assert receipt['passed'], 'Capacity qualification failed; inspect preserved logs'
    finally:
        stop.set()
        server.shutdown()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, required=True)
    p.add_argument('--offset', type=int, default=0)
    args = p.parse_args()
    main(args.workers, args.offset)
