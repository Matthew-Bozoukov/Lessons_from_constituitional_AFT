# ABOUTME: Offline accounting and isolated loopback routing qualification; never calls Tinker.
# ABOUTME: Uses temporary ledgers and an exact disposable iptables rule on unused test ports.
import concurrent.futures
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def accounting(wrapper):
    from src.infra.endpoints.tinker_budget import Budget
    class Fast(Budget):
        save = wrapper.compact_save
    rates = dict(input_per_million=.33, cached_input_per_million=.066, output_per_million=.84)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'budget.json'
        b = Fast(path, None, 'test', 'test', price_rates=rates)
        unknown = b.reserve(100, 100)
        def operation(n):
            key = b.reserve(1000+n, 100)
            b.settle(key, n % 100, 500)
            return key
        with concurrent.futures.ThreadPoolExecutor(max_workers=80) as pool:
            keys = list(pool.map(operation, range(160)))
        data = json.loads(path.read_text())
        assert len(set(keys)) == 160 and len(data['requests']) == 161
        assert data['requests'][unknown]['state'] == 'reserved'
        for n, key in enumerate(keys):
            row = data['requests'][key]
            expected = ((500+n)*.33+500*.066+(n%100)*.84)/1e6
            assert abs(row['upper_usd']-expected) < 1e-12 and row['state'] == 'completed'
        before = path.read_bytes()
        with patch.object(wrapper.os, 'replace', side_effect=OSError('injected before replace')):
            try:
                b.reserve(20, 20)
                raise AssertionError('Expected injected failure')
            except OSError:
                pass
        assert path.read_bytes() == before
        b.owner.close()
        reopened = Fast(path, None, 'test', 'test', price_rates=rates)
        assert reopened.data == data
        reopened.owner.close()
    return dict(concurrent_operations=160, workers=80, unknown_reservation_preserved=True,
                exact_costs=True, failed_replace_preserves_file=True, restart_preserves_all=True)


def routing(gate_source):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import urllib.request
    old_port, gate_port = 32105, 32104
    received = []
    held = threading.Event()
    entered = threading.Event()
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers['Content-Length']))
            received.append((self.server.label, body))
            if body == b'original-inflight':
                entered.set(); assert held.wait(15)
            self.send_response(200); self.send_header('Content-Length', str(len(body)))
            self.send_header('Connection', 'close'); self.end_headers(); self.wfile.write(body)
        def log_message(self, *args):
            pass
    def server(label):
        s = ThreadingHTTPServer(('127.0.0.1', old_port), Handler)
        s.label = label
        threading.Thread(target=s.serve_forever, daemon=True).start()
        return s
    def call(body):
        request = urllib.request.Request(f'http://127.0.0.1:{old_port}/v1/chat/completions', data=body)
        with urllib.request.urlopen(request, timeout=20) as response:
            assert response.read() == body
        return body
    rule = ['OUTPUT','-p','tcp','-d','127.0.0.1','--dport',str(old_port),'-m','owner',
            '--uid-owner','0','-m','comment','--comment','lasr-compact-qualification',
            '-j','REDIRECT','--to-ports',str(gate_port)]
    installed = False
    old = new = process = None
    with tempfile.TemporaryDirectory(prefix='lasr-gate-test-') as tmp:
        os.chmod(tmp, 0o755)
        root = Path(tmp); gate = root/'gate.py'; shutil.copyfile(gate_source, gate)
        state = root/'state'; state.mkdir(); os.chown(state,65534,65534)
        control = root/'control.json'
        def mode(value):
            q=control.with_suffix('.tmp');q.write_text(json.dumps(value));os.replace(q,control)
        mode(dict(port=None, fallback_port=old_port, lease_until=time.time()+30))
        def nobody():
            os.setgroups([]);os.setgid(65534);os.setuid(65534)
        try:
            old = server('old')
            process = subprocess.Popen(['/usr/bin/python3',str(gate),str(gate_port),str(control),str(state/'status.json')],preexec_fn=nobody)
            for _ in range(100):
                if (state/'status.json').exists():break
                assert process.poll() is None;time.sleep(.05)
            assert (state/'status.json').exists()
            with concurrent.futures.ThreadPoolExecutor(max_workers=81) as pool:
                first=pool.submit(call,b'original-inflight');assert entered.wait(5)
                subprocess.run(['iptables','-t','nat','-I',*rule],check=True);installed=True
                future=[pool.submit(call,f'queued-{i}'.encode()) for i in range(80)]
                time.sleep(.5)
                assert len(received)==1 and not any(f.done() for f in future)
                held.set();assert first.result()==b'original-inflight'
                old.shutdown();old.server_close();old=None
                new=server('new');mode(dict(port=old_port))
                assert len(set(f.result() for f in future))==80
            assert received[0]==('old',b'original-inflight')
            assert len(received)==81 and all(a=='new' for a,b in received[1:])
            # A drain-only controller failure returns queued requests to old port.
            mode(dict(port=None,fallback_port=old_port,lease_until=time.time()+.2))
            assert call(b'rollback-before-cutover')==b'rollback-before-cutover'
            subprocess.run(['iptables','-t','nat','-D',*rule],check=True);installed=False
            assert call(b'direct-after-rule-removal')==b'direct-after-rule-removal'
            for _ in range(100):
                status=json.loads((state/'status.json').read_text())
                if status['active']==0:break
                time.sleep(.05)
            assert not status['errors'] and status['active']==0
            return dict(inflight_preserved=True,queued_requests=80,exactly_once=True,
                        exact_response_bytes=True,drain_failure_fallback=True,routing_removed=True)
        finally:
            held.set()
            if installed:subprocess.run(['iptables','-t','nat','-D',*rule],check=True)
            if process:process.terminate();process.wait(timeout=5)
            for s in (old,new):
                if s:s.shutdown();s.server_close()


if __name__=='__main__':
    sys.path.insert(0,'/srv/lasr/repo')
    directory=Path(sys.argv[1])
    result=dict(accounting=accounting(load(directory/'compact_budget_bridge.py','compact_budget_candidate')),
                routing=routing(directory/'bridge_request_gate.py'))
    print(json.dumps(result))
