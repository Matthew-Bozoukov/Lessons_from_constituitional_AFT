# ABOUTME: Real pinned client and HTTP transport tests for sampling, rejected histories and socket failures.
# ABOUTME: Run with the Linux mini-SWE-agent environment; no model, provider or Docker required.
import os
import unittest
if os.name != 'posix':
    raise unittest.SkipTest('Production task wrappers use Linux locks')
try:
    import minisweagent
except ImportError:
    raise unittest.SkipTest('Run in the pinned agent environment')
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import yaml

from minisweagent.agents.default import DefaultAgent
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.exceptions import FormatError, LimitsExceeded
from src.eval.capabilities.swebench_mini.fleet_protocol import configure_protocol, install_protocol, ambiguous_failure
from src.eval.capabilities.swebench_mini.fleet_task import install_token_limits
from src.eval.capabilities.swebench_mini.browser import diagnose


class FakeEnv:
    def get_template_vars(self): return {}
    def serialize(self): return {}


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.calls = []
        self.kind = 'normal'
        self.config = yaml.safe_load(Path('configs/eval/swebench_mini/lite.yaml').read_text())
        test = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path == '/tokenize':
                    payload = json.dumps({'count': 20}).encode()
                else:
                    test.calls.append(data)
                    self.chat_requests = getattr(self, 'chat_requests', 0) + 1
                    if test.kind == 'drop_reused' and self.chat_requests > 1:
                        # Emulate a server closing a reused keep-alive socket while
                        # a new request races its idle timeout. Fresh sockets work.
                        self.connection.shutdown(socket.SHUT_RDWR); self.connection.close(); return
                    if test.kind == 'drop':
                        self.connection.shutdown(socket.SHUT_RDWR); self.connection.close(); return
                    if test.kind in ('400', '429', '500'):
                        payload = json.dumps({'error': {'message': 'synthetic rejection', 'type': 'invalid_request_error'}}).encode()
                        self.send_response(int(test.kind)); self.send_header('Content-Length', str(len(payload)))
                        self.end_headers(); self.wfile.write(payload); return
                    message = {'role': 'assistant', 'content': 'audit answer', 'reasoning': 'TRACE-' + str(len(test.calls))}
                    if test.kind != 'no_tool':
                        arguments = '{"command":"echo audit"}'
                        if test.kind in ('bad_json','raw_bad_json'): arguments = '{oops'
                        if test.kind == 'null_args': arguments = 'null'
                        if test.kind == 'bad_command': arguments = '{"command":123}'
                        message['tool_calls'] = [{'id': 'call-'+str(len(test.calls)), 'type': 'function',
                            'function': {'name': 'unknown' if test.kind == 'bad_tool' else 'bash', 'arguments': arguments}}]
                        if test.kind == 'raw_bad_json':
                            message['provider_specific_fields'] = {'harmony_boundary':'handoff'}
                    payload = json.dumps({'id': 'audit', 'object': 'chat.completion', 'created': 1, 'model': 'audit',
                        'choices': [{'index': 0, 'finish_reason': 'length' if test.kind == 'length' else 'tool_calls', 'message': message}],
                        'usage': {'prompt_tokens': 20, 'completion_tokens': 7, 'total_tokens': 27}}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                partial = test.kind == 'partial' and self.path != '/tokenize'
                self.send_header('Content-Length', str(len(payload) + (500 if partial else 0)))
                self.end_headers()
                self.wfile.write(payload)
                if partial:
                    self.wfile.flush(); self.connection.shutdown(socket.SHUT_RDWR); self.connection.close()

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.request = {'out': str(self.root), 'sampling': self.config['sampling'], 'protocol_version': 'lite-v4',
            'endpoint': f'http://127.0.0.1:{self.server.server_port}/v1', 'model_request_timeout_seconds': 1,
            'model': 'hosted_vllm/audit', 'context_window': 1000,
            'token_admission': {'directory': str(self.root/'slots'), 'budget_tokens': 1000,
                                'expires': time.time()+60, 'fairness_seconds': 1}}
        class Model(LitellmModel): pass
        options = {'model': {'model_kwargs': {'api_base': self.request['endpoint'], 'api_key': 'EMPTY',
                    'timeout': 1, 'num_retries': 0, 'temperature': 0, 'drop_params': True}}}
        configure_protocol(options, self.request)
        client = install_protocol(Model, self.request)
        self.addCleanup(client.close)
        install_token_limits(Model, LimitsExceeded, 64, 256, self.request)
        self.model = Model(model_name='hosted_vllm/audit', cost_tracking='ignore_errors', **options['model'])
        self.agent = DefaultAgent(self.model, FakeEnv(), system_template='audit', instance_template='audit',
                                  cost_limit=0, step_limit=20)
        self.agent.messages = [{'role': 'user', 'content': 'audit task'}]

    def test_sampling_reaches_real_http_body(self):
        self.agent.query()
        for key, value in self.config['sampling'].items():
            self.assertEqual(self.calls[0][key], value, key)
        self.assertNotIn('extra_body', self.calls[0])
        row = json.loads(next((self.root/'http').glob('*.json')).read_text())
        self.assertTrue(row['complete'])
        self.assertEqual(row['parameters']['temperature'], 1.0)
        self.assertNotIn('api_key', json.dumps(row))
        messages = []
        for sha in row['message_hashes']:
            message = json.loads((self.root/'http/messages'/f'{sha}.json').read_text())
            self.assertEqual(sha, hashlib.sha256(json.dumps(message, ensure_ascii=False, sort_keys=True).encode()).hexdigest())
            messages.append(message)
        self.assertEqual(messages, self.calls[0]['messages'])
        raw = json.loads(next((self.root/'http').glob('*.body')).read_bytes())
        self.assertEqual(raw['choices'][0]['message']['reasoning'], 'TRACE-1')
        self.assertNotIn('reasoning_content', raw['choices'][0]['message'])
        self.assertEqual(self.agent.messages[-1]['reasoning_content'], 'TRACE-1')

    def test_all_rejected_response_variants_preserve_reasoning(self):
        for kind in ('no_tool', 'bad_tool', 'bad_json', 'null_args', 'bad_command'):
            with self.subTest(kind=kind):
                self.kind = kind
                with self.assertRaises(FormatError) as caught:
                    self.agent.query()
                history = caught.exception.messages
                self.assertEqual(history[0]['role'], 'assistant')
                self.assertTrue(history[0]['extra']['format_error_preserved'])
                self.assertEqual(history[0]['extra']['actions'], [])
                self.assertIn('TRACE-', history[0]['reasoning_content'])
                self.agent.add_messages(*history)
                self.kind = 'normal'
                self.agent.query()
                self.assertTrue(any(m.get('reasoning_content') == history[0]['reasoning_content'] for m in self.calls[-1]['messages']))
                self.assertEqual(diagnose({'messages': list(history)})['discarded_response_corrections'], 0)
                self.agent.add_messages({'role': 'tool', 'tool_call_id': self.agent.messages[-1]['tool_calls'][0]['id'], 'content': 'audit'})

    def test_terminal_truncation_is_saved_and_never_executed(self):
        self.kind = 'length'
        with self.assertRaises(LimitsExceeded) as caught:
            self.agent.query()
        self.assertEqual(caught.exception.messages[0]['reasoning_content'], 'TRACE-1')
        self.assertEqual(caught.exception.messages[0]['extra']['actions'], [])
        self.assertEqual(len(self.calls), 1)

    def test_raw_argument_backend_preserves_json_error_for_next_turn(self):
        self.kind='raw_bad_json'
        with self.assertRaises(FormatError) as caught:
            self.agent.query()
        history=caught.exception.messages
        self.assertEqual(history[0]['tool_calls'][0]['function']['arguments'],'{oops')
        self.assertEqual(history[1]['role'],'tool')
        self.assertIn('Expecting property name',history[1]['content'])
        self.agent.add_messages(*history)
        self.kind='normal'
        self.agent.query()
        self.assertEqual(self.calls[-1]['messages'][-2]['tool_calls'][0]['function']['arguments'],'{oops')

    def test_socket_loss_fences_only_own_replica_without_hidden_sdk_retry(self):
        self.kind = 'drop'
        with self.assertRaises(Exception): self.model._query(self.agent.messages)
        self.assertEqual(len(self.calls), 1)
        self.assertTrue((self.root/'slots/poison.json').exists())
        row = json.loads(next((self.root/'http').glob('*.json')).read_text())
        self.assertIn('error_type', row)
        self.assertFalse(row['complete'])

    def test_fresh_http_connections_avoid_idle_reuse_disconnect(self):
        self.server.RequestHandlerClass.protocol_version = 'HTTP/1.1'
        self.kind = 'drop_reused'
        for _ in range(3):
            # Bypass the outer agent retry loop so a regression fails immediately.
            self.model._query(self.agent.messages)
        self.assertEqual(len(self.calls), 3)
        records = [json.loads(p.read_text()) for p in (self.root/'http').glob('*.json')]
        self.assertEqual(len(records), 3)
        self.assertTrue(all(row.get('complete') and row.get('status') == 200 for row in records))
        self.assertTrue(all(row.get('http_keepalive') is False for row in records))
        self.assertFalse((self.root/'slots/poison.json').exists())

    def test_partial_response_bytes_survive_disconnect(self):
        self.kind = 'partial'
        with self.assertRaises(Exception): self.model._query(self.agent.messages)
        record = json.loads(next((self.root/'http').glob('*.json')).read_text())
        body = next((self.root/'http').glob('*.body')).read_bytes()
        self.assertFalse(record['complete'])
        self.assertGreater(len(body), 0)
        self.assertEqual(record['response_sha256'], hashlib.sha256(body).hexdigest())

    def test_http_rejection_does_not_poison_cache_but_requires_diagnosis(self):
        self.kind = '400'
        with self.assertRaises(Exception): self.model._query(self.agent.messages)
        self.assertFalse((self.root/'slots/poison.json').exists())
        self.assertTrue((self.root/'systemic-failure.json').exists())

    def test_429_can_retry_without_poison_or_systemic_halt(self):
        self.kind = '429'
        with self.assertRaises(Exception): self.model._query(self.agent.messages)
        self.assertFalse((self.root/'slots/poison.json').exists())
        self.assertFalse((self.root/'systemic-failure.json').exists())
        self.kind = 'normal'
        self.agent.query()
        self.assertEqual(len(self.calls), 2)

    def test_500_is_conservatively_fenced(self):
        self.kind = '500'
        with self.assertRaises(Exception): self.model._query(self.agent.messages)
        self.assertTrue((self.root/'slots/poison.json').exists())

    def test_legacy_sampling_is_unchanged(self):
        config = {'model': {'model_kwargs': {'temperature': 0}}}
        before = copy.deepcopy(config)
        self.assertEqual(configure_protocol(config, {})['version'], 'legacy')
        self.assertEqual(config, before)


if __name__ == '__main__': unittest.main()
