# ABOUTME: Inspect HTTP transport with shared KV admission, exact budgets and durable wire evidence.
# ABOUTME: Uses Inspect's native OpenAI-compatible serialization; never rents or starts a model server.
import asyncio
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import httpx2 as httpx

from src.eval.capabilities.swebench_mini.fleet_admission import prompt_tokens, output_allowance, token_slot, poison
from src.eval.capabilities.swebench_mini.fleet_state import atomic


class InspectTransport(httpx.AsyncBaseTransport):
    def __init__(self, request, inner=None):
        self.request = request
        self.root = Path(request['out'])
        self.inner = inner or httpx.AsyncHTTPTransport(retries=0)
        self.total = 0
        self.calls = 0
        self.terminal = None
        self.rejected_batch = False
        self.responses = []

    def fence(self, reason, systemic=False):
        if self.request.get('token_admission'):
            poison(self.request['token_admission']['directory'], reason)
        if systemic:
            atomic(self.root/'systemic-failure.json', {'kind': reason})

    async def handle_async_request(self, request):
        if request.url.path != '/v1/chat/completions':
            raise RuntimeError('Inspect attempted an unexpected model endpoint: '+request.url.path)
        if self.terminal or self.calls >= self.request['step_limit']:
            raise RuntimeError('Inspect requested generation after a terminal budget')
        body = json.loads(request.content)
        assert not body.get('stream'), 'Use non-streaming JSON; partial transport bytes are still saved'
        for key, expected in self.request['sampling'].items():
            assert body.get(key) == expected, f'Inspect dropped/changed sampling parameter {key}'
        count = await asyncio.to_thread(prompt_tokens, self.request['endpoint'], body['model'],
            body['messages'], body.get('tools', []), audit_dir=self.root/'tokenization')
        remaining = self.request['max_task_tokens'] - self.total
        allowance = output_allowance(count, self.request['max_response_tokens'], remaining,
                                     self.request['context_window'])
        if not allowance:
            self.terminal = 'context_limit'
            # A local context limit is terminal and does not contact the GPU.
            return httpx.Response(400, json={'error': {'message': 'context length exceeded',
                'type': 'invalid_request_error', 'code': 'context_length_exceeded'}}, request=request)
        body['max_tokens'] = allowance
        encoded = json.dumps(body, ensure_ascii=False).encode()
        identifier = uuid.uuid4().hex
        directory = self.root/'http'; directory.mkdir(parents=True, exist_ok=True)
        blobs = directory/'messages'; blobs.mkdir(exist_ok=True)
        refs = []
        for message in body['messages']:
            sha = hashlib.sha256(json.dumps(message, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if not (blobs/(sha+'.json')).exists():
                atomic(blobs/(sha+'.json'), message)
            refs.append(sha)
        record = {'id': identifier, 'at': time.time(), 'message_hashes': refs,
                  'parameters': {k:v for k,v in body.items() if k != 'messages'},
                  'request_sha256': hashlib.sha256(encoded).hexdigest(), 'complete': False}
        atomic(directory/(identifier+'.json'), record)
        headers = dict(request.headers)
        headers.pop('content-length', None)
        headers['x-request-id'] = identifier
        headers['accept-encoding'] = 'identity'
        forwarded = httpx.Request(request.method, request.url, content=encoded, headers=headers,
                                  extensions=request.extensions)
        admission = self.request.get('token_admission')
        slot = token_slot(admission['directory'], count+allowance, admission['budget_tokens'],
            expires=admission['expires'], fairness_seconds=admission['fairness_seconds']) if admission else nullcontext(0)
        waited = await asyncio.to_thread(slot.__enter__)
        started = time.monotonic()
        response = None
        try:
            response = await self.inner.handle_async_request(forwarded)
            raw = bytearray()
            with (directory/(identifier+'.body')).open('wb') as f:
                try:
                    async for chunk in response.aiter_raw():
                        f.write(chunk); f.flush(); raw.extend(chunk)
                finally:
                    os.fsync(f.fileno())
            record.update(complete=True, status=response.status_code,
                          response_sha256=hashlib.sha256(raw).hexdigest(), response_bytes=len(raw))
            atomic(directory/(identifier+'.json'), record)
            if response.status_code >= 500 or response.status_code == 408:
                self.fence('Ambiguous inference HTTP '+str(response.status_code))
            elif response.status_code in (400, 401, 403, 404, 422):
                atomic(self.root/'systemic-failure.json', {'kind':'request_rejected','status':response.status_code})
            if response.status_code == 200:
                data = json.loads(raw)
                usage = data['usage']
                assert isinstance(usage['completion_tokens'], int) and 0 <= usage['completion_tokens'] <= allowance
                if usage['prompt_tokens'] != count:
                    atomic(self.root/'tokenization-mismatch.json', {'expected':count,'actual':usage['prompt_tokens']})
                    self.fence('tokenization_mismatch', systemic=True)
                    raise RuntimeError('Serving tokenization mismatch')
                self.calls += 1
                self.total += usage['completion_tokens']
                self.responses.append(data)
                if data['choices'][0]['finish_reason'] == 'length':
                    self.terminal = ('context_limit' if allowance < min(remaining,self.request['max_response_tokens'])
                                     else 'response_token_limit')
                if self.total >= self.request['max_task_tokens']:
                    self.terminal = 'task_token_limit'
                self.rejected_batch = False
                for call in data['choices'][0]['message'].get('tool_calls') or []:
                    try:
                        args = json.loads(call['function']['arguments'])
                        assert isinstance(args, dict)
                        assert call['function']['name'] in {'bash','submit'}
                        parameter = 'command' if call['function']['name'] == 'bash' else 'answer'
                        assert set(args) == {parameter} and isinstance(args[parameter], str)
                    except (KeyError, TypeError, ValueError, AssertionError):
                        self.rejected_batch = True
                atomic(self.root/'generation-progress.json', {'calls':self.calls,'completion_tokens':self.total,
                                                             'terminal':self.terminal})
            return httpx.Response(response.status_code, content=bytes(raw),
                headers={k:v for k,v in response.headers.items() if k.lower() not in {'content-length','content-encoding','transfer-encoding'}},
                request=request)
        except BaseException as exc:
            # A dropped socket/cancel can leave an orphaned decode. Fence before releasing KV.
            if response is None or not record['complete']:
                self.fence('Ambiguous inference transport '+type(exc).__name__)
            elif response.status_code == 200:
                self.fence('Invalid inference response '+type(exc).__name__, systemic=True)
            atomic(directory/(identifier+'.json'), dict(record,error_type=type(exc).__name__))
            raise
        finally:
            if response is not None:
                await response.aclose()
            await asyncio.to_thread(slot.__exit__, None, None, None)
            with (self.root/'inference-timing.jsonl').open('a') as f:
                f.write(json.dumps({'prompt_tokens':count,'output_allowance':allowance,'queue_seconds':waited,
                                   'request_seconds':time.monotonic()-started,'time':time.time()})+'\n')

    async def aclose(self):
        await self.inner.aclose()
