# ABOUTME: Explicit sampling, lossless rejected-response history and durable HTTP evidence for Lite.
# ABOUTME: Records transport failures without confusing HTTP rejection with ambiguous running inference.
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

from src.eval.capabilities.swebench_mini.fleet_state import atomic


def configure_protocol(config, request):
    """Historical requests retain their old policy; new protocols must pin all parameters."""
    if 'sampling' not in request:
        return {'version': 'legacy', 'preserve_format_errors': False}
    sampling = request['sampling']
    required = {'temperature', 'top_p', 'top_k', 'min_p', 'presence_penalty',
                'frequency_penalty', 'repetition_penalty'}
    assert set(sampling) == required, 'Pin the complete sampling recipe'
    assert all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
               for v in sampling.values()), 'Invalid sampling value'
    assert 0 < sampling['temperature'] <= 2 and 0 < sampling['top_p'] <= 1
    assert isinstance(sampling['top_k'], int) and sampling['top_k'] > 0
    assert 0 <= sampling['min_p'] <= 1 and 0 < sampling['repetition_penalty'] <= 2
    assert all(-2 <= sampling[k] <= 2 for k in ('presence_penalty', 'frequency_penalty'))
    options = config['model']['model_kwargs']
    options.update({k: sampling[k] for k in ('temperature', 'top_p', 'presence_penalty', 'frequency_penalty')})
    # These are vLLM extensions, not OpenAI parameters: never let drop_params swallow them.
    options['extra_body'] = dict(options.get('extra_body', {}),
        **{k: sampling[k] for k in ('top_k', 'min_p', 'repetition_penalty')})
    options['drop_params'] = False
    return {'version': request['protocol_version'], 'sampling': dict(sampling),
            'preserve_format_errors': True, 'http_evidence': 'content-addressed-messages-and-raw-response-v1'}


def rejected_history(raw, corrections):
    """Retain reasoning; do not execute any member of a rejected tool-call batch."""
    message = copy.deepcopy(raw['choices'][0]['message'])
    message['role'] = 'assistant'
    calls = message.get('tool_calls') or []
    renderable = True
    ids = set()
    for call in calls:
        try:
            args = json.loads(call['function']['arguments'])
            assert isinstance(args, dict) and call['id'] and call['id'] not in ids
            ids.add(call['id'])
        except (KeyError, TypeError, ValueError, AssertionError):
            renderable = False
    feedback = []
    if calls and renderable:
        feedback = [{'role': 'tool', 'tool_call_id': call['id'],
                     'content': 'Tool call rejected by the agent; no commands in this response were executed.'}
                    for call in calls]
    elif calls:
        # Invalid argument JSON cannot pass the Qwen template's mapping renderer.
        # Keep its exact bytes in raw evidence and quote the rejected call in history.
        message['content'] = (message.get('content') or '') + '\n\n[Rejected tool calls; not executed]\n' + json.dumps(calls, ensure_ascii=False)
        message.pop('tool_calls', None)
    message['extra'] = {'response': raw, 'actions': [], 'format_error_preserved': True,
                        'tool_calls_quoted': bool(calls and not renderable)}
    return [message, *feedback, *corrections]


def ambiguous_failure(exc):
    """A completed 4xx rejection is not an orphaned decode. Unknown failures are conservative."""
    status = getattr(exc, 'status_code', None)
    if isinstance(status, int) and 400 <= status < 500 and status != 408:
        return False
    return True


def make_audit_client(request):
    """Use the real hosted_vllm HTTPX path and persist bytes before LiteLLM parsing.

    Prefix messages are stored once by hash rather than copying growing history on
    every turn. Headers/credentials are never recorded. Partial response bytes survive
    a dropped socket. This does not expose vLLM's internal pre-parser generations.
    """
    import httpx
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    directory = Path(request['out']) / 'http'
    directory.mkdir(parents=True, exist_ok=True)
    blobs = directory / 'messages'
    blobs.mkdir(exist_ok=True)

    class RecordingStream(httpx.SyncByteStream):
        def __init__(self, inner, path, record):
            self.inner, self.path, self.record = inner, path, record

        def __iter__(self):
            digest = hashlib.sha256()
            count = 0
            started = time.monotonic()
            complete = False
            try:
                with self.path.with_suffix('.body').open('wb') as stream:
                    try:
                        for chunk in self.inner:
                            stream.write(chunk)
                            stream.flush()
                            digest.update(chunk)
                            count += len(chunk)
                            yield chunk
                    finally:
                        # Preserve bytes already received even when the socket dies.
                        os.fsync(stream.fileno())
                complete = True
            finally:
                atomic(self.path, dict(self.record, complete=complete, response_bytes=count,
                    response_sha256=digest.hexdigest(), seconds=time.monotonic()-started))

        def close(self):
            self.inner.close()

    class RecordingTransport(httpx.BaseTransport):
        def __init__(self):
            self.inner = httpx.HTTPTransport(retries=0)

        def handle_request(self, http_request):
            identifier = uuid.uuid4().hex
            http_request.headers['x-request-id'] = identifier
            body = json.loads(http_request.content)
            references = []
            for message in body.pop('messages', []):
                encoded = json.dumps(message, ensure_ascii=False, sort_keys=True).encode()
                sha = hashlib.sha256(encoded).hexdigest()
                path = blobs / (sha + '.json')
                if not path.exists():
                    atomic(path, message)
                references.append(sha)
            path = directory / (identifier + '.json')
            record = {'id': identifier, 'at': time.time(), 'path': http_request.url.path,
                      'parameters': body, 'message_hashes': references, 'complete': False,
                      'request_sha256': hashlib.sha256(http_request.content).hexdigest()}
            atomic(path, record)
            try:
                response = self.inner.handle_request(http_request)
            except Exception as exc:
                atomic(path, dict(record, error_type=type(exc).__name__))
                raise
            record.update(status=response.status_code,
                          content_encoding=response.headers.get('content-encoding', 'identity'))
            atomic(path, record)
            response.stream = RecordingStream(response.stream, path, record)
            return response

        def close(self):
            self.inner.close()

    timeout = request['model_request_timeout_seconds']
    return HTTPHandler(timeout=timeout,
        client=httpx.Client(transport=RecordingTransport(), timeout=timeout, trust_env=False))


def install_protocol(model_class, request):
    """Install once in each isolated agent subprocess, before the budget wrapper."""
    if 'sampling' not in request:
        return None
    from minisweagent.exceptions import FormatError
    original_query, original_parse = model_class._query, model_class._parse_actions
    client = make_audit_client(request)

    def query(self, messages, **kwargs):
        try:
            return original_query(self, messages, **dict(kwargs, client=client))
        except Exception as exc:
            if getattr(exc, 'status_code', None) in (400, 401, 403, 404, 422) and type(exc).__name__ != 'ContextWindowExceededError':
                atomic(Path(request['out']) / 'systemic-failure.json',
                       {'kind': 'request_rejected', 'status': exc.status_code, 'error_type': type(exc).__name__})
            raise

    def parse(self, response):
        raw = response.model_dump()
        # Capture before action parsing, even if that parser rejects the response.
        try:
            actions = original_parse(self, response)
            if any(not isinstance(action.get('command'), str) for action in actions):
                raise ValueError('Bash command must be a string')
            return actions
        except FormatError as exc:
            raise FormatError(*rejected_history(raw, exc.messages)) from exc
        except (TypeError, ValueError, KeyError) as exc:
            # Upstream can raise TypeError for scalar/null JSON arguments.
            correction = {'role': 'user', 'content': 'Tool call error: invalid tool arguments. Use the bash tool with a JSON object containing a string command.',
                          'extra': {'interrupt_type': 'FormatError'}}
            raise FormatError(*rejected_history(raw, [correction])) from exc

    model_class._query, model_class._parse_actions = query, parse
    return client
