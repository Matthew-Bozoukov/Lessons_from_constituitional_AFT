# ABOUTME: Cross-process token reservations for one replica, with bounded bypass and crash fencing.
# ABOUTME: Uses the serving tokenizer and measured cache capacity; never shortens output to fit a batch.
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import re
import time
import uuid
import urllib.request

from src.eval.capabilities.swebench_mini.fleet_state import atomic, lock, read


def cache_capacity(log, fraction, context):
    counts = re.findall(r'GPU KV cache size: ([\d,]+) tokens', log)
    assert counts and 0 < fraction < 1, 'Missing measured GPU cache capacity'
    measured = int(counts[-1].replace(',', ''))
    budget = math.floor(measured * fraction)
    assert budget >= context, 'GPU cannot admit one full-context request with headroom'
    return {'measured_tokens': measured, 'fraction': fraction, 'budget_tokens': budget}


def prompt_tokens(endpoint, model, messages, tools):
    # vLLM 0.26 ChatCompletionRequest normalizes this legacy alias before
    # validation; TokenizeChatRequest does not, so otherwise it drops reasoning
    # from every preceding assistant turn and silently undercounts the prompt.
    normalized = []
    for message in messages:
        message = dict(message)
        reasoning = message.pop('reasoning_content', None)
        if reasoning is not None and message.get('reasoning') is None:
            message['reasoning'] = reasoning
        if message.get('tool_calls') is not None:
            message['tool_calls'] = list(message['tool_calls'])
        normalized.append(message)
    payload = {'model': model.removeprefix('hosted_vllm/'), 'messages': normalized,
               'tools': tools, 'add_generation_prompt': True}
    request = urllib.request.Request(endpoint.removesuffix('/v1').rstrip('/') + '/tokenize',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=60) as response:
        count = json.load(response)['count']
    assert isinstance(count, int) and count > 0, 'Invalid serving tokenizer count'
    return count


def output_allowance(prompt, response_limit, remaining, context):
    assert prompt >= 0 and response_limit > 0 and remaining > 0
    return max(0, min(response_limit, remaining, context - prompt))


def poison(directory, reason):
    atomic(Path(directory) / 'poison.json', {'reason': reason, 'time': time.time()})


def live_records(directory, own):
    records = []
    for path in sorted(directory.glob('*.json')):
        if path.name == 'poison.json':
            raise ConnectionError('Replica fenced: ' + read(path)['reason'])
        row = read(path)
        if row['id'] != own:
            with (directory / (row['id'] + '.lease')).open('a') as stream:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    path.unlink()
                    (directory / (row['id'] + '.lease')).unlink(missing_ok=True)
                    if row['state'] == 'active':
                        poison(directory, 'Active request owner died; fence GPU before reuse')
                        raise ConnectionError('Active request owner died')
                    continue
        records.append(row)
    return records


def may_admit(records, own, capacity, now, fairness_seconds):
    active = sum(r['tokens'] for r in records if r['state'] == 'active')
    pending = sorted((r for r in records if r['state'] == 'waiting'), key=lambda r: (r['created'], r['id']))
    if not pending:
        return False
    # A large request can be bypassed briefly, then new admissions stop until it fits.
    candidates = pending[:1] if now - pending[0]['created'] >= fairness_seconds else pending
    chosen = next((r for r in candidates if active + r['tokens'] <= capacity), None)
    return chosen is not None and chosen['id'] == own


@contextmanager
def token_slot(directory, tokens, capacity, *, expires, fairness_seconds=30, poll=.1):
    assert 0 < tokens <= capacity, 'Request exceeds measured admission capacity'
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    identifier = uuid.uuid4().hex
    path = directory / (identifier + '.json')
    lease = directory / (identifier + '.lease')
    started = time.monotonic()
    row = {'id': identifier, 'tokens': tokens, 'state': 'waiting', 'created': time.time()}
    with lease.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            with lock(directory / '.mutex'):
                atomic(path, row)
            while time.time() < expires:
                with lock(directory / '.mutex'):
                    records = live_records(directory, identifier)
                    if may_admit(records, identifier, capacity, time.time(), fairness_seconds):
                        row['state'] = 'active'
                        atomic(path, row)
                        break
                time.sleep(poll)
            else:
                raise TimeoutError('GPU lease ended while waiting for token admission')
            yield time.monotonic() - started
        finally:
            with lock(directory / '.mutex'):
                path.unlink(missing_ok=True)
                lease.unlink(missing_ok=True)
