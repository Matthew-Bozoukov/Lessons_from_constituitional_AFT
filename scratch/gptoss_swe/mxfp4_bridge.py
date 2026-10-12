# ABOUTME: Send our unchanged reference Harmony token prompts to raw vLLM completions.
# ABOUTME: Retain exact output tokens and explicit stop boundaries without template or JSON repair.
import json
import os
from pathlib import Path
from types import SimpleNamespace
import uuid

import httpx


def request_body(prompt_ids, options):
    stop = list(options.stop)
    if not stop or any(not isinstance(t, int) for t in stop):
        raise ValueError('Expected reference Harmony integer stop tokens')
    body = dict(model='gptoss-mxfp4', prompt=prompt_ids, n=1, stream=False,
        max_tokens=options.max_tokens, temperature=options.temperature,
        top_p=options.top_p, top_k=options.top_k if options.top_k > 0 else -1,
        min_p=0.0, presence_penalty=0.0, frequency_penalty=0.0,
        repetition_penalty=1.0, stop_token_ids=stop, ignore_eos=True,
        add_special_tokens=False, skip_special_tokens=False,
        include_stop_str_in_output=True, return_token_ids=True)
    if getattr(options, 'seed', None) is not None:
        body['seed'] = options.seed
    return body


def sample_result(body, response):
    if len(response['choices']) != 1:
        raise ValueError('Unexpected number of raw vLLM choices')
    choice = response['choices'][0]
    ids = choice.get('token_ids')
    if not isinstance(ids, list) or any(not isinstance(i, int) for i in ids):
        raise ValueError('Raw token IDs missing; text retokenization is forbidden')
    if choice.get('prompt_token_ids') != body['prompt']:
        raise ValueError('vLLM changed or did not echo the exact Harmony prompt IDs')
    if response['usage']['prompt_tokens'] != len(body['prompt']):
        raise ValueError('Prompt token accounting differs')
    if response['usage']['completion_tokens'] != len(ids):
        raise ValueError('Output token accounting differs; refuse hidden stop-token loss')
    if len(ids) > body['max_tokens']:
        raise ValueError('Response exceeded allowance')
    reason = choice['finish_reason']
    if reason == 'stop':
        if not ids or ids[-1] not in body['stop_token_ids']:
            raise ValueError('Explicit Harmony stop token missing; never synthesize it')
    elif reason != 'length':
        raise ValueError('Unrecognized finish reason: ' + str(reason))
    return SimpleNamespace(sequences=[SimpleNamespace(tokens=ids, stop_reason=reason)],
        prompt_cache_hit_tokens=int((response['usage'].get('prompt_tokens_details') or {}).get('cached_tokens') or 0))


class RawClient:
    def __init__(self, url, key, traces):
        self.url, self.key, self.traces = url, key, Path(traces)

    async def sample_async(self, *, prompt, num_samples, sampling_params):
        assert num_samples == 1
        body = request_body(prompt.to_ints(), sampling_params)
        trace = self.traces / uuid.uuid4().hex
        trace.mkdir(parents=True)
        (trace/'request.json').write_text(json.dumps(body), encoding='utf-8')
        # A new connection per request avoids Qwen's observed stale keepalive resets.
        # No automatic retry after an ambiguous paid dispatch.
        async with httpx.AsyncClient(timeout=7200, limits=httpx.Limits(max_keepalive_connections=0)) as client:
            try:
                response = await client.post(self.url+'/v1/completions', json=body,
                    headers={'Authorization': 'Bearer '+self.key})
                (trace/'response.txt').write_text(response.text, encoding='utf-8')
                response.raise_for_status()
                return sample_result(body, response.json())
            except BaseException as exc:
                (trace/'error.json').write_text(json.dumps(dict(error=repr(exc))), encoding='utf-8')
                raise


def main():
    import uvicorn
    from tinker_cookbook.renderers.base import ToolCall
    from tinker_cookbook.third_party.openai_compat import openai_tools_to_tinker
    from src.infra.endpoints.tinker_server import Runtime, create_app
    from src.infra.endpoints.tinker_harmony import HarmonyRenderer
    runtime = Runtime(checkpoint='vllm://openai/gpt-oss-120b@'+os.environ['MXFP4_REVISION'],
        model='openai/gpt-oss-120b', reasoning='medium',
        renderer=HarmonyRenderer(reasoning_effort='medium', current_date='2026-10-12'),
        sampling_client=RawClient(os.environ['MXFP4_URL'], os.environ['MXFP4_API_KEY'], os.environ['MXFP4_RAW_TRACES']),
        sampling_params=SimpleNamespace, tool_call_type=ToolCall, convert_tools=openai_tools_to_tinker,
        api_key=os.environ['TINKER_API_KEY'], instance_id=os.environ['MXFP4_INSTANCE'],
        sampling_model='openai/gpt-oss-120b:MXFP4',
        default_max_tokens=None if os.environ['DEFAULT_MAX_TOKENS']=='remaining' else int(os.environ['DEFAULT_MAX_TOKENS']),
        context_window=int(os.environ['TINKER_CONTEXT_WINDOW']), trace_dir=os.environ['TINKER_TRACE_DIR'])
    # The reused wire field named tinker_metadata denotes renderer diagnostics, not provider identity.
    uvicorn.run(create_app(runtime), host='0.0.0.0', port=1234, log_level='warning')


if __name__ == '__main__':
    main()
