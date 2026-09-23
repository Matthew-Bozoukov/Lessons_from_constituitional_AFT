# ABOUTME: Verify timeout propagation through the installed LiteLLM hosted_vllm HTTP path.
# ABOUTME: Run in the pinned agent environment; HTTP sends are mocked and no model is called.
import unittest
from unittest.mock import patch
import httpx
import litellm
from scratch.swebench_lite_task import configure_request_transport


class TransportTests(unittest.TestCase):
    def test_explicit_override_reaches_httpx(self):
        seen = []

        def answer(client, request, **kwargs):
            self.assertEqual(request.url.host, 'localhost')
            seen.append(request.extensions['timeout'])
            return httpx.Response(200, request=request, json={
                'id': 'probe', 'object': 'chat.completion', 'created': 0, 'model': 'probe',
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}})

        config = {'model': {'model_kwargs': {}}}
        with patch.object(httpx.Client, 'send', answer), patch.dict('os.environ'):
            for override in (False, True):
                if override:
                    configure_request_transport(config, {})
                litellm.completion(model='hosted_vllm/probe', messages=[{'role': 'user', 'content': 'test'}],
                    api_base='http://localhost:9999/v1', api_key='synthetic', **config['model']['model_kwargs'])
        self.assertEqual(seen[0]['read'], 600.0)
        self.assertEqual(seen[1]['read'], 1800.0)


if __name__ == '__main__':
    unittest.main()
