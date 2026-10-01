# ABOUTME: Inspect transport contract and safety regression tests, no GPU or paid APIs.
# ABOUTME: Run in the pinned Linux Inspect environment with python -m unittest tests.test_swebench_inspect.
import os
import unittest
if os.name != 'posix':
    raise unittest.SkipTest('Shared fleet admission uses Linux locks')
try:
    import inspect_ai
    import httpx2 as httpx
except ImportError:
    raise unittest.SkipTest('Run in the pinned Inspect environment')
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch
import yaml

from src.eval.capabilities.swebench_mini.inspect_transport import InspectTransport
from src.eval.capabilities.swebench_mini.fleet_state import read
from src.eval.capabilities.swebench_mini.inspect_task import make_task


class Wire(httpx.AsyncByteStream):
    def __init__(self, data, fail=False): self.data, self.fail = data, fail
    async def __aiter__(self):
        yield self.data
        if self.fail: raise httpx.ReadError('synthetic partial disconnect')


class TransportTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.sampling=yaml.safe_load(Path('configs/eval/swebench_mini/lite.yaml').read_text())['sampling']
        self.request=dict(out=str(self.root),sampling=self.sampling,endpoint='http://127.0.0.1/v1',
            step_limit=3,max_task_tokens=90,max_response_tokens=60,context_window=200,
            token_admission=dict(directory=str(self.root/'slots'),budget_tokens=500,
                                 expires=time.time()+60,fairness_seconds=.01))
        self.status=200;self.fail=False;self.wires=[]
        self.data={'choices':[{'finish_reason':'tool_calls','message':{
            'role':'assistant','content':'ok','reasoning':'preserved','tool_calls':[
                {'id':'a','type':'function','function':{'name':'bash','arguments':'{"command":"true"}'}}]}}],
            'usage':{'prompt_tokens':20,'completion_tokens':30}}
        def handle(req):
            self.wires.append(json.loads(req.content))
            return httpx.Response(self.status,stream=Wire(json.dumps(self.data).encode(),self.fail))
        self.transport=InspectTransport(self.request,httpx.MockTransport(handle))
        self.tokens=patch('src.eval.capabilities.swebench_mini.inspect_transport.prompt_tokens',return_value=20)
        self.tokens.start();self.addCleanup(self.tokens.stop)

    async def generate(self, **overrides):
        body=dict(self.sampling,model='test',messages=[{'role':'user','content':'test'}],stream=False)
        body.update(overrides)
        return await self.transport.handle_async_request(httpx.Request('POST','http://127.0.0.1/v1/chat/completions',json=body))

    async def test_clamps_output_against_remaining_and_records_wire(self):
        await self.generate();await self.generate();await self.generate()
        self.assertEqual([w['max_tokens'] for w in self.wires],[60,60,30])
        self.assertEqual(self.transport.terminal,'task_token_limit')
        self.assertEqual(len(list((self.root/'http').glob('*.body'))),3)
        self.assertEqual(self.transport.calls,3)
        with self.assertRaisesRegex(RuntimeError,'terminal'): await self.generate()
        self.assertEqual(len(self.wires),3)

    async def test_context_limit_never_calls_inference(self):
        self.request['context_window']=20
        response=await self.generate()
        self.assertEqual(response.status_code,400)
        self.assertEqual(self.transport.terminal,'context_limit')
        self.assertEqual(self.wires,[])

    async def test_length_is_terminal(self):
        self.data['choices'][0]['finish_reason']='length'
        await self.generate()
        self.assertEqual(self.transport.terminal,'response_token_limit')

    async def test_malformed_success_is_systemic_and_fenced(self):
        self.data['usage']['completion_tokens']=10000
        with self.assertRaises(AssertionError):await self.generate()
        self.assertTrue((self.root/'systemic-failure.json').exists())
        self.assertTrue((self.root/'slots/poison.json').exists())

    async def test_no_tool_reasoning_response_is_retained(self):
        self.data['choices'][0]['message'].pop('tool_calls')
        self.data['choices'][0]['finish_reason']='stop'
        await self.generate()
        self.assertEqual(self.transport.responses[0]['choices'][0]['message']['reasoning'],'preserved')
        self.assertFalse(self.transport.rejected_batch)

    async def test_step_guard(self):
        self.request['step_limit']=1
        await self.generate()
        with self.assertRaisesRegex(RuntimeError,'terminal'):await self.generate()

    async def test_missing_sampling_fails_before_any_model_call(self):
        with self.assertRaisesRegex(AssertionError,'top_k'):await self.generate(top_k=None)
        self.assertFalse(self.wires)

    async def test_exact_token_mismatch_fences(self):
        self.data['usage']['prompt_tokens']=21
        with self.assertRaisesRegex(RuntimeError,'tokenization'):await self.generate()
        self.assertTrue((self.root/'tokenization-mismatch.json').exists())
        self.assertTrue((self.root/'systemic-failure.json').exists())
        self.assertTrue((self.root/'slots/poison.json').exists())

    async def test_partial_body_saved_and_fenced(self):
        self.fail=True
        with self.assertRaises(httpx.ReadError):await self.generate()
        self.assertTrue((self.root/'slots/poison.json').exists())
        self.assertTrue(next((self.root/'http').glob('*.body')).read_bytes())
        self.assertFalse(read(next((self.root/'http').glob('*.json')))['complete'])

    async def test_server_error_fences(self):
        self.status=503
        await self.generate()
        self.assertTrue((self.root/'slots/poison.json').exists())

    async def test_request_rejection_is_systemic(self):
        self.status=422
        await self.generate()
        self.assertEqual(read(self.root/'systemic-failure.json')['status'],422)

    async def test_rate_limit_does_not_poison(self):
        self.status=429
        await self.generate()
        self.assertFalse((self.root/'slots/poison.json').exists())
        self.assertEqual(self.transport.calls,0)

    async def test_rejects_entire_malformed_or_unknown_batch(self):
        for call in ({'name':'bash','arguments':'null'}, {'name':'bash','arguments':'{broken'},
                     {'name':'bash','arguments':'{"command":4}'}, {'name':'other','arguments':'{}'},
                     {'name':'submit','arguments':'{}'}, {'name':'bash','arguments':'{"command":"true","unexpected":1}'}):
            self.data['choices'][0]['message']['tool_calls'][0]['function']=call
            self.transport.calls=0;self.transport.total=0
            await self.generate()
            self.assertTrue(self.transport.rejected_batch,call)

    async def test_unknown_endpoint_refused(self):
        with self.assertRaisesRegex(RuntimeError,'unexpected'):
            await self.transport.handle_async_request(httpx.Request('GET','http://127.0.0.1/v1/models'))


class TaskContractTests(unittest.TestCase):
    def test_native_task_uses_frozen_image_no_gold_and_output_turn_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            request=dict(out=directory,instance=dict(instance_id='django__django-11099',
                problem_statement='Synthetic issue',base_commit='a'*40,version='1.9',repo='django/django',
                patch='GOLD MUST BE REMOVED',test_patch='GOLD TEST',hints_text='SECRET HINT'),
                image='swebench/example@sha256:'+'b'*64,cpus=2,memory='4g',pids=512,
                environment={},campaign='synthetic',attempt='synthetic',step_limit=250,max_task_tokens=65536,
                protocol_version='lite-inspect-v1',sampling={'temperature':1},inspect={'instructions':'Synthetic instructions'})
            task=make_task(request,None)
            self.assertEqual(task.turn_limit,250)
            self.assertEqual(task.token_limit,65536)
            self.assertEqual(task.token_limit_type,'output')
            self.assertIsNone(task.message_limit)
            self.assertFalse(task.scorer)
            self.assertEqual(task.metadata['agent_backend'],'inspect')
            sample=list(task.dataset)[0]
            self.assertEqual(sample.metadata['image_name'],request['image'])
            self.assertEqual(sample.metadata['patch'],'')
            self.assertEqual(sample.metadata['hints_text'],'')
            self.assertNotIn('GOLD',(Path(directory)/'frozen-dataset/test.jsonl').read_text())
            compose=yaml.safe_load((Path(directory)/'compose.yaml').read_text())['services']['default']
            self.assertEqual(compose['pull_policy'],'never')
            self.assertEqual(compose['network_mode'],'none')


if __name__=='__main__':unittest.main()
