# ABOUTME: Offline HTTP and real cookbook-renderer compatibility tests for GPT-OSS Tinker tools.
# ABOUTME: Run with uv run --project src/infra/endpoints/tinker_env --frozen python -m pytest tests/test_tinker_server.py.

from types import SimpleNamespace

import pytest

pytest.importorskip("tinker_cookbook", reason="run this compatibility suite in the locked tinker_env")
import tinker
from fastapi.testclient import TestClient
from tinker_cookbook.renderers.base import ToolCall
from tinker_cookbook.renderers.gpt_oss import GptOssRenderer
from tinker_cookbook.third_party.openai_compat import openai_tools_to_tinker

from src.infra.endpoints.tinker_server import Runtime, build_messages, create_app


class TextCodec:
    """Tiny reversible codec; actual upstream renderer/SDK types remain under test."""
    specials = {"<|return|>": 200002, "<|call|>": 200012}
    def encode(self, text, **kwargs):
        result = []
        while text:
            for marker, token in self.specials.items():
                if text.startswith(marker):
                    result.append(token)
                    text = text[len(marker):]
                    break
            else:
                result.append(ord(text[0]))
                text = text[1:]
        return result

    def decode(self, tokens, **kwargs):
        inverse = {v: k for k, v in self.specials.items()}
        return "".join(inverse[t] if t in inverse else chr(t) for t in tokens)


class Sampler:
    def __init__(self, codec):
        self.codec, self.requests = codec, []
        self.reply = '<|channel|>analysis<|message|>Think.<|end|><|start|>assistant to=functions.bash<|channel|>commentary <|constrain|>json<|message|>{"command":"pwd"}<|call|>'
        self.stop_reason = "stop"

    async def sample_async(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(sequences=[SimpleNamespace(tokens=self.codec.encode(self.reply), stop_reason=self.stop_reason)])


@pytest.fixture
def runtime():
    codec = TextCodec()
    return Runtime(
        checkpoint="tinker://example/sampler_weights/test", model="openai/gpt-oss-120b", reasoning="medium",
        renderer=GptOssRenderer(codec, use_system_prompt=True, reasoning_effort="medium", current_date="2026-09-29"),
        sampling_client=Sampler(codec), sampling_params=tinker.SamplingParams, tool_call_type=ToolCall,
        convert_tools=openai_tools_to_tinker, api_key="test-key", instance_id="test-instance")


def client(runtime):
    return TestClient(create_app(runtime), headers={"Authorization": "Bearer test-key"})


def request(**kwargs):
    return {"model": "openai/gpt-oss-120b", "messages": [{"role": "user", "content": "Inspect the repository"}],
            "tools": [{"type": "function", "function": {"name": "bash", "parameters": {
                "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}],
            **kwargs}


def test_harmony_tool_result_and_reasoning_roundtrip(runtime):
    http = client(runtime)
    body = request(temperature=0.4, top_p=0.93, top_k=20, seed=0, max_tokens=1000)
    first = http.post("/v1/chat/completions", json=body)
    assert first.status_code == 200, first.text
    result = first.json()
    choice = result["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    msg = choice["message"]
    assert msg["reasoning_content"] == "Think."
    cid = msg["tool_calls"][0]["id"]
    assert isinstance(cid, str) and cid.startswith("call_")
    params = runtime.sampling_client.requests[0]["sampling_params"]
    assert (params.temperature, params.top_p, params.top_k, params.seed, params.max_tokens) == (0.4, 0.93, 20, 0, 1000)
    assert params.stop == [200002, 200012]
    body["messages"] += [msg, {"role": "tool", "tool_call_id": cid, "content": "/testbed"}]
    runtime.sampling_client.reply = '<|channel|>final<|message|>Done.<|return|>'
    second = http.post("/v1/chat/completions", json=body)
    assert second.status_code == 200, second.text
    prompt = runtime.renderer.tokenizer.decode(runtime.sampling_client.requests[1]["prompt"].to_ints())
    assert "analysis<|message|>Think." in prompt
    assert "functions.bash to=assistant<|channel|>commentary<|message|>/testbed" in prompt


def test_qwen_sampling_and_token_count_match_real_rendering(runtime):
    http = client(runtime)
    body = request(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
        repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0, max_tokens=16384)
    first = http.post('/v1/chat/completions', json=body)
    assert first.status_code == 200, first.text
    msg = first.json()['choices'][0]['message']
    body['messages'] += [msg, dict(role='tool', tool_call_id=msg['tool_calls'][0]['id'], content='shell output')]
    count = http.post('/tokenize', json=body).json()
    second = http.post('/v1/chat/completions', json=body)
    assert second.status_code == 200, second.text
    assert count['count'] == second.json()['usage']['prompt_tokens']
    actual = runtime.sampling_client.requests[-1]
    assert count['tokens'] == actual['prompt'].to_ints()
    params = actual['sampling_params']
    assert (params.temperature, params.top_p, params.top_k, params.max_tokens) == (1.0, .95, 20, 16384)


def test_exhausted_budget_never_calls_sampler(runtime):
    class Exhausted:
        def reserve(self, *args):
            raise RuntimeError('Frozen Tinker spending ceiling reached')
    runtime.budget = Exhausted()
    response = client(runtime).post('/v1/chat/completions', json=request())
    assert response.status_code == 402
    assert not runtime.sampling_client.requests
    assert second.json()["choices"][0]["message"]["content"] == "Done."
    assert result["tinker_metadata"]["checkpoint"] == runtime.checkpoint


@pytest.mark.parametrize("control", [
    {"tool_choice": "required"}, {"tool_choice": "none"}, {"tool_choice": {"type": "function", "function": {"name": "bash"}}},
    {"parallel_tool_calls": False}, {"stop": ["END"]}, {"presence_penalty": 0.5},
    {"reasoning_effort": "high"}, {"response_format": {"type": "json_object"}},
    {"stream": True}, {"n": 2}, {"max_tokens": 0}, {"top_p": 0}, {"seed": -1},
    {"extra_unknown_sampling": 1}, {"max_tokens": 10, "max_completion_tokens": 20},
])
def test_unsupported_or_invalid_controls_refuse_before_sampling(runtime, control):
    response = client(runtime).post("/v1/chat/completions", json=request(**control))
    assert response.status_code == 400
    assert runtime.sampling_client.requests == []


def test_truncated_parsed_tool_is_never_executable(runtime):
    runtime.sampling_client.stop_reason = "length"
    result = client(runtime).post("/v1/chat/completions", json=request()).json()
    assert result["choices"][0]["finish_reason"] == "length"
    assert "tool_calls" not in result["choices"][0]["message"]
    assert '"command":"pwd"' in result["tinker_metadata"]["raw_completion"]


def test_malformed_tool_is_retained_but_not_executable(runtime):
    runtime.sampling_client.reply = '<|channel|>commentary to=functions.bash<|message|>{bad json<|call|>'
    result = client(runtime).post("/v1/chat/completions", json=request()).json()
    assert "tool_calls" not in result["choices"][0]["message"]
    assert result["tinker_metadata"]["unparsed_tool_calls"]


def test_wrong_credentials_model_context_are_rejected(runtime):
    assert TestClient(create_app(runtime)).get("/v1/models").status_code == 401
    assert client(runtime).post("/v1/chat/completions", json=request(model="another-model")).status_code == 400
    runtime.context_window = 100
    assert client(runtime).post("/v1/chat/completions", json=request(max_tokens=100)).status_code == 400
    assert runtime.sampling_client.requests == []
    info = client(runtime).get("/v1/models").json()["data"][0]
    assert info["checkpoint"] == runtime.checkpoint and info["instance_id"] == runtime.instance_id


@pytest.mark.parametrize("messages", [
    [{"role": "tool", "tool_call_id": "missing", "content": "result"}],
    [{"role": "assistant", "tool_calls": [{"id": None, "function": {"name": "bash", "arguments": "{}"}}]}],
    [{"role": "assistant", "tool_calls": [{"id": "call1", "function": {"name": "bash", "arguments": "{}"}}]},
     {"role": "tool", "tool_call_id": "call1", "name": "other", "content": "result"}],
])
def test_ambiguous_history_never_renders_unknown_function(runtime, messages):
    response = client(runtime).post("/v1/chat/completions", json=request(messages=messages))
    assert response.status_code == 400
    assert runtime.sampling_client.requests == []


def test_late_system_instruction_retains_its_position(runtime):
    messages = [{"role": "system", "content": "First"}, {"role": "user", "content": "Question"},
                {"role": "system", "content": "Later"}]
    converted = build_messages(messages, None, runtime)
    assert [m["content"] for m in converted] == ["First", "Question", "Later"]


def test_pinned_mini_swe_client_uses_the_shim_tools_history_and_sampling(runtime):
    """Actual client/HTTP/renderer integration; sampler and tool execution remain synthetic."""
    import os
    import socket
    import subprocess
    import threading
    import time
    from pathlib import Path
    import uvicorn

    root = Path(__file__).resolve().parents[1]
    python = root / "src/eval/capabilities/swebench_mini/envs/agent/.venv" / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        pytest.skip("sync the existing pinned mini-SWE-agent environment to run its HTTP compatibility check")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(runtime), log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        code = '''
import os
from minisweagent.models.litellm_model import LitellmModel
m = LitellmModel(model_name="hosted_vllm/openai/gpt-oss-120b", cost_tracking="ignore_errors",
    model_kwargs={"api_base": os.environ["TEST_SHIM_URL"], "api_key": "test-key", "drop_params": False,
                  "temperature": 1.0, "top_p": 0.95, "max_tokens": 16384,
                  "parallel_tool_calls": True, "extra_body": {"top_k": 20}})
history = [{"role": "user", "content": "Inspect the repository"}]
reply = m.query(history)
assert reply["tool_calls"][0]["id"].startswith("call_")
assert reply["reasoning_content"] == "Think."
history += [reply, {"role": "tool", "tool_call_id": reply["tool_calls"][0]["id"], "content": "/testbed"}]
reply2 = m.query(history)
assert reply2["tool_calls"][0]["function"]["name"] == "bash"
print("MINI_TINKER_ROUNDTRIP_OK")
'''
        completed = subprocess.run([str(python), "-c", code], cwd=root, capture_output=True, text=True,
            timeout=45, env={**os.environ, "TEST_SHIM_URL": f"http://127.0.0.1:{port}/v1",
                             "PYTHONIOENCODING": "utf-8", "MSWEA_SILENT_STARTUP": "1"})
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert "MINI_TINKER_ROUNDTRIP_OK" in completed.stdout
        assert len(runtime.sampling_client.requests) == 2
        params = runtime.sampling_client.requests[0]["sampling_params"]
        assert (params.temperature, params.top_p, params.top_k, params.max_tokens) == (1.0, .95, 20, 16384)
        prompt = runtime.renderer.tokenizer.decode(runtime.sampling_client.requests[1]["prompt"].to_ints())
        assert "analysis<|message|>Think." in prompt and "functions.bash to=assistant" in prompt
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()
