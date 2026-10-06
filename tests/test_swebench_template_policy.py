# ABOUTME: SWE-bench explicitly retains prior reasoning without changing other eval defaults.
# ABOUTME: Offline tests cover config validation and the actual server template write path.
import json
from pathlib import Path
from unittest.mock import Mock

from jinja2 import Template
from omegaconf import OmegaConf
import pytest

from src.infra.endpoints import vllm
from src.model_profile import model_profile


@pytest.mark.parametrize('config,preserve', [
    ('configs/eval/swebench_mini/lite.yaml', True),
    ('configs/eval/odcv/lite.yaml', False),
    ('configs/eval/mask.yaml', False),
])
def test_eval_policy_is_explicit_and_other_defaults_remain_false(config, preserve):
    cfg = OmegaConf.load(config)
    plan = vllm.plan_serving(model_profile('qwen36').serving, dict(cfg.serving),
                             'Qwen/Qwen3.6-27B', 'think')
    assert plan['preserve_thinking'] is preserve


@pytest.mark.parametrize('value', ['true', 1, None])
def test_non_boolean_policy_fails_before_serving(value):
    with pytest.raises(SystemExit, match='must be a boolean'):
        vllm.plan_serving({}, {'context_window': 1024, 'preserve_thinking': value}, 'qwen', 'think')


def test_preserving_history_requires_pinned_thinking():
    with pytest.raises(SystemExit, match='requires the pinned think mode'):
        vllm.plan_serving({}, {'context_window': 1024, 'preserve_thinking': True}, 'qwen', 'default')


@pytest.mark.parametrize('preserve', [False, True])
def test_server_writes_template_policy_from_plan(monkeypatch, tmp_path, preserve):
    tokenizer = tmp_path/'tokenizer.json'
    tokenizer.write_text(json.dumps({'chat_template': '{{ preserve_thinking }}|{{ enable_thinking }}'}))
    monkeypatch.setattr(vllm, 'hf_download', Mock(return_value=str(tokenizer)))
    monkeypatch.setattr(vllm, 'native_context_window', lambda *args: 262144)
    server = vllm.VllmServer(tmp_path, serve_requirements={'context_window': 1024,
                                                         'preserve_thinking': preserve})
    monkeypatch.setattr(server.executor, 'start_server', Mock())
    monkeypatch.setattr(server, '_wait_healthy', Mock())
    spec = vllm.TargetSpec(hf_path='Qwen/Qwen3.6-27B', base_model='Qwen/Qwen3.6-27B',
                           adapter=False, mode='think', model_key='qwen36', base_revision='pin', lora_rank=None)
    server._start(spec, None)
    argv = server.executor.start_server.call_args.args[0]
    template = Path(argv[argv.index('--chat-template')+1]).read_text()
    # Opposite request kwargs cannot override the recorded serving policy.
    assert Template(template).render(preserve_thinking=not preserve, enable_thinking=False) == f'{preserve}|True'
