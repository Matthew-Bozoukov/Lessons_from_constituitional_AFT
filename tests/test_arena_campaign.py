# ABOUTME: Verify resumable Arena orchestration and pod release before CPU judging.
# ABOUTME: Tests use local fixtures and fake serving/publication boundaries only.
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from omegaconf import OmegaConf

from src.eval import run_eval
from src.infra.endpoints.vllm import TargetSpec


def test_two_arm_restart_skips_completed_generation_and_releases_before_pool(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = OmegaConf.create({'campaign_dir': str(tmp_path / 'campaign'), 'generation': {'max_tokens': 12000}})
    config = tmp_path / 'config.yaml'
    OmegaConf.save(cfg, config)
    events = []
    specs = {k: TargetSpec(f'org/qwen36-0-{k}', 'Qwen/Qwen3.6-27B', True, 'think',
                          f'qwen36_0_{k}', 64, revision=k, base_revision='base') for k in ['plain', 'da-15']}
    class Server:
        def __init__(self, **kw): pass
        def ensure(self, spec):
            events.append('serve:' + spec.revision)
            return NS(spec=spec)
        def stop(self): events.append('stop')
    def runner(target, cfg, out, *, reference=''):
        (out / 'rollouts').mkdir(exist_ok=True)
        (out / 'rollouts/answers.jsonl').write_text('{}\n')
        return {'arm': target.spec.model_key}
    def publish(out, **kw):
        events.append('publish')
        return 'https://example.invalid/' + kw['model_key']
    attempts = []
    def pool(runs, cfg, out):
        assert events[-1] == 'release'
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError('interrupted judging')
        return {'pooled_from': [], 'model_key': 'vs_qwen36_0_plain', 'mode': 'think'}
    monkeypatch.setattr(run_eval, 'VllmServer', Server)
    monkeypatch.setattr(run_eval, '_credentials_preflight', lambda **kw: None)
    monkeypatch.setattr(run_eval, '_publish', publish)
    monkeypatch.setattr(run_eval, 'resolve_target', lambda path: next(s for s in specs.values() if s.hf_path == path))
    monkeypatch.setattr(run_eval, 'resolve_pool', lambda _: pool)
    args = NS(config=str(config), name='arena_hard', overrides=[], no_push=True,
              target=[specs['da-15'].hf_path], server=None, port=18943)
    kwargs = ['--reference=' + specs['plain'].hf_path]
    with pytest.raises(RuntimeError, match='interrupted judging'):
        run_eval._run(args, kwargs, lambda: events.append('release'), runner=runner)
    assert events.count('publish') == 2
    events.clear()
    run_eval._run(args, kwargs, lambda: events.append('release'), runner=runner)
    assert events == ['stop', 'release', 'publish']
    cfg.generation.max_tokens = 6000
    OmegaConf.save(cfg, config)
    with pytest.raises(ValueError, match='inputs changed'):
        run_eval._run(args, kwargs, runner=runner)


def test_receipt_rejects_modified_answers(tmp_path):
    spec = NS(model_key='arm')
    root = tmp_path / 'arms/arm/rollouts'
    root.mkdir(parents=True)
    answer = root / 'answers.jsonl'
    answer.write_bytes(b'original')
    receipt = {'sha256': {'rollouts/answers.jsonl': hashlib.sha256(b'original').hexdigest()}}
    (tmp_path / 'arm.json').write_text(json.dumps(receipt))
    assert run_eval._arena_receipt(tmp_path, spec) == receipt
    answer.write_bytes(b'changed')
    with pytest.raises(ValueError, match='artifact changed'):
        run_eval._arena_receipt(tmp_path, spec)


def test_raw_transcript_write_uses_utf8_under_windows_locale(tmp_path, monkeypatch):
    import ast
    from src.eval.capabilities.arena_hard import arena_hard_gen
    tree = ast.parse(Path(arena_hard_gen.__file__).read_text(encoding='utf-8'))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.Expr)
                and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                and n.value.func.attr == 'write_text'
                and any(isinstance(v, ast.Constant) and v.value == 'raw_samples.md'
                        for v in ast.walk(n.value.func.value)))
    original = Path.write_text
    def cp1252(path, text, encoding=None, **kw):
        return original(path, text, encoding=encoding or 'cp1252', **kw)
    monkeypatch.setattr(Path, 'write_text', cp1252)
    record = dict(uid='one', category='hard_prompt', finish_reason='stop',
                  prompt='Explain α', think='理由', answer='✓')
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<raw-transcript>', 'exec'),
         dict(out_dir=tmp_path, arm='arm', served='model', dump=[record]))
    assert '理由' in (tmp_path / 'raw_samples.md').read_text(encoding='utf-8')


def test_exact_budget_uses_pinned_template_and_requires_revision(monkeypatch):
    from transformers import AutoTokenizer
    from src.eval.capabilities.arena_hard.arena_hard_gen import prompt_token_counter, output_budget_policy
    cfg = OmegaConf.create({'base_model': 'org/base', 'generation': {'exact_prompt_budget': True},
                            'target_identity': {'base_revision': 'base-pin', 'mode': 'think'}})
    calls = []
    class Tokenizer:
        chat_template = 'template'
        def apply_chat_template(self, messages, **kwargs):
            assert 'set enable_thinking = true' in self.chat_template
            assert 'set preserve_thinking = false' in self.chat_template
            assert kwargs == {'tokenize': False, 'add_generation_prompt': True}
            return 'rendered'
        def encode(self, text, **kwargs):
            assert text == 'rendered' and kwargs == {'add_special_tokens': False}
            calls.append(text)
            return list(range(8449))
    def load(base, revision):
        assert (base, revision) == ('org/base', 'base-pin')
        return Tokenizer()
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', load)
    count = prompt_token_counter(cfg)
    assert count('prompt') == count('prompt') == 8449
    assert len(calls) == 1
    assert output_budget_policy(cfg).endswith('-pinned-tokenizer-cap-v1')
    del cfg.target_identity.base_revision
    with pytest.raises(ValueError, match='pinned HF base'):
        prompt_token_counter(cfg)
