# ABOUTME: Native Arena token manifests and channel-safe extraction regression coverage.
# ABOUTME: Offline tests reject renderer drift and uncounted prompts without paid calls.
import hashlib
import json
from types import SimpleNamespace
import pytest
from omegaconf import OmegaConf
from src.eval.capabilities.arena_hard import arena_hard_gen as gen, runner


def fixture(tmp_path):
    cfg = OmegaConf.load('configs/eval/arena_hard.yaml')
    cfg.generation.native_protocol = dict(backend='tinker-native', reasoning='medium', render_date='2026-10-14')
    cfg.generation.native_token_counts = str(tmp_path/'counts.json')
    data = dict(native_protocol=OmegaConf.to_container(cfg.generation.native_protocol),
        rows=[dict(prompt_sha256=hashlib.sha256(b'hello').hexdigest(), prompt_tokens=131)])
    (tmp_path/'counts.json').write_text(json.dumps(data))
    return cfg


def test_native_exact_counts_reject_unseen_prompts(tmp_path):
    cfg = fixture(tmp_path)
    counter = gen.prompt_token_counter(cfg)
    assert counter('hello') == 131
    with pytest.raises(ValueError, match='absent'):
        counter('changed')


def test_native_renderer_identity_is_required_for_token_counts(tmp_path):
    cfg = fixture(tmp_path)
    cfg.generation.native_protocol.reasoning = 'high'
    with pytest.raises(ValueError, match='mismatch'):
        gen.prompt_token_counter(cfg)


def test_native_protocol_distinguishes_hf_and_native_answers(tmp_path):
    cfg = fixture(tmp_path)
    settings = runner._generation_settings(cfg,'think')
    assert settings['answer_extraction']['parser'] == 'native-harmony-final-channel-v1'
    assert settings['native_protocol']['reasoning'] == 'medium'
    assert gen.output_budget_policy(cfg) == 'native-harmony-exact-prompt-v1'
    del cfg.generation.native_protocol
    assert 'native_protocol' not in runner._generation_settings(cfg,'think')
    assert gen.output_budget_policy(cfg) == gen.OUTPUT_BUDGET_POLICY
