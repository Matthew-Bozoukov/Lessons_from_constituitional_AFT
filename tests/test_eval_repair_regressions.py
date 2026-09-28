# ABOUTME: Offline endpoint regressions for the September 2026 evaluation repairs.
# ABOUTME: Exercise real runner boundaries with synthetic completions, never paid APIs.

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from omegaconf import OmegaConf


@pytest.mark.parametrize("served", ["base", "openai/gpt-oss-120b"])
def test_arena_publishes_by_arm_without_mutating_harness(tmp_path, monkeypatch, served):
    from src.eval.capabilities.arena_hard import arena_hard_gen as gen, runner
    from src.eval.layout import assert_layout
    from src.infra.endpoints.vllm import TargetSpec

    cfg = OmegaConf.load("configs/eval/arena_hard.yaml")
    cfg.vendor_dir = str(tmp_path / "vendor")
    cfg.arm_defaults = {"n_hard_prompt": 1, "n_creative_writing": 0}
    cfg.arms = []
    cfg.generation.stream = False
    bench = Path(cfg.vendor_dir) / "data" / cfg.bench_name
    bench.mkdir(parents=True)
    (bench / "question.jsonl").write_text(json.dumps(
        {"uid": "fixture", "category": "hard_prompt", "prompt": "Say hello"}) + "\n")
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return NS(choices=[NS(message=NS(content="Hello", reasoning="A trace"), finish_reason="stop")])
    client = NS(chat=NS(completions=NS(create=create)),
                models=NS(list=lambda: NS(data=[NS(id=served, max_model_len=16384)])))
    monkeypatch.setattr(gen, "OpenAI", lambda **kw: client)
    monkeypatch.setattr(gen.tiktoken, "encoding_for_model", lambda *a: NS(encode=lambda s, **kw: list(s)))
    monkeypatch.setattr(gen, "style_features", lambda s: {
        "token_len": 1, "header_count": {}, "list_count": {}, "bold_count": {}})
    target = NS(spec=TargetSpec("org/model", "org/base", False, "think", "fixture_arm", None,
                                revision="first"), model_name=served,
                base_url="http://unused.invalid/v1", api_key="EMPTY")
    out = tmp_path / "run"
    runner.run(target, cfg, out, reference="org/reference")
    assert_layout(out)
    answer = json.loads((out / "rollouts/answers.jsonl").read_text())
    assert answer["model"] == served
    assert answer["generation_record"]["think"] == "A trace"
    assert calls[0]["stream"] is False
    assert not (bench / "model_answer").exists()

    # Standalone resume: same request hits; changed temperature/prompt/weights misses.
    cfg.arms = runner.register([], "fixture_arm", "org/model", "target", cfg)
    cfg.output_dir = str(tmp_path / "standalone")
    cfg.target_identity = {"revision": "first"}
    config = tmp_path / "config.yaml"
    def generate():
        OmegaConf.save(cfg, config)
        gen.main(config=str(config), arm="fixture_arm", served_model=served)
    generate()
    n = len(calls)
    generate()
    assert len(calls) == n
    cfg.generation.temperature = 0.93
    generate()
    assert len(calls) == n + 1
    cfg.target_identity.revision = "second"
    generate()
    assert len(calls) == n + 2
    (bench / "question.jsonl").write_text(json.dumps(
        {"uid": "fixture", "category": "hard_prompt", "prompt": "Say goodbye"}) + "\n")
    generate()
    assert len(calls) == n + 3


@pytest.mark.parametrize("content,finish,empty,failed", [
    (None, "stop", 1, False), ("", "stop", 1, False),
    ("", "length", 0, True), ("partial", "length", 0, True),
])
def test_mask_distinguishes_finished_empty_from_truncation(content, finish, empty, failed):
    from src.eval.misalignment.mask.runner import _HARNESS
    spec = importlib.util.spec_from_file_location("mask_health_fixture", _HARNESS / "generate_responses.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    async def create(**kwargs):
        return NS(choices=[NS(message=NS(content=content, reasoning="trace"), finish_reason=finish)])
    result = asyncio.run(mod.generate_responses_async(NS(chat=NS(completions=NS(create=create))),
        "fixture", {"lying": [{"role": "user", "content": "fixture"}]}, 100, 1, asyncio.Semaphore(1)))
    assert result['__empty_content__'].get('lying', 0) == empty
    assert result['__finish_reason__']['lying'] == [finish]
    assert result['__raw_content__']['lying'] == [content]
    assert result['lying'][0].startswith('[ERROR') == failed


def test_mmlu_loads_requested_revision(monkeypatch):
    import datasets
    from src.eval.capabilities.mmlu.mmlu import load_split
    calls = []
    def load(*args, **kwargs):
        calls.append(kwargs)
        return []
    monkeypatch.setattr(datasets, 'load_dataset', load)
    load_split('test', revision='pinned')
    load_split('dev', revision='pinned')
    assert [c['revision'] for c in calls] == ['pinned', 'pinned']


def test_agentic_misalignment_requires_every_finished_classified_cell(tmp_path):
    from src.eval.misalignment.agentic_misalignment.runner import validate_results
    prompts = tmp_path / 'prompts/condition'
    prompts.mkdir(parents=True)
    (prompts / 'system_prompt.txt').write_text('fixture')
    path = tmp_path / 'models/model/condition/sample_001/response.json'
    path.parent.mkdir(parents=True)
    row = {'raw_response': 'answer', 'metadata': {'model': 'vllm/model', 'condition': 'condition',
           'sample_id': 1, 'api_metadata': {'stop_reason': 'stop'}},
           'classification': {'harmful_behavior': False}}
    path.write_text(json.dumps(row))
    validate_results(tmp_path, 'vllm/model', 1, classified=True)
    with pytest.raises(ValueError, match='coverage'):
        validate_results(tmp_path, 'vllm/model', 2)
    row['classification']['harmful_behavior'] = 'false'
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='verdict'):
        validate_results(tmp_path, 'vllm/model', 1, classified=True)
    row['metadata']['api_metadata']['stop_reason'] = 'length'
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='Incomplete'):
        validate_results(tmp_path, 'vllm/model', 1)


def test_agentic_client_preserves_out_of_band_trace(monkeypatch):
    import sys
    from src.eval.misalignment.agentic_misalignment.runner import _HARNESS
    spec = importlib.util.spec_from_file_location('agentic_client_fixture', _HARNESS / 'api_client/model_client.py')
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)
    spec.loader.exec_module(mod)
    calls = []
    async def create(**kwargs):
        calls.append(kwargs)
        return NS(choices=[NS(message=NS(content=None, reasoning='trace'), finish_reason='length')], usage=None)
    client = object.__new__(mod.ModelClient)
    client.vllm_client = NS(chat=NS(completions=NS(create=create)))
    reply = asyncio.run(client._call_vllm('vllm/openai/gpt-oss-120b', [], 10000, 1.0))
    assert reply.completion == '' and reply.reasoning_content == 'trace'
    assert reply.finish_reason == 'length'
    assert calls[0]['model'] == 'openai/gpt-oss-120b'
    assert 'extra_body' not in calls[0]


def test_psychosis_checkpoints_a_cut_off_turn_before_stopping():
    from src.eval.misalignment.psychosis.conversation import run_conversation
    checkpoints = []
    replies = iter([('finished', None, 'stop'), ('partial', 'trace', 'length')])
    with pytest.raises(ValueError, match='turn 2'):
        run_conversation('Fixture', 'instructions', 3,
                         lambda messages: '<message>hello</message>', lambda messages: next(replies),
                         on_turn=lambda turns: checkpoints.append(list(turns)))
    assert [len(c) for c in checkpoints] == [1, 2]
    assert checkpoints[-1][-1].assistant == 'partial'


def test_odcv_rewrites_only_loopback_host():
    from src.eval.misalignment.odcv.runner import _bridge_url
    assert _bridge_url('http://127.0.0.1:1234/v1', 'bridge') == 'http://bridge:1234/v1'
    assert _bridge_url('http://[::1]:1234/v1', 'bridge') == 'http://bridge:1234/v1'
    url = 'https://localhost.example.com/v1/127.0.0.1?name=localhost'
    assert _bridge_url(url, 'bridge') == url


def test_arena_rejects_missing_ordering_and_changed_reused_prompt(tmp_path):
    from src.eval.capabilities.arena_hard import arena_hard_judge as judge
    questions = [{'uid': 'q1', 'category': 'hard_prompt', 'prompt': 'original question'}]
    record = {'uid': 'q1', 'category': 'hard_prompt', 'model': 'arm',
              'games': [{'score': 'A=B'}, {'score': 'A=B'}]}
    assert judge._complete_judgments([record], questions) == [record]
    for games in ([{'score': 'A=B'}], [{'score': 'A=B'}, {'score': None}]):
        with pytest.raises(ValueError, match='paired judgments'):
            judge._complete_judgments([{**record, 'games': games}], questions)
    with pytest.raises(ValueError, match='paired judgments'):
        judge._complete_judgments([], questions)
    cfg = OmegaConf.create({'vendor_dir': str(tmp_path), 'bench_name': 'fixture'})
    answer = tmp_path / 'data/fixture/model_answer/arm.jsonl'
    answer.parent.mkdir(parents=True)
    row = {'uid': 'q1', 'messages': [{'role': 'user', 'content': 'different question'},
                                   {'role': 'assistant', 'content': {'answer': 'reply'}}]}
    answer.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='different prompt'):
        judge._validate_answers(cfg, 'arm', questions)
    row['messages'][0]['content'] = 'original question'
    answer.write_text(json.dumps(row))
    judge._validate_answers(cfg, 'arm', questions)


def test_arena_removes_generated_credentials_even_when_judging_fails(tmp_path, monkeypatch):
    from src.eval.capabilities.arena_hard import arena_hard_judge as judge
    endpoint = tmp_path / 'endpoint.yaml'
    endpoint.write_text('fake credential')
    def fail(*args, **kwargs):
        raise RuntimeError('fixture failure')
    monkeypatch.setattr(judge.subprocess, 'run', fail)
    with pytest.raises(RuntimeError, match='fixture failure'):
        judge._run_vendor(tmp_path, tmp_path / 'setting.yaml', endpoint)
    assert not endpoint.exists()


def test_arena_validation_failure_preserves_raw_verdicts_before_stopping(tmp_path, monkeypatch):
    from src.eval.capabilities.arena_hard import pool
    runs = []
    for key, reference in [('baseline', True), ('candidate', False)]:
        path = tmp_path / key
        (path / 'metadata').mkdir(parents=True)
        (path / 'rollouts').mkdir()
        (path / 'metadata/sources.json').write_text(json.dumps({'reference_arm': reference}))
        (path / 'rollouts/answers.jsonl').write_text('{}\n')
        runs.append({'model_key': key, 'target': f'org/{key}', 'mode': 'think', 'out_dir': str(path)})
    cfg = OmegaConf.load('configs/eval/arena_hard.yaml')
    cfg.vendor_dir = str(tmp_path / 'vendor')
    Path(cfg.vendor_dir).mkdir()
    def validate(config):
        assert config.judge_validation.comparison_arm == 'candidate'
        path = Path(config.vendor_dir) / 'data' / config.bench_name / 'model_judgment/reference/candidate.jsonl'
        path.parent.mkdir(parents=True)
        path.write_text('{"fixture": "raw verdict"}\n')
        return {'passes': False, 'n_compared': config.judge_validation.n_questions}
    monkeypatch.setattr(pool.arena_hard_judge, 'validate_judge', validate)
    with pytest.raises(ValueError, match='validation failed'):
        pool.pool(runs, cfg, tmp_path / 'output')
    assert (tmp_path / 'output/rollouts/judge_validation/reference/candidate.jsonl').exists()
    assert not (tmp_path / 'output/results/leaderboard.json').exists()
