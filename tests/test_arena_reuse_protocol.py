# ABOUTME: Arena reuse pinning, generation-protocol parity, and torn-checkpoint regression tests.
# ABOUTME: All metadata, answers and model responses are local fixtures; no network or model calls.

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub.errors import EntryNotFoundError
from omegaconf import OmegaConf

from src.eval.capabilities.arena_hard import runner
from src.eval.capabilities.arena_hard.arena_hard_gen import OUTPUT_BUDGET_POLICY, read_partial_checkpoint
from src.infra.endpoints import vllm


def config(tmp_path):
    cfg = OmegaConf.load('configs/eval/arena_hard.yaml')
    cfg.vendor_dir = str(tmp_path / 'vendor')
    cfg.arms = [{'name': key, 'adapter': 'org/model', 'role': 'target',
                 'n_hard_prompt': 2, 'n_creative_writing': 0} for key in ('a', 'b')]
    questions = Path(cfg.vendor_dir) / 'data' / cfg.bench_name / 'question.jsonl'
    questions.parent.mkdir(parents=True)
    questions.write_text('\n'.join(json.dumps({'uid': f'q{i}', 'category': 'hard_prompt',
                                               'prompt': f'Question {i}?'}) for i in range(2)) + '\n')
    return cfg


def arm(tmp_path, cfg, key, *, context=16384, budget=6000):
    directory = tmp_path / key
    (directory / 'metadata').mkdir(parents=True)
    (directory / 'rollouts').mkdir()
    answer_file = directory / 'rollouts/answers.jsonl'
    answer_file.write_text('\n'.join(json.dumps({
        'uid': f'q{i}', 'messages': [{'role': 'user', 'content': f'Question {i}?'},
                                   {'role': 'assistant', 'content': {'answer': 'Answer'}}],
        'generation_record': {'max_tokens': budget},
    }) for i in range(2)) + '\n')
    protocol = runner.generation_protocol(cfg, 'think', key, answer_file,
        {'effective_context_window': context, 'output_budget_policy': OUTPUT_BUDGET_POLICY})
    (directory / 'metadata/generation_protocol.json').write_text(json.dumps(protocol))
    return {'model_key': key, 'mode': 'think', 'out_dir': str(directory), 'target': f'org/{key}'}


def test_reuse_resolves_once_and_uses_one_revision_for_identity_answers_and_protocol(monkeypatch, tmp_path):
    sha = 'a' * 40
    first = tmp_path / 'moving.json'
    first.write_text(json.dumps({'target': 'org/2026-09-04-qwen36-0-wrong', 'mode': 'nothink'}))
    pinned = tmp_path / 'pinned.json'
    pinned.write_text(json.dumps({'target': 'org/2026-09-04-ah-qwen36-0-da-10',
                                  'model_key': 'qwen36_0_da_10',
                                  'base_model': 'Qwen/Qwen3.6-27B', 'mode': 'think',
                                  'base_model_revision': 'b' * 40,
                                  'base_revision_from': 'training_meta'}))
    protocol = tmp_path / 'protocol.json'
    protocol.write_text(json.dumps({'source': 'pinned-protocol'}))
    answers = tmp_path / 'answers.jsonl'
    answers.write_text('{"uid":"pinned-answer"}\n')
    calls, resolutions = [], []
    def download(repo, name, **kwargs):
        calls.append((name, kwargs.get('revision')))
        if name == 'metadata/run_meta.json':
            return str(pinned if kwargs.get('revision') == sha else first)
        if kwargs.get('revision') == sha and name == 'metadata/generation_protocol.json':
            return str(protocol)
        if kwargs.get('revision') == sha and name == 'rollouts/answers.jsonl':
            return str(answers)
        raise EntryNotFoundError('fixture has no file')
    def resolve(repo, repo_type='model', **kwargs):
        resolutions.append((repo_type, kwargs.get('revision')))
        return sha
    monkeypatch.setattr(vllm, 'hf_download', download)
    monkeypatch.setattr(vllm, '_repo_sha', resolve)
    monkeypatch.setattr(runner, 'hf_download', download)
    monkeypatch.setattr(runner, 'hf_api', lambda: pytest.fail('reuse must not resolve HEAD again'))
    spec = vllm.resolve_target('org/prior-arena', revision='reviewed-branch')
    assert resolutions == [('dataset', 'reviewed-branch')]
    assert spec.revision == sha and spec.mode == 'think' and spec.model_key == 'qwen36_0_da_10'
    assert spec.base_revision == 'b' * 40 and spec.base_revision_from == 'training_meta'
    source = runner.answers_from_run(spec.answers, tmp_path / 'destination.jsonl', revision=spec.revision)
    assert source['revision'] == sha and source['generation_protocol'] == {'source': 'pinned-protocol'}
    assert (tmp_path / 'destination.jsonl').read_bytes() == answers.read_bytes()
    assert ('metadata/run_meta.json', sha) in calls
    assert ('metadata/generation_protocol.json', sha) in calls
    assert ('rollouts/answers.jsonl', sha) in calls


def test_reuse_without_protocol_is_explicitly_uncertified_and_does_not_copy_answers(monkeypatch, tmp_path):
    def missing(*args, **kwargs):
        raise EntryNotFoundError('historical artifact')
    monkeypatch.setattr(runner, 'hf_download', missing)
    with pytest.raises(ValueError, match='compatibility is uncertified'):
        runner.answers_from_run('org/old-arena', tmp_path / 'answers.jsonl', revision='a' * 40)
    assert not (tmp_path / 'answers.jsonl').exists()


def test_fresh_generation_protocols_match(tmp_path):
    cfg = config(tmp_path)
    runs = [arm(tmp_path, cfg, key) for key in ('a', 'b')]
    result = runner.validate_generation_protocols(runs, cfg)
    assert result['status'] == 'matched' and result['n_questions'] == 2


@pytest.mark.parametrize('field,value', [
    ('thinking_mode', 'nothink'), ('temperature', 0.5), ('top_p', 0.9),
    ('max_tokens', 16000), ('configured_context_window', 8192),
    ('answer_extraction', {'parser': 'old-visible-answer-v0', 'strip_think_for_judging': True}),
])
def test_changed_generation_settings_refuse_comparison(tmp_path, field, value):
    cfg = config(tmp_path)
    runs = [arm(tmp_path, cfg, key) for key in ('a', 'b')]
    path = Path(runs[1]['out_dir']) / 'metadata/generation_protocol.json'
    protocol = json.loads(path.read_text())
    protocol['settings'][field] = value
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match='incompatible generation settings'):
        runner.validate_generation_protocols(runs, cfg)


@pytest.mark.parametrize('change', ['context', 'budget', 'prompt', 'missing'])
def test_actual_context_budget_prompt_and_missing_evidence_are_checked(tmp_path, change):
    cfg = config(tmp_path)
    runs = [arm(tmp_path, cfg, 'a'), arm(tmp_path, cfg, 'b',
            context=8192 if change == 'context' else 16384, budget=3000 if change == 'budget' else 6000)]
    path = Path(runs[1]['out_dir']) / 'metadata/generation_protocol.json'
    if change == 'missing':
        path.unlink()
    if change == 'prompt':
        protocol = json.loads(path.read_text())
        protocol['questions']['q0']['identity']['prompt_sha256'] = 'old-question'
        path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError):
        runner.validate_generation_protocols(runs, cfg)


def test_reuse_runner_retains_original_protocol_and_health(monkeypatch, tmp_path):
    cfg = config(tmp_path)
    prior = arm(tmp_path, cfg, 'a')
    prior_dir = Path(prior['out_dir'])
    metrics = prior_dir / 'metadata/gen_gen_metrics.json'
    metrics.write_text('{"by_slice":{"hard_prompt":{"health":"original"}}}')
    def download(repo, name, **kwargs):
        assert kwargs['revision'] == 'a' * 40
        path = prior_dir / name
        if not path.exists():
            raise EntryNotFoundError('optional metadata absent')
        return str(path)
    monkeypatch.setattr(runner, 'hf_download', download)
    monkeypatch.setattr(runner, 'hf_api', lambda: pytest.fail('must use already-pinned source'))
    spec = SimpleNamespace(hf_path='org/prior-arena', base_model='Qwen/Qwen3.6-27B', model_key='a',
                           revision='a' * 40, base_revision=None, mode='think', answers='org/prior-arena')
    destination = tmp_path / 'reused'
    runner.run(SimpleNamespace(spec=spec), cfg, destination, reference=spec.hf_path)
    assert (destination / 'metadata/generation_protocol.json').read_text() == json.dumps(
        json.loads((prior_dir / 'metadata/generation_protocol.json').read_text()), indent=2)
    assert (destination / 'metadata/gen_gen_metrics.json').read_bytes() == metrics.read_bytes()
    source = json.loads((destination / 'metadata/sources.json').read_text())
    assert source['answers'] == {'repo': spec.answers, 'revision': spec.revision}


@pytest.mark.parametrize('tail', [b'{"uid":', b'{"uid":"\xe2\x82'])
def test_torn_last_partial_record_is_recovered_and_original_preserved(tmp_path, tail):
    path = tmp_path / 'answers.partial.jsonl'
    original = b'{"uid":"kept","request_hash":"h"}\n' + tail
    path.write_bytes(original)
    rows, evidence = read_partial_checkpoint(path)
    assert rows == [{'uid': 'kept', 'request_hash': 'h'}]
    assert evidence.read_bytes() == original
    with path.open('ab') as stream:
        stream.write(b'{"uid":"next"}\n')
    assert [json.loads(line)['uid'] for line in path.read_text().splitlines()] == ['kept', 'next']


def test_interior_checkpoint_corruption_is_not_silently_dropped(tmp_path):
    path = tmp_path / 'answers.partial.jsonl'
    original = b'{bad\n{"uid":"later"}\n'
    path.write_bytes(original)
    with pytest.raises(ValueError, match='interior corruption'):
        read_partial_checkpoint(path)
    assert path.read_bytes() == original
    assert list(tmp_path.glob('*.torn-*')) == []


def test_complete_last_record_without_newline_can_be_followed_by_new_records(tmp_path):
    path = tmp_path / 'answers.partial.jsonl'
    path.write_bytes(b'{"uid":"kept"}')
    rows, evidence = read_partial_checkpoint(path)
    assert rows == [{'uid': 'kept'}] and evidence is None
    assert path.read_bytes() == b'{"uid":"kept"}\n'


def test_changed_live_context_invalidates_answer_cache_on_the_same_endpoint(tmp_path, monkeypatch):
    from src.eval.capabilities.arena_hard import arena_hard_gen as gen
    cfg = config(tmp_path)
    cfg.output_dir = str(tmp_path / 'generated')
    cfg.generation.stream = False
    cfg.generation.parallel = 1
    cfg.arms[0].synthetic_fraction = None
    config_file = tmp_path / 'config.yaml'
    OmegaConf.save(cfg, config_file)
    state = {'context': 16384, 'calls': 0}
    def create(**kwargs):
        state['calls'] += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='Answer', reasoning='Trace'),
                                                       finish_reason='stop')])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        models=SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id='base', max_model_len=state['context'])])))
    monkeypatch.setattr(gen, 'OpenAI', lambda **kwargs: client)
    monkeypatch.setattr(gen, 'map_threaded', lambda fn, n, **kwargs: [fn(i) for i in range(n)])
    monkeypatch.setattr(gen, 'style_features', lambda text: {'token_len': 1, 'header_count': {}, 'list_count': {}, 'bold_count': {}})
    gen.main(config=str(config_file), arm='a', served_model='base')
    assert state['calls'] == 2
    gen.main(config=str(config_file), arm='a', served_model='base')
    assert state['calls'] == 2
    state['context'] = 8192
    gen.main(config=str(config_file), arm='a', served_model='base')
    assert state['calls'] == 4
    metric_files = list((tmp_path / 'generated/a').glob('*/gen_metrics.json'))
    assert json.loads(max(metric_files, key=lambda p: p.parent.name).read_text())['effective_context_window'] == 8192


def test_torn_checkpoint_evidence_survives_a_failed_recovery_attempt(tmp_path):
    from src.eval.capabilities.arena_hard.arena_hard_gen import retain_checkpoint_diagnostics
    path = tmp_path / 'cache/answers.partial.jsonl'
    path.parent.mkdir()
    originals = [b'{"uid":"one"}\n{"uid":', b'{"uid":"two"}\n{"uid":"broken']
    for original in originals:
        path.write_bytes(original)
        _, evidence = read_partial_checkpoint(path)
        assert evidence.read_bytes() == original
        # A model/network exception may stop this attempt before publication.
    rows, new_evidence = read_partial_checkpoint(path)
    assert rows == [{'uid': 'two'}] and new_evidence is None
    published = tmp_path / 'successful-generation'
    retained = retain_checkpoint_diagnostics(path, published)
    assert len(retained) == 2
    assert {p.read_bytes() for p in published.iterdir()} == set(originals)
