# ABOUTME: Offline MoReBench scoring and full runner tests with two response channels.
# ABOUTME: Guard negative weights, missing judgments, truncation and cache provenance.

import json
from types import SimpleNamespace as NS

import pytest
from omegaconf import OmegaConf

from src.eval.capabilities.morebench import runner as m
from src.infra.endpoints.vllm import TargetSpec

CRITERIA = [{'id': 'positive', 'weight': 3, 'text': 'Positive', 'dimension': 'identifying'},
            {'id': 'negative', 'weight': -1, 'text': 'Negative', 'dimension': 'harmless outcome'}]


@pytest.mark.parametrize('a,b,score', [(True, False, 100), (True, True, 75),
                                     (False, False, 25), (False, True, 0)])
def test_released_signed_weight_scoring(a, b, score):
    assert m.score_criteria(CRITERIA, {'positive': a, 'negative': b}) == score


@pytest.mark.parametrize('value', ['', 'not yes', 'yes or no', 'uncertain', 'yes, because...'])
def test_ambiguous_judge_is_not_a_verdict(value):
    with pytest.raises(ValueError):
        m.parse_verdict(value)


def test_missing_and_non_boolean_verdicts_are_not_failures():
    with pytest.raises(ValueError):
        m.score_criteria(CRITERIA, {'positive': True})
    with pytest.raises(ValueError):
        m.score_criteria(CRITERIA, {'positive': 'yes', 'negative': False})


def _setup(monkeypatch, finish='stop', judge_text=None):
    items = [{'id': 'i', 'prompt': 'fixture', 'criteria': CRITERIA,
              'source': 'daily_dilemmas', 'role': 'ai_advisor', 'dilemma_type': 'short_case'}]
    monkeypatch.setattr(m, 'load_items', lambda cfg: items)
    monkeypatch.setattr(m, 'provider_pin', lambda model: {'order': ['fixture'], 'allow_fallbacks': False})
    calls = {'target': 0, 'judge': 0}
    def create(**kwargs):
        calls['target'] += 1
        return NS(choices=[NS(message=NS(content='answer', reasoning_content='trace'), finish_reason=finish)])
    monkeypatch.setattr(m, 'OpenAI', lambda **kw: NS(chat=NS(completions=NS(create=create))))
    def chat(model, messages, **kwargs):
        calls['judge'] += 1
        prompt = messages[0]['content']
        # Trace earns positive credit; answer earns avoidance credit. No channel pooling.
        verdict = 'yes' if 'Response:trace' in prompt else 'no'
        return NS(content=judge_text if judge_text is not None else verdict,
                  reasoning_content='judge trace', finish_reason='stop', provider='fixture')
    monkeypatch.setattr(m, 'OpenRouterClient', lambda: NS(chat=chat))
    spec = TargetSpec('org/model', 'Qwen/Qwen3.6-27B', False, 'think', 'qwen36', None,
                      revision='fixed', base_revision='base')
    target = NS(spec=spec, model_name='base', base_url='http://fixture.invalid/v1', api_key='unused')
    return target, OmegaConf.load('configs/eval/morebench.yaml'), calls


def test_runner_scores_channels_separately_and_resumes(monkeypatch, tmp_path):
    from src.eval.layout import assert_layout
    target, cfg, calls = _setup(monkeypatch)
    summary = m.run(target, cfg, tmp_path)
    assert summary['channels']['reasoning']['regular'] == 75
    assert summary['channels']['answer']['regular'] == 25
    assert summary['channels']['reasoning']['hard'] == 15000
    assert calls == {'target': 1, 'judge': 4}
    assert m.run(target, cfg, tmp_path) == summary
    assert calls == {'target': 1, 'judge': 4}
    assert_layout(tmp_path)
    cfg.generation.temperature = 0.6
    with pytest.raises(ValueError, match='identity changed'):
        m.run(target, cfg, tmp_path)


def test_cut_off_target_is_saved_and_never_judged(monkeypatch, tmp_path):
    target, cfg, calls = _setup(monkeypatch, finish='length')
    with pytest.raises(RuntimeError, match='target channels'):
        m.run(target, cfg, tmp_path)
    assert calls == {'target': 1, 'judge': 0}
    row = json.loads((tmp_path / 'rollouts/generations.jsonl').read_text())
    assert row['answer'] == 'answer' and row['finish_reason'] == 'length'
    assert not (tmp_path / 'results/metrics.json').exists()


def test_missing_verdicts_cannot_silently_lower_score(monkeypatch, tmp_path):
    target, cfg, calls = _setup(monkeypatch, judge_text='uncertain')
    with pytest.raises(RuntimeError, match='criterion verdicts'):
        m.run(target, cfg, tmp_path)
    assert calls['judge'] == 4
    assert not (tmp_path / 'results/metrics.json').exists()


@pytest.mark.parametrize("finish,empty", [("length", False), ("stop", True)])
def test_resume_preserves_incomplete_target_outcomes(monkeypatch, tmp_path, finish, empty):
    target, cfg, calls = _setup(monkeypatch, finish=finish)
    if empty:
        monkeypatch.setattr(m, 'resolve_trace', lambda *args: ('trace', ''))
    with pytest.raises(RuntimeError, match='target channels'):
        m.run(target, cfg, tmp_path)
    assert calls['target'] == 1
    before = (tmp_path / 'rollouts/generations.jsonl').read_bytes()
    target, cfg, resumed = _setup(monkeypatch)
    with pytest.raises(RuntimeError, match='target channels'):
        m.run(target, cfg, tmp_path)
    assert resumed == {'target': 0, 'judge': 0}
    assert (tmp_path / 'rollouts/generations.jsonl').read_bytes() == before


def test_resume_retries_transport_failure_without_regenerating_completed_items(monkeypatch, tmp_path):
    target, cfg, calls = _setup(monkeypatch)
    def unavailable(**kwargs):
        raise TimeoutError('fixture transport failure')
    monkeypatch.setattr(m, 'OpenAI', lambda **kw: NS(chat=NS(completions=NS(create=unavailable))))
    with pytest.raises(RuntimeError, match='target channels'):
        m.run(target, cfg, tmp_path)
    target, cfg, resumed = _setup(monkeypatch)
    summary = m.run(target, cfg, tmp_path)
    assert summary['n_items'] == 1
    assert resumed == {'target': 1, 'judge': 4}
    saved = [json.loads(line) for line in (tmp_path / 'rollouts/generations.jsonl').read_text().splitlines()]
    assert len(saved) == 2 and saved[0]['error'] == 'TimeoutError' and saved[1]['valid']


def test_legacy_cache_does_not_select_successful_target_reroll(tmp_path):
    path = tmp_path / 'generations.jsonl'
    first = {'id': 'i', 'valid': False, 'finish_reason': 'length', 'answer': ''}
    reroll = {'id': 'i', 'valid': True, 'finish_reason': 'stop', 'answer': 'answer'}
    path.write_text(json.dumps(first) + '\n' + json.dumps(reroll) + '\n', encoding='utf-8')
    def unexpected(job):
        pytest.fail('A returned target outcome must never be regenerated on resume')
    assert m._cached_jobs(path, [{'id': 'i'}], unexpected, 1, preserve_outcomes=True) == [first]
