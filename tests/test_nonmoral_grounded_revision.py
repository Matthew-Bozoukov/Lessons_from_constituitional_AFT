# ABOUTME: Exercises grounded revision through the real SynthDoc stages without paid calls.
# ABOUTME: Checks valid review labels, immutable inputs, persistent retry limits and dispatch deadlines.
import json
import time
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from scratch.nonmoral import broader_data as data
from src.data.synth.ours.stage_runtime import lint_problems


def config():
    return OmegaConf.to_container(OmegaConf.load('configs/data/synth/nonmoral-grounded-revision.yaml'), resolve=True)


def test_review_contract_rejects_unknown_labels():
    spec = config()['stages'][-1]['lint']
    assert not lint_problems(dict(decision='accept', original_valid='yes', improvement='concrete'), spec)
    assert len(lint_problems(dict(decision='maybe', original_valid='true', improvement='longer'), spec)) == 3


def test_real_engine_retains_source_and_review(tmp_path):
    cfg = config()
    source = tmp_path/'source'
    source.mkdir()
    original = dict(scenario_id='test_1', domain='craft', user='Pick a blue or green cover.',
                    original_system='', original_reasoning='Green hides marks.', original_response='Green.')
    (source/'inputs.jsonl').write_text(json.dumps(original)+'\n')
    cfg.update(source=dict(local_dir=str(source), snapshot='inputs.jsonl'), total_scenarios=1,
               output_dir=str(tmp_path/'output'))
    replies = iter(['<reasoning>Both fit; green matches the binding.</reasoning><response>Green.</response><changes>Disclosed criterion.</changes>',
                    '<decision>Accept.</decision><issues>none</issues><original_valid>yes</original_valid><improvement>tie</improvement><checks>Two feasible colours.</checks>'])
    calls = []
    class Fake:
        def chat(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(content=next(replies), finish_reason='stop', prompt_tokens=100,
                                   completion_tokens=50, provider='offline')
    run_dir = tmp_path/'run'
    run_dir.mkdir()
    data.run(cfg, resume=str(run_dir), client=Fake())
    result = data.read_rows(run_dir/'dataset.jsonl')
    assert len(result) == 1 and len(calls) == 2
    assert all(result[0][key] == value for key, value in original.items())
    assert result[0]['review_decision'] == 'accept'
    assert result[0]['original_valid'] == 'yes'
    assert calls[0]['model'] == cfg['models']['author']['model']
    assert calls[1]['model'] == cfg['models']['reviewer']['model']


def test_attempt_limit_survives_restart_and_deadline_blocks_dispatch(tmp_path, monkeypatch):
    paid = []
    monkeypatch.setattr(data.CappedClient, 'chat', lambda self, *args, **kw: paid.append((args, kw)))
    model = config()['models']['author']['model']
    def client(deadline):
        return data.RevisionClient(None, tmp_path/'spend.json', 5, {model}, deadline=deadline)
    request = dict(model=model, messages=[dict(role='user', content='Record ID: case_1\nRequest')])
    first = client(time.time()+60)
    first.chat(**request)
    second = client(time.time()+60)
    second.chat(**request)
    with pytest.raises(RuntimeError, match='Two-attempt'):
        second.chat(**request)
    request['messages'][0]['content'] = 'Record ID: case_2\nRequest'
    with pytest.raises(RuntimeError, match='deadline'):
        client(time.time()-1).chat(**request)
    (tmp_path/'STOP_DISPATCH').touch()
    with pytest.raises(RuntimeError, match='stop marker'):
        second.chat(**request)
    assert len(paid) == 2
