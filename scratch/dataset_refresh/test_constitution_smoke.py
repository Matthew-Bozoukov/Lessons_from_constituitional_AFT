# ABOUTME: Offline checks for bounded dispatch and standard synth stage wiring.
# ABOUTME: Run: uv run --no-sync pytest -q scratch/dataset_refresh/test_constitution_smoke.py
import json
import re
import pytest
from omegaconf import OmegaConf
from src.infra.endpoints.openrouter import ChatResult
from src.data.synth.ours.pipeline import run
from src.data.synth.ours.stage_runtime import call_json, Usage
from scratch.dataset_refresh.constitution_smoke import SingleAttemptClient, validate_review, calibrate


def config():
    return OmegaConf.to_container(OmegaConf.load('scratch/dataset_refresh/da-lowstakes-implicit-values.yaml'), resolve=True)


def reply(content):
    return ChatResult(content=content, prompt_tokens=10, completion_tokens=10, finish_reason='stop', cost=0.001)


def review():
    return dict(verdict='pass', stakes=1, decisive_fact_check='checked',
                competing_considerations='two', mechanism='test', domain='test', findings=[],
                target_check='checked', standalone_check='checked', bounds_check='checked')


def test_parser_cannot_buy_retry(tmp_path):
    calls = []
    def send(**kw):
        calls.append(kw)
        return reply('not JSON')
    client = SingleAttemptClient(tmp_path, config(), send=send)
    with pytest.raises(RuntimeError, match='No second attempt'):
        call_json(client, Usage(), 'anthropic/claude-sonnet-5', 'test',
                  '[record_id=test; stage=scenario]', 1, 100, 'scenario')
    assert len(calls) == 1
    again = SingleAttemptClient(tmp_path, config(), send=send)
    with pytest.raises(RuntimeError):
        again.chat(model='anthropic/claude-sonnet-5', messages=calls[0]['messages'], temperature=1, max_tokens=100)
    assert len(calls) == 1


def test_budget_stops_before_dispatch(tmp_path):
    cfg = config()
    cfg['budget_usd'] = 0.000001
    calls = []
    client = SingleAttemptClient(tmp_path, cfg, send=lambda **kw: calls.append(kw))
    with pytest.raises(RuntimeError, match='Budget stop'):
        client.chat(model='anthropic/claude-sonnet-5', messages=[{'role':'user','content':'[record_id=x; stage=scenario]'}], temperature=1, max_tokens=100)
    assert not calls


def test_calibration_wiring(tmp_path):
    cases = OmegaConf.to_container(OmegaConf.load('scratch/dataset_refresh/constitution_smoke_calibration.yaml'), resolve=True)['cases']
    def send(**kw):
        rid = re.search('record_id=calibration_([^;]+)', kw['messages'][-1]['content']).group(1)
        case = next(c for c in cases if c['id'] == rid)
        r = review()
        r['verdict'] = case['expected']
        if case.get('expected_code'):
            r['findings'] = [dict(code=case['expected_code'], quote='fixture quote', why='fixture reason')]
        return reply(json.dumps({'review': r}))
    calibrate(config(), tmp_path, SingleAttemptClient(tmp_path, config(), send=send),
              'scratch/dataset_refresh/constitution_smoke_calibration.yaml')
    assert len(json.loads((tmp_path/'calibration_results.json').read_text(encoding='utf-8'))) == 12


def test_standard_engine_all_stages(tmp_path):
    calls = []
    def send(**kw):
        calls.append(kw)
        stage = re.search('stage=([^\\]]+)', kw['messages'][-1]['content']).group(1)
        if stage == 'scenario':
            content = json.dumps({'system':'Ordinary assistant', 'user':'An ordinary request', 'applicability': {'target':'test'}})
        elif stage == 'review':
            content = json.dumps({'review': review()})
        else:
            content = '<changes>Editing notes never trained</changes><reasoning>Reasoning</reasoning><response>Advice</response>'
        return reply(content)
    cfg = config()
    out = tmp_path / 'generation'
    out.mkdir()
    manifest = run(cfg, smoke=True, resume=str(out), client=SingleAttemptClient(tmp_path, cfg, send=send))
    assert manifest['counts']['export'] == 18
    assert len(calls) == 72
    rows = [json.loads(s) for s in (out/'dataset.jsonl').read_text().splitlines()]
    assert len({r['metadata']['trait_id'] for r in rows}) == 9
    for row in rows:
        validate_review(row['metadata']['review'])
        assert row['messages'][-1]['reasoning_content'] == 'Reasoning'
        assert 'Editing notes' not in json.dumps(row)
        assert 'applicability' not in row['metadata']
