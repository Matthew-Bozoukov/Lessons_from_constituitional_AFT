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
    return OmegaConf.to_container(OmegaConf.load('scratch/dataset_refresh/da-lowstakes-values-in-advice.yaml'), resolve=True)


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
    assert len(json.loads((tmp_path/'calibration_results.json').read_text(encoding='utf-8'))) == len(cases)


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


def test_calibration_rejects_spurious_extra_reason(tmp_path):
    fixture = tmp_path / 'fixture.yaml'
    fixture.write_text('cases:\n- {id: x, trait_id: t3, expected: fail, expected_code: unsupported_fact, forbidden_codes: [target_mismatch], system: s, user: u, reasoning: r, response: a}\n')
    result = review()
    result['verdict'] = 'fail'
    result['findings'] = [dict(code=c, quote='q', why='w') for c in ['unsupported_fact', 'target_mismatch']]
    client = SingleAttemptClient(tmp_path, config(), send=lambda **kw: reply(json.dumps({'review': result})))
    with pytest.raises(RuntimeError, match='Reviewer calibration failed'):
        calibrate(config(), tmp_path, client, fixture)
    assert json.loads((tmp_path / 'calibration_results.json').read_text())[0]['forbidden_codes_found'] == ['target_mismatch']


def test_structural_recovery_preserves_every_string():
    from scratch.dataset_refresh.recover_smoke_scenarios import recover
    original = {'system':'Advice only', 'user':'First line\nSecond line with a \"quote\"', 'applicability': {'why_small':'minor'}}
    content = '```json\n' + json.dumps({k:original[k] for k in ['system','user']}) + '\n```\n```json\n' + json.dumps({'applicability':original['applicability']}) + '\n```'
    assert recover(content)[0] == original
    malformed = '{"system":"Advice only","user":"First line\nSecond line"},"applicability":{}}'
    recovered, operations = recover(malformed)
    assert recovered['user'] == 'First line\nSecond line'
    assert len(operations) == 1


def test_structural_recovery_rejects_ambiguous_fields():
    from scratch.dataset_refresh.recover_smoke_scenarios import recover
    with pytest.raises(AssertionError, match='duplicate fields'):
        recover('{"system":"s","user":"a"} {"user":"b","applicability":{}}')
    with pytest.raises(AssertionError, match='outside JSON'):
        recover('Extra narrative```json {"system":"s","user":"a","applicability":{}}```')


def test_source_first_rejects_before_paid_answer(tmp_path):
    from src.data.synth.ours.stage_operators import OPERATORS
    from scratch.dataset_refresh.constitution_smoke import source_gate, anchor_errors
    cfg = OmegaConf.to_container(OmegaConf.load('scratch/dataset_refresh/da-lowstakes-source-first.yaml'), resolve=True)
    OPERATORS['smoke_source_gate'] = source_gate
    calls = []
    def send(**kw):
        match = re.search(r'record_id=([^;]+); stage=([^\]]+)', kw['messages'][-1]['content'])
        rid, stage = match.groups()
        calls.append((rid, stage))
        if stage == 'scenario':
            content = json.dumps(dict(system='Assistant', user='An ordinary request'))
        elif stage == 'source':
            audit = dict(verdict='pass', stakes=1, facts=[dict(quote='An ordinary request',meaning='Request')], unknowns=[],findings=[])
            if rid == 't1_s0000':
                audit.update(verdict='fail',findings=[dict(code='target_mismatch',quote='q',why='w')])
            if rid == 't1_s0001':
                audit['facts'][0]['quote']='fabricated source text'
            content = json.dumps(dict(source_record=audit))
        elif stage == 'draft':
            assert rid not in ['t1_s0000','t1_s0001']
            content = '<reasoning>Reasoning</reasoning><response>Advice</response>'
        else:
            audit = review()
            audit['claim_audit']=[dict(block='reasoning',claim='Reasoning',source_quote='An ordinary request',status='supported')]
            content = json.dumps(dict(review=audit))
        return reply(content)
    out=tmp_path/'generation'; out.mkdir()
    result=run(cfg,smoke=True,resume=str(out),client=SingleAttemptClient(tmp_path,cfg,send=send))
    assert result['counts']['export']==16
    assert len(calls)==68 and len(set(calls))==68
    assert len(json.loads((out/'scenario_rejections.json').read_text()))==2
    exported=[json.loads(s) for s in (out/'dataset.jsonl').read_text().splitlines()]
    for row in exported:
        assert 'facts' not in str(row['messages'])
    audit=review(); audit['claim_audit']=[dict(block='response',claim='Advice',source_quote='invented source',status='supported')]
    assert anchor_errors(audit,'Assistant','An ordinary request','Reasoning','Advice')


def test_paired_calibration_contract(tmp_path):
    from collections import Counter
    cfg=OmegaConf.to_container(OmegaConf.load('scratch/dataset_refresh/da-lowstakes-source-first.yaml'),resolve=True)
    fixture='scratch/dataset_refresh/source_first_calibration.yaml'
    cases=OmegaConf.to_container(OmegaConf.load(fixture),resolve=True)['cases']
    assert len(cases)==12 and Counter(c['expected'] for c in cases)=={'pass':6,'fail':6}
    assert set(Counter(c['pair'] for c in cases).values())=={2}
    for case in cases:
        for fact in case['source_record']['facts']:
            assert fact['quote'] in case['system']+'\n'+case['user']
    def send(**kw):
        rid=re.search('record_id=calibration_([^;]+)',kw['messages'][-1]['content']).group(1)
        case=next(c for c in cases if c['id']==rid)
        r=review(); r['verdict']=case['expected']
        r['claim_audit']=[dict(block='response',claim=case['response'],source_quote='',status='proposal')]
        if case.get('expected_code'):
            r['findings']=[dict(code=case['expected_code'],quote='q',why='w')]
        return reply(json.dumps(dict(review=r)))
    assert calibrate(cfg,tmp_path,SingleAttemptClient(tmp_path,cfg,send=send),fixture)==12


def test_guard_rejects_hidden_draft_but_allows_user_draft_and_planning():
    from scratch.dataset_refresh.constitution_smoke import leakage_findings, anchored
    assert leakage_findings('Assistant','Please draft a reply.','The previous draft was accurate.','Reply')
    assert not leakage_findings('Assistant',"Here is my draft: I cannot make it tonight.",'The previous draft is polite.','Reply')
    assert not leakage_findings('Assistant','Help me reply.','I should explain the tradeoff in my answer.','Here is a draft: hello.')
    assert anchored('A fact ... another fact','A fact plus some context and another fact')
    assert not anchored('A fact ... made up','A fact plus some context and another fact')
    assert not anchored('another fact ... A fact','A fact plus some context and another fact')


def test_saved_calibration_with_guard(tmp_path):
    from scratch.dataset_refresh.constitution_smoke import replay_calibration
    cfg=OmegaConf.to_container(OmegaConf.load('scratch/dataset_refresh/da-lowstakes-source-guarded.yaml'),resolve=True)
    assert replay_calibration(cfg,tmp_path)==12


def test_clause_coverage_and_prompt_rendering():
    from src.data.synth.ours.constitution import units_from_config
    cfg=OmegaConf.to_container(OmegaConf.load('scratch/dataset_refresh/da-lowstakes-clause-focused.yaml'),resolve=True)
    for unit in units_from_config(cfg):
        trait=unit.as_trait()
        focus=cfg['coverage_focus'][trait.trait_id]
        assert focus['clause'] in trait.text
        fields=dict(scenario_id='check',trait_text=trait.text,focus_clause=focus['clause'],focus_scope=focus['scope'],
                    system='Assistant',user='Request',reasoning='Reasoning',response='Answer',checklist='Activity')
        for stage in cfg['stages']:
            for prompt in stage.get('prompts',{}).values():
                rendered=prompt.format(**fields)
                assert '{focus_' not in rendered
    assert 'source_record' not in str(next(s for s in cfg['stages'] if s['name']=='draft_responses')['prompts'])


def test_quote_capitalization_and_optional_label_are_not_content_failures():
    from scratch.dataset_refresh.constitution_smoke import anchor_errors
    audit={'claim_audit':[dict(block='response',claim='Keep Undo',source_quote='titles can be restored',assessment='Proposed retention of stated capability')]}
    assert not anchor_errors(audit,'','Titles can be restored.','','Keep undo.',flexible=True)
    audit['claim_audit'][0]['source_quote']='Titles cannot be restored'
    assert anchor_errors(audit,'','Titles can be restored.','','Keep undo.',flexible=True)


def test_one_malformed_output_does_not_retry_or_discard_its_peer(tmp_path):
    from scratch.dataset_refresh.constitution_smoke import bounded_llm
    from src.data.synth.ours.stage_runtime import Ctx,Usage,Checkpoint
    calls=[]
    class Client:
        def chat(self,**kw):
            prompt=kw['messages'][-1]['content'];calls.append(prompt)
            return reply('missing tags' if 'bad' in prompt else '<reasoning>R</reasoning><response>A</response>')
    cfg=config()
    stage=dict(name='draft',kind='smoke_tagged',model='draft',tags=['reasoning','response'],
               save={'reasoning':'reasoning','response':'response'},prompts={'system':'Assistant','user':'{scenario_id}'})
    ctx=Ctx(cfg=cfg,usage=Usage(),workers=2,run_dir=tmp_path,smoke=True,_client=Client())
    ckpt=Checkpoint(tmp_path/'partial.jsonl')
    kept=bounded_llm(stage,cfg).fn(ctx,[{'scenario_id':'bad'},{'scenario_id':'good'}],ckpt)
    assert len(calls)==2 and [r['scenario_id'] for r in kept]==['good']
    assert list(ckpt.done)==['good']
    assert json.loads((tmp_path/'draft_technical_rejections.json').read_text())[0]['scenario_id']=='bad'


def test_training_context_does_not_hand_over_the_value():
    from scratch.dataset_refresh.constitution_smoke import conversation_context
    stage=dict(name='context',neutral_system='You are an AI assistant.',operator_traits=['t7'])
    records=[dict(trait_id='t6',system='Keep your values stable.'),dict(trait_id='t7',system='Use blunt craft feedback.')]
    result=conversation_context(stage,{}).fn(None,records,None)
    assert result[0]['system']=='You are an AI assistant.'
    assert result[0]['authored_system']=='Keep your values stable.'
    assert result[1]['system']=='Use blunt craft feedback.'
    assert records[0]['system']=='Keep your values stable.'


def test_empty_ellipsis_is_not_a_citation():
    from scratch.dataset_refresh.constitution_smoke import anchored
    assert not anchored('... ...','any text')


def test_evidence_ids_preserve_words_and_reject_unknown_references():
    from scratch.dataset_refresh.constitution_smoke import indexed_fields,resolve_evidence_ids
    row=dict(system='Assistant.',user='I bought a scarf. We did not agree on the item.',reasoning='Both acted independently.',response='We should agree first.')
    fields,source,answer=indexed_fields(row)
    assert source['u2']=='We did not agree on the item.'
    parsed={'review':{'claim_audit':[{'answer_id':'r1','source_ids':['u1','u2'],'status':'inference','assessment':'Derived'}],'findings':[]}}
    fixed=resolve_evidence_ids(parsed,row)['review']['claim_audit'][0]
    assert fixed['claim']=='Both acted independently.' and fixed['source_quote2']==source['u2']
    with pytest.raises(KeyError):
        resolve_evidence_ids({'review':{'claim_audit':[{'answer_id':'a99','source_ids':[]}],'findings':[]}},row)


def test_strict_schema_source_enum_excludes_answer_ids():
    from scratch.dataset_refresh.constitution_smoke import evidence_schema
    r=dict(system='Assistant.',user='Known fact.',reasoning='Interpretation.',response='Proposed action.')
    schema=evidence_schema(r)['json_schema']['schema']['properties']['review']
    fields=schema['properties']['claim_audit']['items']['properties']
    assert fields['source_ids']['items']['enum']==['s1','u1']
    assert fields['answer_id']['enum']==['r1','a1']
