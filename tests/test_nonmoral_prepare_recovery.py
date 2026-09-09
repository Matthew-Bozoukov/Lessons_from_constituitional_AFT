# ABOUTME: Offline rejection and lineage tests for the one-batch local recovery materializer.
# ABOUTME: Exercises real shared source/author gates and conditional model-review execution without provider calls.
import copy
import json

import pytest
from omegaconf import OmegaConf

from scratch.nonmoral import prepare_recovery as recovery
from src.data.synth.stage_operators import op_llm_tagged
from src.data.synth.stage_runtime import Ctx, Usage
from src.infra.endpoints.openrouter import ChatResult


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def fixture(tmp_path, schema='records', changed=True, already_accepted=False):
    root=tmp_path/'root'; parent=root/'production/batch01'; directory=tmp_path/'recovery'
    source={'scenario_id':'broader_batch01_001','domain':'Teaching','user':'Choose and explain.'}
    original={**source,'reasoning':'The total is 18. I choose A.','response':'A.',
              'quality_decision':'accept','quality_issues':'Original exact evidence','choice_summary':'A selected'}
    final=copy.deepcopy(original)
    if changed: final['reasoning']=final['reasoning'].replace('18','19')
    for path,data in [(parent/'sources/dataset.jsonl',[source]),(parent/'answers/dataset.jsonl',[original]),
                      (directory/'recovery_candidates.jsonl',[final])]:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(r)+'\n' for r in data),encoding='utf-8')
    stage={'name':'quality_review','kind':'llm_tagged','model':'reviewer','checkpoint':'scenario_id',
           'tags':['decision','issues','summary'],'save':{'quality_decision':'decision','quality_issues':'issues','choice_summary':'summary'},
           'prompts':{'system':'Review','user':'{user}\n{reasoning}\n{response}'}}
    cfg={'models':{'reviewer':{'model':recovery.SONNET,'max_tokens':200}},
         'production':{'defer_model_review':False},'stages':[stage],'workers':1,'max_fail_pct':100}
    config=tmp_path/'config.yaml'; OmegaConf.save(OmegaConf.create(cfg),config)
    for phase in ['sources','answers']:
        write(parent/phase/'status.json',{'run_dir':str(parent/phase),'config':cfg,
              'config_sha256':recovery.file_sha256(config),'dataset_sha256':recovery.file_sha256(parent/phase/'dataset.jsonl')})
    write(parent/'source_review.json',{'reviewer':'source reader','source_sha256':recovery.file_sha256(parent/'sources/dataset.jsonl'),
         'dispositions':{source['scenario_id']:{'decision':'accept','reason':'Complete valid source'}}})
    oldreview=parent/'answer_review.json'
    write(oldreview,{'dataset_sha256':recovery.file_sha256(parent/'answers/dataset.jsonl'),
          'dispositions':{source['scenario_id']:{'decision':'accept' if already_accepted else 'reject','reason':'Original local decision'}}})
    edits=[{'field':'reasoning','old':'18','new':'19'}] if changed else []
    item={'decision':'corrected' if changed else 'accept','reason':'Verified literal arithmetic correction',
          'original_conversation_sha256':recovery.conversation_sha256(original),
          'revised_conversation_sha256':recovery.conversation_sha256(final)}
    snapshot=str(parent/'answers/dataset.jsonl'); sha=recovery.file_sha256(snapshot)
    review={'status':'complete','reviewer':'local reviewer'}
    if schema=='records':
        item['replacements']=edits; review.update(records={source['scenario_id']:item},original_snapshot_sha256=sha)
        write(directory/'author_review.json',{'snapshot_path':snapshot})
    elif schema=='inputs':
        item['replacements']=edits; review.update(dispositions={source['scenario_id']:item},inputs={'batch01':{
            'dataset':snapshot,'dataset_sha256':sha,'review':str(oldreview),'review_sha256':recovery.file_sha256(oldreview)}})
    else:
        item['exact_replacements']=edits; review.update(dispositions={source['scenario_id']:item},original_snapshots={snapshot:sha,str(oldreview):recovery.file_sha256(oldreview)})
    write(directory/'recovery_review.json',review)
    return root,directory,config,original,final,item


@pytest.mark.parametrize('schema',['records','inputs','original_snapshots'])
def test_actual_formats_materialize_through_existing_gates(tmp_path,schema):
    root,directory,config,*_=fixture(tmp_path,schema)
    plan=recovery.prepare(config,[directory],'batch08',root)
    assert not (root/'production/batch08').exists()
    assert plan['authors'][0]['needs_model_review'] is True
    assert 'quality_decision' not in plan['authors'][0]
    target=recovery.materialize(plan)
    assert len(recovery.author_review_inputs(target/'author_review.json',target/'answers/dataset.jsonl')[0])==1
    assert len(recovery.source_review_inputs(target/'source_review.json',target/'sources/dataset.jsonl')[0])==1
    assert recovery.load(target/'sources/status.json')['origin']=='local_derivative_not_model_generation'
    assert recovery.load(target/'sources/status.json')['config_sha256']==recovery.file_sha256(target/'config.yaml')
    assert recovery.load(target/'audit/lineage.json')['summary']['paid_calls']==0


@pytest.mark.parametrize('fault',['prompt','original_hash','revised_hash','nonliteral','unlogged_change'])
def test_replay_rejects_invalid_recovery(tmp_path,fault):
    _,_,_,original,final,item=fixture(tmp_path)
    if fault=='prompt': final['user']='Repaired missing prompt data'
    elif fault=='original_hash': item['original_conversation_sha256']='stale'
    elif fault=='revised_hash': item['revised_conversation_sha256']='stale'
    elif fault=='nonliteral': item['replacements'][0]['old']='not present'
    else: final['response']='Unlogged semantic rewrite'
    with pytest.raises(ValueError): recovery.replay(original,final,item,'replacements')


def test_previously_accepted_cannot_be_added_twice(tmp_path):
    root,directory,config,*_=fixture(tmp_path,already_accepted=True)
    plan=recovery.prepare(config,[directory],'batch08',root)
    assert plan['authors']==[]
    assert len(plan['excluded'])==1
    with pytest.raises(ValueError,match='no new candidates'): recovery.materialize(plan)
    assert not (root/'production/batch08').exists()


def test_logged_replace_all_normalizes_actual_occurrences(tmp_path):
    _,_,_,original,final,item=fixture(tmp_path)
    original['reasoning']='18 then 18'; final['reasoning']='19 then 19'
    item['original_conversation_sha256']=recovery.conversation_sha256(original)
    item['revised_conversation_sha256']=recovery.conversation_sha256(final)
    result,edits=recovery.replay(original,final,item,'replacements')
    assert result['reasoning']=='19 then 19' and edits[0]['occurrences']==2
    final['reasoning']='19 then 18'
    item['revised_conversation_sha256']=recovery.conversation_sha256(final)
    with pytest.raises(ValueError,match='literal replacement replay'):
        recovery.replay(original,final,item,'replacements')


def test_unapproved_source_and_stale_snapshot_are_rejected(tmp_path):
    root,directory,config,*_=fixture(tmp_path)
    p=root/'production/batch01/source_review.json'; review=recovery.load(p)
    review['dispositions']['broader_batch01_001']['decision']='reject'; write(p,review)
    with pytest.raises(ValueError,match='unapproved'): recovery.prepare(config,[directory],'batch08',root)
    p=root/'production/batch01/answers/dataset.jsonl'; p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError,match='Stale'): recovery.prepare(config,[directory],'batch08',root)


def test_unchanged_review_passes_shared_conditional_stage_without_call(tmp_path):
    root,directory,config,original,*_=fixture(tmp_path,changed=False)
    plan=recovery.prepare(config,[directory],'batch08',root)
    unchanged=plan['authors'][0]
    assert unchanged['needs_model_review'] is False
    assert unchanged['quality_issues']==original['quality_issues']
    changed={**unchanged,'scenario_id':'another','reasoning':'A revised trace','needs_model_review':True}
    for key in ['quality_decision','quality_issues','choice_summary']: changed.pop(key)
    class Client:
        calls=0
        def chat(self,*args,**kwargs):
            self.calls+=1
            return ChatResult('<decision>accept</decision><issues>Fresh exact review</issues><summary>A</summary>',10,10,'stop')
    client=Client(); cfg=plan['config']
    ctx=Ctx(cfg=cfg,usage=Usage(),workers=1,run_dir=tmp_path/'run',smoke=False);ctx._client=client
    rows=op_llm_tagged(cfg['stages'][-1],cfg).fn(ctx,[unchanged,changed],None)
    assert client.calls==1
    by={r['scenario_id']:r for r in rows}
    assert by[unchanged['scenario_id']]==unchanged
    assert by['another']['quality_issues']=='Fresh exact review'
