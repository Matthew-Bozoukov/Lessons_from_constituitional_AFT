# ABOUTME: Offline checks for exact stakes patches and real SynthDoc stage wiring.
# ABOUTME: Guards source preservation, overlap rejection, and single-turn export.
import json
from types import SimpleNamespace
import pytest
from omegaconf import OmegaConf
from scratch.nonmoral.stakes import CONFIG, apply_edits, materialize
from src.data.synth.ours.pipeline import run
from src.data.synth.ours.hf_cache import read_jsonl


def source():
    return dict(scenario_id='case_1',trait_name='craft',system='Help.',user='Choose a cover.',
                reasoning='Green matches the binding.',response='Green.',feedback='')


def test_exact_edits_preserve_other_fields_and_reject_ambiguity():
    r=source()
    out=apply_edits(r,[dict(field='user',old='cover',new='cover for a $2 print')])
    assert out['user']=='Choose a cover for a $2 print.'
    assert all(out[k]==r[k] for k in ('system','reasoning','response'))
    with pytest.raises(ValueError,match='occurs 0'):
        apply_edits(r,[dict(field='user',old='COVER',new='x')])
    with pytest.raises(ValueError,match='Overlapping'):
        apply_edits(r,[dict(field='user',old='cover',new='x'),dict(field='user',old='a cover',new='y')])
    with pytest.raises(ValueError,match='Forbidden'):
        apply_edits(r,[dict(field='system',old='',new='x')])


def test_actual_pipeline_preserves_originals(tmp_path):
    r=source()
    pair=dict(low=[dict(field='user',old='',new=' A mistake wastes $2.')],
              high=[dict(field='user',old='',new=' A mistake wastes $2000.')])
    cfg=OmegaConf.to_container(OmegaConf.load(CONFIG),resolve=True)
    (tmp_path/'inputs.jsonl').write_text(json.dumps(r)+'\n')
    cfg.update(source=dict(local_dir=str(tmp_path),snapshot='inputs.jsonl'),total_scenarios=1)
    replies=iter(['<pair>'+json.dumps(pair)+'</pair>',
                  '<review>'+json.dumps(dict(decision='accept'))+'</review>'])
    class Fake:
        def chat(self,**kwargs):
            return SimpleNamespace(content=next(replies),finish_reason='stop',prompt_tokens=10,
                                   completion_tokens=10,provider='offline')
    d=tmp_path/'run'; d.mkdir()
    run(cfg,resume=str(d),client=Fake())
    result=read_jsonl(d/'dataset.jsonl')[0]
    assert all(result[k]==v for k,v in r.items())
    _,low,high=materialize(result)
    assert low['user']!=high['user']
    assert low['response']==high['response']==r['response']


def test_shared_template_changes_only_numeric_stakes():
    r=source()
    r['pair_json']=json.dumps(dict(edits=[dict(field='user',old='',new=' Bad fit wastes $__STAKE__ of printing.'),
                    dict(field='reasoning',old='',new=' Avoid the $__STAKE__ wasted print run.')],
                    low_value='20',high_value='2000',unit='dollars'))
    _,low,high=materialize(r)
    for k in ('user','reasoning','response','system'):
        assert low[k].replace('20','AMOUNT')==high[k].replace('2000','AMOUNT')


def test_auditor_fix_changes_only_generated_fields(tmp_path,monkeypatch):
    import hashlib
    from scratch.nonmoral import stakes
    r=source()
    pair=dict(edits=[dict(field='user',old='',new=' Bad fit wastes $__STAKE__ of printing.'),
                    dict(field='reasoning',old='',new=' Avoid the $__STAKE__ wasted print run.')],
                    low_value='20',high_value='2000',unit='dollars')
    r.update(pair_json=json.dumps(pair),review_json='{"decision":"accept"}')
    (tmp_path/'accepted.jsonl').write_text(json.dumps(r)+'\n')
    audit=dict(findings={'case_1':dict(status='new_issue',pair_json_sha256=hashlib.sha256(r['pair_json'].encode()).hexdigest(),
         proposed_pair_replacements=[dict(path=['high_value'],old='2000',new='1000')])})
    (tmp_path/'independent_audit.json').write_text(json.dumps(audit))
    for name in ('audit_middle.json','audit_later.json'):
        (tmp_path/name).write_text('{"dispositions":{}}')
    monkeypatch.setattr(stakes,'ROOT',tmp_path)
    stakes.audit_corrections()
    result=read_jsonl(tmp_path/'locally_corrected_candidates.jsonl')[0]
    assert all(result[k]==r[k] for k in ('system','user','reasoning','response'))
    assert 'review_json' not in result
    assert result['pre_local_correction_review_json']==r['review_json']
    assert json.loads(result['pair_json'])['high_value']=='1000'
