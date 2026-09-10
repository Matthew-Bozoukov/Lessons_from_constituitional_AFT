# ABOUTME: Offline contracts for author-first local gating of deferred Sonnet review.
# ABOUTME: Checks call selection, immutable lineage, explicit final adjudication and legacy stage behavior.
import json
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from scratch.nonmoral import broader_data as data


def dump_rows(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')


@pytest.fixture
def deferred(tmp_path, monkeypatch):
    config = OmegaConf.to_container(OmegaConf.load('configs/data/synth/nonmoral-broader.yaml'),resolve=True)
    config['production']['defer_model_review'] = True
    config['production']['phase_cap_usd']['review'] = 5
    config_path = tmp_path/'recipe.yaml'
    OmegaConf.save(OmegaConf.create(config),config_path)
    root = tmp_path/'corpus'
    root.mkdir()
    (root/'spend.json').write_text('[]')
    monkeypatch.setattr(data,'ROOT',root)
    batch = root/'production/batch07'
    source = batch/'sources/run/dataset.jsonl'
    rows = [dict(scenario_id=f'case{i}',domain='craft',user=f'Create complete artifact {i}.') for i in range(3)]
    dump_rows(source,rows)
    data.write_json(batch/'sources/status.json',dict(run_dir=str(source.parent),config_sha256=data.file_sha256(config_path)))
    source_review = batch/'source_review.json'
    data.write_json(source_review,dict(reviewer='local',source_sha256=data.file_sha256(source),
        dispositions={r['scenario_id']:dict(decision='accept',reason='Complete source.') for r in rows}))
    events = []
    monkeypatch.setattr(data,'verify_live_prices',lambda *a: events.append('price') or {})
    class Chat:
        def retry_with(self,**kwargs):
            return lambda *a,**kw: None
    monkeypatch.setattr(data,'OpenRouterClient',lambda: SimpleNamespace(chat=Chat()))
    monkeypatch.setattr(data,'CappedClient',lambda *a,**kw: object())
    def engine(effective,resume,client):
        names = [s.name for s in data.build_stages(effective)]
        selected = data.read_rows(Path(effective['source']['local_dir'])/'inputs.jsonl')
        events.append((names,[r['scenario_id'] for r in selected]))
        if len(names) == 2 and names[-1] == config['production']['answer_stage']['name']:
            selected = [dict(r,reasoning='Compare two feasible craft approaches.',response='Complete artifact.') for r in selected]
        else:
            selected = [dict(r,quality_decision='reject',quality_issues='Model assessment for explicit adjudication.',choice_summary='Craft choice') for r in selected]
        dump_rows(Path(resume)/'dataset.jsonl',selected)
    monkeypatch.setattr(data,'run',engine)
    args = SimpleNamespace(batch='batch07',phase='answers',source_review=source_review,
                           answer_review=None,execute=True,config=config_path)
    return config,args,batch,events


def make_author_gate(batch):
    status = json.loads((batch/'answers/status.json').read_text())
    path = Path(status['run_dir'])/'dataset.jsonl'
    rows = data.read_rows(path)
    gate = dict(status='complete',reviewer='local full reader',author_snapshot_sha256=data.file_sha256(path),
                dispositions={r['scenario_id']:dict(decision=decision,reason='Concrete local finding.',
                    conversation_sha256=data.conversation_sha256(r))
                    for r,decision in zip(rows,['accept','reject','hold'])})
    gate_path = batch/'author_review.json'
    data.write_json(gate_path,gate)
    return gate_path,rows,path


def test_deferred_dispatch_filters_paid_reviews_and_requires_final_adjudication(deferred):
    cfg,args,batch,events = deferred
    data.production(cfg,args)
    assert events[-1][0] == ['planned_cases',cfg['production']['answer_stage']['name']]
    assert len(events[-1][1]) == 3
    gate,authors,author_path = make_author_gate(batch)
    assert data.accepted_production(data.ROOT) == ([],[])
    with pytest.raises(ValueError,match='not complete'):
        data.final_answer_phase(batch)
    args.phase='review'; args.source_review=None; args.answer_review=gate
    data.production(cfg,args)
    assert events[-1] == (['planned_cases',cfg['stages'][-1]['name']],['case0'])
    assert data.accepted_production(data.ROOT) == ([],[])
    phase,status,path = data.final_answer_phase(batch)
    assert phase == 'review'
    final_rows = data.read_rows(path)
    assert final_rows[0]['quality_decision'] == 'reject'
    data.write_json(batch/'answer_review.json',dict(reviewer='root adjudication',dataset_sha256=data.file_sha256(path),
        dispositions={'case0':dict(decision='accept',reason='Explicitly resolved model disagreement.',
                                  conversation_sha256=data.conversation_sha256(final_rows[0]))}))
    accepted,provenance = data.accepted_production(data.ROOT)
    assert accepted == final_rows and len(data.read_rows(author_path)) == 3
    assert provenance[0]['author_dataset_sha256'] == data.file_sha256(author_path)
    paid_checks = events.count('price')
    with pytest.raises(ValueError,match='already dispatched'):
        data.production(cfg,args)
    assert events.count('price') == paid_checks and not (data.ROOT/'generation.lock').exists()
    # Even with a newly updated final dataset/status hash, changing the answer cannot bypass its author gate.
    final_rows[0]['response'] = 'A different artifact'
    dump_rows(path,final_rows)
    status['dataset_sha256'] = data.file_sha256(path)
    data.write_json(batch/'review/status.json',status)
    with pytest.raises(ValueError,match='Changed, unapproved'):
        data.final_answer_phase(batch)


def test_incomplete_gate_and_model_config_drift_fail_before_calls(deferred):
    cfg,args,batch,events = deferred
    data.production(cfg,args)
    gate,_,_ = make_author_gate(batch)
    args.phase='review'; args.source_review=None; args.answer_review=gate
    document = json.loads(gate.read_text())
    del document['dispositions']['case2']
    data.write_json(gate,document)
    before = list(events)
    with pytest.raises(ValueError,match='every authored ID'):
        data.production(cfg,args)
    assert events == before
    make_author_gate(batch)
    cfg['production']['models']['reviewer']['temperature'] = 0.9
    OmegaConf.save(OmegaConf.create(cfg),args.config)
    with pytest.raises(ValueError,match='exact frozen source config'):
        data.production(cfg,args)
    assert events == before


def test_legacy_answers_keep_author_and_model_stages(deferred):
    cfg,args,batch,events = deferred
    cfg['production'].pop('defer_model_review')
    data.production(cfg,args)
    assert events[-1][0] == ['planned_cases',cfg['production']['answer_stage']['name'],cfg['stages'][-1]['name']]
    assert data.final_answer_phase(batch)[0] == 'answers'
