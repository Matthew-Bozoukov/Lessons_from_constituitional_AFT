# ABOUTME: Offline preservation tests keep failed artifacts byte-exact and refuse open/changed evidence.
# ABOUTME: The archive is explicitly nontraining and includes original raw calls and source recipes.
import json
import tarfile
from pathlib import Path
import pytest
from scratch.dataset_refresh import preserve_partial as mod


@pytest.fixture
def snapshot(tmp_path,monkeypatch):
    root=tmp_path/'origin';budget=root/'budget';evidence=tmp_path/'evidence';arm='nonmoral-advice'
    cfg={'pipeline':arm,'constitution':'constitution.md','constitution_sha256':'c'*64,'models':{'author':{'model':'historic-model'}},'prompts':{'literal':'Full historical prompt.'}}
    mod.runtime.write_json(root/arm/'config.json',cfg)
    mod.runtime.save_checkpoint(root/'run_meta.json',{'arms':{arm:{'config_sha256':mod.runtime.digest(cfg)}}})
    mod.runtime.save_checkpoint(root/arm/'records/t1_000_v0/result.json',{'status':'failed','trait_id':'t1','error':'Original failure.'})
    req={'messages':[{'role':'user','content':'Keep exact bytes: café.'}],'model':'historic-model'}
    entry={'call_id':0,'status':'settled','charged_or_reserved_usd':0.1,'request_sha256':mod.runtime.digest(req)}
    mod.runtime.write_json(budget/'spend.json',[entry]);mod.runtime.write_json(budget/'raw_calls/000000.json',{'request':req,'accounting':entry})
    mod.runtime.write_json(evidence/'audit.json',{'scope':'Incomplete review; no global quality approval.'})
    manifest=tmp_path/'input.json';mod.runtime.write_json(manifest,{'root_directories':[str(root)],'shared_budget_root':str(budget),'root_archive_exclusions':{str(root):['budget']},'supporting_evidence_directories':[str(evidence)],'final_summary':'Fixture:0 cleared rows;716short. One failed record preserved; no release selection.'})
    def code(out,commit):
        p=out/'source_code.tar.gz';p.write_bytes(b'committed-source-fixture');return {'commit':commit,'sha256':mod.runtime.digest(p.read_bytes())}
    monkeypatch.setattr(mod,'freeze_code',code)
    return manifest,tmp_path/'snapshot',root,budget,evidence


def test_closed_snapshot_keeps_failed_bytes_and_no_train_split(snapshot):
    manifest,out,root,budget,evidence=snapshot
    mod.prepare(manifest,out,'f'*40,1,'2026-09-15')
    assert not (out/'dataset.jsonl').exists() and not (out/'mixture.jsonl').exists()
    readiness=mod.publication.read_json(out/'readiness.json')
    assert readiness['train_ready'] is False and readiness['phase_census'][0]['effective_terminal_counts']=={'failed':1}
    front=mod.publication.read_json(out/'card_front_matter.json')
    assert front['configs'][0]['data_files'][0]['split']=='audit'
    with tarfile.open(out/'archives/origin_00.tar.gz') as archive:
        name='nonmoral-advice/records/t1_000_v0/result.json'
        assert archive.extractfile(name).read()==(root/name).read_bytes()
        assert not any(n.startswith('budget/') for n in archive.getnames())
    with tarfile.open(out/'archives/shared_budget.tar.gz') as archive:
        assert archive.extractfile('raw_calls/000000.json').read()==(budget/'raw_calls/000000.json').read_bytes()
    hashes=mod.publication.read_json(out/'files_manifest.json')
    assert hashes=={k:v for k,v in mod.inventory(out).items() if k!='files_manifest.json'}
    assert 'INCOMPLETE RESEARCH AUDIT' in (out/'README.md').read_text(encoding='utf8')


@pytest.mark.parametrize('kind',['reserved','cutoff','overbudget','raw_changed','changed_config','nested_output'])
def test_open_or_inconsistent_snapshot_refused(snapshot,kind):
    manifest,out,root,budget,evidence=snapshot;cutoff=1
    if kind in ('reserved','overbudget'):
        rows=mod.publication.read_json(budget/'spend.json');rows[0]['status' if kind=='reserved' else 'charged_or_reserved_usd']='reserved' if kind=='reserved' else 251
        mod.runtime.write_json(budget/'spend.json',rows)
    elif kind=='cutoff':cutoff=0
    elif kind=='raw_changed':
        raw=mod.publication.read_json(budget/'raw_calls/000000.json');raw['request']['model']='other';mod.runtime.write_json(budget/'raw_calls/000000.json',raw)
    elif kind=='changed_config':mod.runtime.write_json(root/'nonmoral-advice/config.json',{})
    elif kind=='nested_output':out=evidence/'nested'
    with pytest.raises(ValueError):mod.prepare(manifest,out,'f'*40,cutoff,'2026-09-15')


def test_archive_refuses_source_change(snapshot,monkeypatch):
    _,out,_,_,evidence=snapshot;original=mod.tarfile.TarFile.addfile
    def add(self,item,fileobj=None):
        original(self,item,fileobj);(evidence/'audit.json').write_text('{}')
    monkeypatch.setattr(mod.tarfile.TarFile,'addfile',add)
    with pytest.raises(ValueError,match='changed'):mod.archive_tree(evidence,out/'changed.tar.gz')


def test_historical_missing_metadata_receipt_is_explicit(snapshot):
    manifest,out,root,_,_=snapshot
    (root/'run_meta.receipt.json').unlink()
    mod.prepare(manifest,out,'f'*40,1,'2026-09-15')
    row=mod.runtime.read_rows(out/'audit_index.jsonl')[0]
    assert row['run_meta_receipt']=='absent_in_historical_phase'


def test_only_exact_provenance_authorization_allowlisted():
    for text in ('User authorized iterative repairs; one external Sonnet correction after independent audit.',
                 'Explicit user-authorized iterative row repair; root approved completed short draft as untrained input.'):
        mod.publication.safe_audit({'authorization':text})
    for value in ({'authorization':'Bearer example'}, {'headers':{}}, {'api_key':'example'}):
        with pytest.raises(ValueError):mod.publication.safe_audit(value)
