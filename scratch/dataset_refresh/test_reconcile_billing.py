# ABOUTME: Offline reconciliation verifies native billing identity and preserves failed content/accounting.
from copy import deepcopy
import json
from pathlib import Path
import pytest
from scratch.dataset_refresh import reconcile_billing as mod


@pytest.fixture
def case(tmp_path):
    budget, proof = tmp_path/'budget', tmp_path/'proof'
    req = {'model':'m', 'messages':[]}
    entry = {'call_id':0,'model':'m','status':'uncertain_failure','request_sha256':mod.runtime.digest(req),'charged_or_reserved_usd':1.0}
    diag = {'generation_id':'g','provider':'p','usage':{'prompt_tokens':2,'completion_tokens':3,'cost':0.2,'cost_details':{'upstream_inference_cost':0.3}}}
    raw = {'request':req,'accounting':entry,'diagnostics':diag}
    mod.runtime.write_json(budget/'raw_calls/000000.json',raw)
    mod.runtime.write_json(budget/'spend.json',[entry])
    mod.runtime.write_json(proof/'inputs.json',{'entries':[{'call_id':0,'entry':entry,'ledger_entry_sha256':mod.runtime.digest(entry),'raw_sha256':mod.runtime.digest((budget/'raw_calls/000000.json').read_bytes()),'generation_id':'g','diagnostics':diag}]})
    mod.runtime.write_json(proof/'models_catalogue.response.json',{'data':[{'id':'m','canonical_slug':'m-permanent'}]})
    mod.runtime.write_json(proof/'000000.response.json',{'data':{'id':'g','model':'m-permanent','provider_name':'p','native_tokens_prompt':2,'native_tokens_completion':3,'total_cost':0.2,'upstream_inference_cost':0.1}})
    bind(proof)
    return budget, proof


def bind(proof):
    mod.runtime.write_json(proof/'000000.verification.json',{'http_status':200,'response_sha256':mod.runtime.digest((proof/'000000.response.json').read_bytes())})
    mod.runtime.write_json(proof/'full_file_manifest.json',{p.name:mod.runtime.digest(p.read_bytes()) for p in proof.iterdir() if p.name!='full_file_manifest.json'})


def test_dry_run_and_apply_preserve_raw_and_failed_status(case):
    budget,proof=case
    before=(budget/'spend.json').read_bytes(); raw=(budget/'raw_calls/000000.json').read_bytes()
    plan=mod.reconcile(budget,proof)
    assert plan['release_usd']==pytest.approx(0.7)
    assert (budget/'spend.json').read_bytes()==before
    mod.reconcile(budget,proof,apply=True)
    entry=mod.read(budget/'spend.json')[0]
    assert entry['status']=='billing_verified_failure' and entry['charged_or_reserved_usd']==0.3
    assert entry['api_reported_cost_usd']==0.2
    assert (budget/'raw_calls/000000.json').read_bytes()==raw
    archive=next((budget/'billing_reconciliations').iterdir())
    assert (archive/'spend_before.json').read_bytes()==before
    assert (archive/'raw_calls/000000.json').read_bytes()==raw
    with pytest.raises(ValueError): mod.reconcile(budget,proof,apply=True)


@pytest.mark.parametrize('field,value',[('id','other'),('model','wrong'),('provider_name','wrong'),('native_tokens_prompt',9),('native_tokens_completion',9),('total_cost',None),('total_cost',float('nan')),('total_cost',2)])
def test_mismatch_or_unknown_cost_refused(case,field,value):
    budget,proof=case
    data=mod.read(proof/'000000.response.json'); data['data'][field]=value
    mod.runtime.write_json(proof/'000000.response.json',data); bind(proof)
    with pytest.raises(ValueError): mod.reconcile(budget,proof,apply=True)
    assert mod.read(budget/'spend.json')[0]['status']=='uncertain_failure'


@pytest.mark.parametrize('change',['ledger','raw','proof'])
def test_changed_evidence_refused(case,change):
    budget,proof=case
    p={'ledger':budget/'spend.json','raw':budget/'raw_calls/000000.json','proof':proof/'000000.response.json'}[change]
    value=mod.read(p)
    if change=='ledger': value[0]['status']='settled'
    else: value['changed']=True
    mod.runtime.write_json(p,value)
    with pytest.raises(ValueError): mod.reconcile(budget,proof,apply=True)
