# ABOUTME: Publication/recovery share validation of reconciled failures without treating them as successful outputs.
import pytest
from scratch.dataset_refresh import billing_evidence as mod
from scratch.dataset_refresh import reconcile_billing
from scratch.dataset_refresh.test_reconcile_billing import case


def test_valid_chain_and_original_raw_retained(case):
    budget,proof=case
    reconcile_billing.reconcile(budget,proof,apply=True)
    call=mod.read_json(budget/'spend.json')[0]
    raw=mod.read_json(budget/'raw_calls/000000.json')
    archive=mod.validate_billing_call(budget,call,raw)
    assert archive.is_dir() and raw['accounting']['status']=='uncertain_failure'


@pytest.mark.parametrize('change',['cost','raw','plan','proof','archive_raw'])
def test_changed_chain_refused(case,change):
    budget,proof=case
    reconcile_billing.reconcile(budget,proof,apply=True)
    call=mod.read_json(budget/'spend.json')[0]; raw=mod.read_json(budget/'raw_calls/000000.json')
    archive=budget/'billing_reconciliations'/call['billing_reconciliation_sha256']
    if change=='cost': call['charged_or_reserved_usd']=0
    elif change=='raw': raw['accounting']['model']='different'
    else:
        path=archive/{'plan':'plan.json','proof':'000000.response.json','archive_raw':'raw_calls/000000.json'}[change]
        value=mod.read_json(path); value['changed']=True; mod.runtime.write_json(path,value)
    with pytest.raises(ValueError): mod.validate_billing_call(budget,call,raw)
