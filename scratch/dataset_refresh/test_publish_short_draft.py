# ABOUTME: Publisher validates the distinct short-draft repair chain without accepting the failed original.
import pytest
from scratch.dataset_refresh import publish_composite as publisher
from scratch.dataset_refresh import recover_short_draft as recovery
from scratch.dataset_refresh import run as base
from scratch.dataset_refresh.test_per_row import case, offline
from scratch.dataset_refresh.test_recover_missing_changes import ready
from scratch.dataset_refresh.test_recover_short_draft import short, fresh


@pytest.fixture
def adopted(short,monkeypatch):
    c=short; fresh(c,monkeypatch)
    recovery.propose(c.root,c.arm,c.candidate['candidate_id'],245)
    recovery.adopt(c.root,c.arm,c.candidate['candidate_id'],base.digest((c.out/recovery.PROPOSAL).read_bytes()),'Independent full-answer approval.')
    return c


def test_actual_recovery_adoption_passes_distinct_publication_path(adopted):
    c=adopted; result=base.load_result(c.out/'result.json')
    publisher.validate_adoption(c.out/'result.json',result,c.cfg)
    assert base.load_checkpoint(c.out/'recovered_failures/short_draft_recovery_200/result.json')['status']=='failed'


@pytest.mark.parametrize('change',['original','author','review','source_review','instruction','physical'])
def test_changed_recovery_evidence_refused(adopted,change):
    c=adopted; result=base.load_result(c.out/'result.json')
    names={'original':'recovered_failures/short_draft_recovery_200/result.json','author':recovery.AUTHOR+'.json',
           'review':'review_200.json','source_review':recovery.SOURCE_REVIEW,'instruction':recovery.MANIFEST,'physical':recovery.AUTHOR+'.physical.json'}
    path=c.out/names[change]; value=base.load_checkpoint(path)
    if change=='original': value['status']='accepted'
    elif change=='instruction': value['appended_recovery_instruction']='Different instruction'
    else:value['changed']=True
    base.save_checkpoint(path,value)
    with pytest.raises((ValueError,base.BudgetStop)): publisher.validate_adoption(c.out/'result.json',result,c.cfg)
