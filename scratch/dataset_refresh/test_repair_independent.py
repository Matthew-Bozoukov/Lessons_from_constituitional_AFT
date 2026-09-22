# ABOUTME: Offline regression coverage for externally audited Sonnet repairs and separate adoption.
# ABOUTME: Verify unchanged requests, strict reviews, zero repeat billing and preserved rejected artifacts.
import json
from pathlib import Path
import pytest
from scratch.dataset_refresh import repair_independent as mod
from scratch.dataset_refresh import run as base
from scratch.dataset_refresh.test_per_row import case, offline, FakeClient, client_for, generate, tagged, clean_verdict, GOOD, BAD


@pytest.fixture
def prepared(case,monkeypatch):
    generate(case,client_for(response=BAD))
    meta=json.loads((case.root/'run_meta.json').read_text())
    base.save_checkpoint(case.root/'run_meta.json',meta)
    original=(case.out/'result.json').read_bytes()
    base.save_checkpoint(case.out/'independent_exclusion.json',{'result_sha256':base.digest(original),'reason':'Invented authorization; wait or seek explicit exception.'})
    monkeypatch.setattr(base,'validate_arm',lambda root,arm:case.cfg)
    case.original=original
    return case


def fake(monkeypatch,bad=False):
    client=FakeClient({'independent_rewrite_100':tagged('The agreed monthly procedure remains binding.',GOOD,True),
        'grounding_100':{'accepted':True,'issues':[]} if bad else clean_verdict(),
        'review_100':{'accepted':True,'factual_reasoning':True,'stakes':1}})
    monkeypatch.setattr(base,'BudgetClient',lambda *a,**k:client)
    return client


def test_proposal_is_sonnet_only_blind_and_does_not_adopt(prepared,monkeypatch):
    c=prepared;client=fake(monkeypatch)
    result=mod.propose(c.root,c.arm,c.candidate['candidate_id'],90)
    assert result['status']=='accepted'
    assert (c.out/'result.json').read_bytes()==c.original
    assert base.load_result(c.out/'result.json')['status']=='rejected'
    assert [x[0] for x in client.calls]==['independent_rewrite_100','grounding_100','review_100']
    grounding=json.dumps(client.calls[1][1]['messages'])
    assert 'SECRET_TRAIT' not in grounding and 'Invented authorization;' not in grounding
    assert result['record']['user']==base.load_checkpoint(c.out/'result.json')['record']['user']
    assert all(req['model']=='anthropic/claude-sonnet-5' for _,req in client.calls)
    mod.propose(c.root,c.arm,c.candidate['candidate_id'],90)
    assert len(client.calls)==3


def test_adoption_preserves_original_and_verifies_receipts(prepared,monkeypatch):
    c=prepared;fake(monkeypatch)
    mod.propose(c.root,c.arm,c.candidate['candidate_id'],90)
    sha=base.digest((c.out/'independent_candidate_100.json').read_bytes())
    with pytest.raises(ValueError,match='Approval'):
        mod.adopt(c.root,c.arm,c.candidate['candidate_id'],'0'*64,'Reviewed')
    mod.adopt(c.root,c.arm,c.candidate['candidate_id'],sha,'Read corrected answer; monthly rule now preserved.')
    assert base.load_result(c.out/'result.json')['status']=='accepted'
    assert base.load_result(c.out/'result.json')['record']['response']==GOOD
    archive=c.out/'recovered_failures'/'independent_repair_100'
    assert (archive/'result.json').read_bytes()==c.original
    assert base.load_checkpoint(archive/'independent_exclusion.json')['result_sha256']==base.digest(c.original)


def test_malformed_critic_stays_failed_and_never_retries(prepared,monkeypatch):
    c=prepared;client=fake(monkeypatch,bad=True)
    result=mod.propose(c.root,c.arm,c.candidate['candidate_id'],90)
    assert result['status']=='failed' and len(client.calls)==2
    assert base.load_result(c.out/'result.json')['status']=='rejected'
    mod.propose(c.root,c.arm,c.candidate['candidate_id'],90)
    assert len(client.calls)==2


def test_started_without_checkpoint_blocks_duplicate_calls(prepared,monkeypatch):
    c=prepared;client=fake(monkeypatch)
    base.save_checkpoint(c.out/'independent_rewrite_100.started.json',{'prior':'uncertain'})
    with pytest.raises(base.BudgetStop,match='already started'):
        mod.propose(c.root,c.arm,c.candidate['candidate_id'],90)
    assert not client.calls


def test_changed_exclusion_source_refused(prepared,monkeypatch):
    c=prepared;fake(monkeypatch)
    base.save_checkpoint(c.out/'independent_exclusion.json',{'result_sha256':'0'*64,'reason':'not bound'})
    with pytest.raises(ValueError,match='bound'):
        mod.propose(c.root,c.arm,c.candidate['candidate_id'],90)
