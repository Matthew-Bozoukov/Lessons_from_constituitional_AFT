# ABOUTME: Verify durable spending reservations, crash retention and exclusive sampler ownership.
# ABOUTME: Linux-only ledger matches the native-Docker Vast driver.
import sys
import pytest

pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='Linux owned inference ledger')


def test_ledger_preserves_ambiguous_calls_and_frozen_identity(tmp_path):
    from src.infra.endpoints.tinker_budget import Budget
    path = tmp_path/'ledger.json'
    b = Budget(path, 1, 'openai/gpt-oss-120b:peft:131072', 'tinker://test')
    first = b.reserve(100000, 10000)
    pending = b.reserve(200000, 10000)
    b.settle(first, 100)
    assert b.data['requests'][pending]['state'] == 'reserved'
    total = sum(x['upper_usd'] for x in b.data['requests'].values())
    b.owner.close()
    reopened = Budget(path, 1, 'openai/gpt-oss-120b:peft:131072', 'tinker://test')
    assert sum(x['upper_usd'] for x in reopened.data['requests'].values()) == total
    with pytest.raises(BlockingIOError):
        Budget(path, 1, 'openai/gpt-oss-120b:peft:131072', 'tinker://test')
    with pytest.raises(RuntimeError, match='ceiling'):
        reopened.reserve(2000000, 10000)
    reopened.owner.close()
    with pytest.raises(AssertionError, match='identity'):
        Budget(path, 2, 'openai/gpt-oss-120b:peft:131072', 'tinker://test')


def test_arms_share_one_cap_and_only_confirmed_cache_hits_get_discount(tmp_path):
    from src.infra.endpoints.tinker_budget import Budget
    allowed = ['tinker://base', 'tinker://control']
    a = Budget(tmp_path/'ledger.json', .20, 'openai/gpt-oss-120b:peft:131072', allowed[0], allowed)
    b = Budget(tmp_path/'ledger.json', .20, 'openai/gpt-oss-120b:peft:131072', allowed[1], allowed)
    key = a.reserve(100000, 10000)
    other = b.reserve(100000, 10000)
    with pytest.raises(RuntimeError):
        a.reserve(100000, 10000)
    a.settle(key, 100, 90000)
    assert a.data['requests'][key]['upper_usd'] == pytest.approx(.022034)
    b.settle(other, 100)
    assert len(b.data['requests']) == 2


def test_explicit_unlimited_mode_preserves_accounting_and_ownership(tmp_path):
    from src.infra.endpoints.tinker_budget import Budget
    path=tmp_path/'ledger.json'
    b=Budget(path,None,'openai/gpt-oss-120b:peft:131072','tinker://base')
    key=b.reserve(1000000000,10000)
    b.settle(key,100)
    assert b.data['ceiling_usd'] is None and b.data['requests'][key]['upper_usd']>12
    b.owner.close()
    reopened=Budget(path,None,'openai/gpt-oss-120b:peft:131072','tinker://base')
    assert reopened.data['requests'][key]['state']=='completed'
    with pytest.raises(BlockingIOError):Budget(path,None,'openai/gpt-oss-120b:peft:131072','tinker://base')
