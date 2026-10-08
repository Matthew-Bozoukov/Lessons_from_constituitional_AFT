# ABOUTME: Ensure admission rebalance prioritizes a starved arm without reordering its own queue.
# ABOUTME: Pure scheduling test, no live files or inference.
from copy import deepcopy
from scratch.gptoss_swe.rebalance_waiters import order_waiters


def test_late_arm_receives_slots_and_active_rows_are_untouched():
    owners={};rows=[]
    for arm,count in [('control',47),('da15',33),('base',0)]:
        for n in range(count):
            key=arm+'-active-'+str(n);owners[key]=arm
            rows.append(dict(id=key,state='active',created=n))
        for n in range(80-count):
            key=arm+'-waiting-'+str(n);owners[key]=arm
            rows.append(dict(id=key,state='waiting',created=100+n))
    saved=deepcopy(rows);ordered=order_waiters(rows,owners)
    assert rows==saved
    assert len(ordered)==160 and len({r['id'] for r in ordered})==160
    assert all(owners[r['id']]=='base' for r in ordered[:33])
    for arm in ('base','control','da15'):
        assert [r['created'] for r in ordered if owners[r['id']]==arm]==sorted(r['created'] for r in ordered if owners[r['id']]==arm)
