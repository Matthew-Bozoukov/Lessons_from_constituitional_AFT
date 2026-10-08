# ABOUTME: One-time, audited admission-only rebalance after the late base-arm handoff.
# ABOUTME: Run on Linux against the live campaign; no sampler restart, task cancellation, or budget mutation.
from collections import Counter, defaultdict
from pathlib import Path
import time


def order_waiters(rows, owners):
    active=Counter(owners[x['id']] for x in rows if x['state']=='active')
    queues=defaultdict(list)
    for row in sorted(rows,key=lambda x:(x['created'],x['id'])):
        if row['state']=='waiting':queues[owners[row['id']]].append(row)
    ordered=[]
    while any(queues.values()):
        arm=min((a for a,q in queues.items() if q),key=lambda a:(active[a],a))
        ordered.append(queues[arm].pop(0));active[arm]+=1
    return ordered


def main():
    from src.eval.capabilities.swebench_mini.fleet_state import atomic,read,lock
    root=Path('/srv/lasr/runs/gptoss-three-20261008')
    directory=root/'task-slots'
    receipt=root/'waiting-arm-rebalance.json'
    assert not receipt.exists(), 'One-time operation already recorded'
    campaign=read(root/'campaign-status-parallel.json')
    assert campaign['phase']=='running' and campaign['legacy_reserved_slots']==0
    with lock(directory/'.mutex'):
        owners={}
        for arm,entry in campaign['arms'].items():
            for fd in Path('/proc/'+str(entry['pid'])+'/fd').iterdir():
                try:path=fd.resolve()
                except OSError:continue
                if path.parent==directory and path.suffix=='.lease':owners[path.stem]=arm
        rows=[read(p) for p in directory.glob('*.json')]
        assert rows and all(r['id'] in owners and r['state'] in ('waiting','active') for r in rows)
        assert sum(r['state']=='active' for r in rows)<=80
        ordered=order_waiters(rows,owners)
        epoch=min(r['created'] for r in rows)-1
        audit=dict(at=time.time(),reason='User requested concurrent arms; late base queue was behind an entire DA15 waiting pool',
                   before=rows,owners=owners,waiting_order=[r['id'] for r in ordered],
                   effect='waiting admission priority only; active leases, task states, outcomes and budget unchanged')
        atomic(receipt,audit)
        for n,row in enumerate(ordered):
            atomic(directory/(row['id']+'.json'),row | {'created':epoch+n/1000,
                   'original_created':row['created'],'priority_receipt':str(receipt)})
    print({'reprioritized_waiters':len(ordered),'first_10_arms':[owners[r['id']] for r in ordered[:10]]})


if __name__=='__main__':main()
