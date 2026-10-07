# ABOUTME: Finalizes completed nonmoral evals after the obsolete source-string audit failure.
# ABOUTME: Never rents GPUs or reruns outcomes; retains original failure records and verifies immutable artifacts.
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scratch.plain_dose.audit_evals import audit_entry
from scratch.nonmoral.account_snapshot import snapshot
from src.infra import runpod
from src.infra.huggingface import hf_api

OUT = ROOT / 'output/nonmoral_plain'
EXPECTED_ERROR = 'ValueError: Source commit does not pin preserve_thinking=false; fetch missing source commits before audit'

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def write(p, data):
    p.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')

def main():
    state = read(OUT / 'campaign_status.json')
    assert state['phase'] == 'needs_attention', 'Wait for original driver to finish'
    owners = {k: read(OUT / k / 'status.json') for k in ('train', 'odcv', 'mask')}
    assert owners['train']['phase'] == 'complete' and owners['train']['terminated']
    publication = read(OUT / 'train/publication_verified.json')
    assert publication['verified'] and publication['steps'] == 143
    receipts = []
    for kind in ('odcv', 'mask'):
        owner = owners[kind]
        assert owner['eval_exit'] == 0 and owner['terminated']
        assert owner['phase'] == 'failed' and owner['error'] == EXPECTED_ERROR
        roots = [f.parent.parent for f in (OUT / kind / 'eval').rglob('metadata/run_meta.json')
                 if (f.parent.parent / 'results/results.json').exists()]
        assert len(roots) == 1
        repo = 'dougalldeepmind/2026-10-07-' + kind + '-qwen36-0-nonmoral-original-15'
        receipt = audit_entry({'eval': kind, 'run_dir': str(roots[0]), 'target': publication['repo'],
            'target_revision': publication['revision'], 'hf_repo': repo,
            'hf_revision': hf_api().dataset_info(repo).sha})
        write(OUT / kind / 'recovered_publication_audit.json', receipt)
        receipts.append(receipt)
    active = runpod.active_pods()
    owned_ids = {s['owned_pod'] for s in owners.values()}
    assert not owned_ids.intersection(p['id'] for p in active), 'Owned GPU remains'
    # Training status omitted a teardown timestamp; next admission snapshot bounds its end.
    train_end = datetime.fromisoformat(state['account_before_evals']['time_utc']).timestamp()
    costs = {}
    for kind, owner in owners.items():
        end = train_end if kind == 'train' else owner['termination_verified_epoch']
        hourly = owner['hourly_usd'] if kind == 'train' else owner['hourly_usd'] + .10
        costs[kind] = {'created_epoch': owner['created_epoch'], 'end_upper_bound_epoch': end,
            'hourly_usd_with_storage_allowance': hourly,
            'conservative_usd': (end-owner['created_epoch'])/3600*hourly}
    assert sum(v['conservative_usd'] for v in costs.values()) < 40
    before = OUT / 'campaign_status_before_audit_recovery.json'
    assert not before.exists(), 'Recovery already recorded'
    write(before, state)
    final = {'verified': True, 'adapter': publication, 'eval_receipts': receipts,
        'zero_owned_gpus': True, 'verified_at': datetime.now(timezone.utc).isoformat(),
        'costs': costs, 'gpu_cost_conservative_usd': sum(v['conservative_usd'] for v in costs.values()),
        'cost_note': 'Upper-bound lifetime estimate with $0.10/hr storage allowance; not a provider invoice. Judges separate.',
        'original_failure_record': str(before), 'audit_repair_commit': '8ccfe82d',
        'account_after': snapshot()}
    write(OUT / 'final_verification.json', final)
    state.update(phase='complete', error=None, recovery={'kind': 'audit-only', 'original_status': str(before)},
        eval_receipts=receipts, zero_owned_gpus=True, final_verification=str(OUT / 'final_verification.json'))
    write(OUT / 'campaign_status.json', state)
    (OUT / 'keep_awake.stop').write_text('verified complete\n', encoding='utf-8')
    print(json.dumps({'verified':True,'costs':costs,'eval_receipts':receipts},indent=2))

if __name__ == '__main__':
    main()
