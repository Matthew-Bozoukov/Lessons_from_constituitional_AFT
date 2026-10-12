# ABOUTME: Publish a completed inference campaign with explicitly proven missing judge verdicts.
# ABOUTME: No inference or judging calls; strict coverage, reservation evidence and owner-exit gates.
import argparse
import ctypes
import json
import math
import subprocess
from scratch.gptoss_swe import odcv_full as full

EXPECTED = {
    'base': {'alignment': set(), 'progress': set()},
    'control': {'alignment': {'mandated/Ai-Pharma-Quality-Control-Data-Suppression/rollout_001'}, 'progress': set()},
    'da15': {'alignment': {'incentivized/Educational-Assessment/rollout_001'},
             'progress': {'incentivized/Healthcare-Data-Analytics/rollout_001'}},
}


def axis_coverage(scores, keys, allowed):
    assert set(scores) == keys, 'Missing or unexpected cached judge cells'
    missing = {k for k, v in scores.items() if v.get('score') in (None, 'N/A')}
    assert missing == allowed, 'Unexplained missing judgments'
    assert all(math.isfinite(float(v['score'])) and 0 <= float(v['score']) <= 5
               for k, v in scores.items() if k not in missing)
    return len(scores) - len(missing)


def dead(receipt):
    pid = full.read(receipt)['pid']
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return
    code = ctypes.c_ulong()
    try:
        assert ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        assert code.value != 259, f'Owner PID {pid} still active; inspect identity before closeout'
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def validate():
    root = full.ROOT
    dead(root/'owner-started.json')
    dead(root/'judging-recovery-owner-started.json')
    assert full.read(root/'judging-recovery-exits.json') == dict(base=0, control=0, da15=0)
    report = full.status_data()
    evidence = {}
    for arm in full.ARMS:
        folder = root/arm
        dead(folder/'owner-started.json')
        dead(folder/'judging-recovery-started.json')
        assert full.read(folder/'status.json')['phase'] in ('complete', 'judged_with_exclusions')
        assert not full.read(folder/'shim-stopped.json')['state']['Running']
        r = report[arm]
        assert r['ended'] == 240 and r['running'] == r['waiting'] == r['reservations'] == 0
        keys = set()
        for path in (folder/'odcv/rollouts').glob('*/*/pass*/messages_record.txt'):
            variant, scenario, pass_name = path.relative_to(folder/'odcv/rollouts').parts[:3]
            assert path.stat().st_size > 0
            keys.add(f'{variant}/{scenario}/rollout_{int(pass_name[4:])-1:03d}')
        assert len(keys) == 240
        summary = full.read(folder/'summary.json')
        ledger = full.read(folder/'judge-budget.json')
        assert sum(e['charged_or_reserved_usd'] for e in ledger) <= float(full.CFG.judge_budget_per_arm_usd)
        explained_reservations = set()
        axes = {}
        for axis, filename in [('alignment', 'scores_gemini-3-flash-preview.json'),
                               ('progress', 'scores_progress_gemini-3-flash-preview.json')]:
            scores = full.read(folder/'odcv/results'/filename)
            allowed = EXPECTED[arm][axis]
            judged = axis_coverage(scores, keys, allowed)
            result = summary if axis == 'alignment' else summary['progress']
            # Upstream n_judged counts attempted cells, including explicit N/A.
            assert result['n_judged'] - result['n_dropped_all_na'] == judged
            assert result['n_dropped_all_na'] == len(allowed) and set(result['dropped']) == allowed
            exclusions = []
            for key in sorted(allowed):
                value = scores[key]
                reason = 'Unusable progress verdict after existing judge attempts; no further retry'
                if axis == 'alignment':
                    assert value['provider_exclusion'] is True
                    index = value['ledger_entry']
                    entry = ledger[index]
                    assert entry['status'] == 'reserved' and entry['response_diagnostic'] == value['response_diagnostic']
                    diagnostic = full.read(folder/entry['response_diagnostic'])
                    assert not diagnostic.get('choices') and 'PROHIBITED_CONTENT' in diagnostic['error']['message']
                    explained_reservations.add(index)
                    reason = diagnostic['error']['message']
                exclusions.append(dict(key=key,reason=reason,original_cached_verdict=value))
            axes[axis] = dict(judged=judged,excluded=len(allowed),exclusions=exclusions)
        if arm == 'da15':
            receipt = full.read(folder/'known-unsent-judge-receipt.json')
            index = receipt['entry']
            assert index == 87 and receipt['original_reservation_retained'] is True
            assert ledger[index]['request_sha256'] == receipt['request_sha256'] == '12809ef3cf9b50a302cac16733832f7a5f0c19aa22db37c1298c7f06cb4ca61d'
            assert ledger[index]['status'] == 'reserved'
            old_log = (folder/'before-judging-recovery/owner.log').read_text(errors='replace')
            assert 'line 62, in create' in old_log and 'PermissionError' in old_log
            explained_reservations.add(index)
        outstanding = {i for i,e in enumerate(ledger) if e['status'] != 'settled'}
        assert outstanding == explained_reservations, 'Unexplained judge request; cannot close'
        axes['retained_reservations'] = [dict(index=i,**ledger[i]) for i in sorted(outstanding)]
        evidence[arm] = axes
    inference = full.read(root/'inference-budget.json')
    assert all(r['state'] == 'completed' for r in inference['requests'].values())
    assert sum(r['inference_usd'] for r in report.values()) <= float(full.CFG.shared_inference_budget_usd)
    assert sum(r['inference_usd']+r['judge_usd'] for r in report.values()) <= float(full.CFG.total_budget_usd)
    return evidence


def publish():
    evidence = validate()
    full.claim(full.ROOT/'closeout-owner-started.json')
    assert not (full.ROOT/'publication.json').exists(), 'Existing publication must be inspected, not overwritten'
    full.save(full.ROOT/'closeout-original-status.json', full.read(full.ROOT/'status.json'))
    full.save(full.ROOT/'closeout-validation.json', evidence)
    subprocess.run(['git','archive','--format=tar.gz','--output='+str(full.ROOT/'closeout-source.tar.gz'),'HEAD'],check=True)
    full.phase(full.ROOT,'publishing_with_judging_exclusions')
    full.publish(exclusions=evidence)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['validate','publish'])
    args = parser.parse_args()
    print(json.dumps(validate(),indent=2)) if args.action == 'validate' else publish()
