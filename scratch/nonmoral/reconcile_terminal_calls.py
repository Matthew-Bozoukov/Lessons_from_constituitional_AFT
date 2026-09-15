# ABOUTME: Close proven terminal API reservations without reducing their conservative cost.
# ABOUTME: Requires a completed latest phase, no generation lock, and persisted exception evidence.
import argparse
import copy
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconcile(root, phase):
    root, phase = Path(root), Path(phase)
    if (root / 'generation.lock').exists():
        raise ValueError('Generation still owns the ledger')
    status_path = phase / 'status.json'
    status = json.loads(status_path.read_text())
    if status['status'] not in ('awaiting_local_source_review', 'awaiting_local_author_review', 'awaiting_local_answer_review'):
        raise ValueError('A completed phase is required')
    ledger = root / 'spend.json'
    entries = json.loads(ledger.read_text())
    if status['cumulative_calls'] != len(entries):
        raise ValueError('Phase is not the latest ledger snapshot')
    receipt_path = phase / 'terminal_charge_receipt.json'
    if receipt_path.exists():
        raise ValueError('Preserve the existing reconciliation receipt')
    receipt = dict(ledger_sha256_before=sha(ledger), phase_status_sha256=sha(status_path),
                   entries=[], accounting='Full terminal reservations retained; actual charges unknown. No retries or refunds assumed.')
    for index, entry in enumerate(entries):
        if entry['status'] != 'reserved':
            continue
        raw_path = root / 'raw_calls' / f'{index:05d}.json'
        raw = json.loads(raw_path.read_text(encoding='utf-8'))
        if raw['status'] != 'terminal_exception':
            raise ValueError(f'Call {index} lacks terminal exception evidence')
        request_hash = hashlib.sha256(json.dumps(raw['messages'], ensure_ascii=False).encode()).hexdigest()
        if request_hash != entry['request_sha256']:
            raise ValueError(f'Call {index} request hash differs')
        receipt['entries'].append(dict(entry_index=index, original_entry=copy.deepcopy(entry),
                                       raw_call_sha256=sha(raw_path), exception_type=raw['exception_type']))
        entry['status'] = 'retained_terminal_reservation'
        entry['terminal_evidence'] = str(receipt_path.relative_to(root))
    # All checks precede mutations. Receipt is written first so interruptions cannot
    # silently erase the original reservation record.
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    ledger.write_text(json.dumps(entries, indent=2) + '\n', encoding='utf-8')
    return dict(retained_calls=len(receipt['entries']), exposure_usd=sum(e['charged_or_reserved_usd'] for e in entries))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reconcile(args.root, args.phase_dir)))
