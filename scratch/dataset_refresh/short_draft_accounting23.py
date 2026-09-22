# ABOUTME: Freezes accounting for the explicitly authorized 23 short-draft proposals.
# ABOUTME: Reads the authoritative ledger under its lock and copies scoped raw receipts without model calls.
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from filelock import FileLock

base = Path(__file__).resolve().parents[2]
quality = base / 'output/2026-09-15_dataset_refresh_quality_screen'
budget = base / 'output/2026-09-14_dataset_refresh/budget'
old = json.loads((quality / 'short_draft_authorized_batch_01_completed.json').read_text())
rows = old['rows']
for cid in ['t7_070_v0', 't7_071_v0', 't7_073_v0', 't7_075_v0', 't7_079_v0']:
    folder = base / 'output/2026-09-15_dataset_refresh_diverse/da-lowstakes-refresh/records' / cid
    proposal = folder / 'short_draft_candidate_200.json'
    manifest = folder / 'short_draft_recovery_200.json'
    data = json.loads(proposal.read_text())
    meta = json.loads(manifest.read_text())
    rows.append(dict(phase='diverse', candidate_id=cid, proposal_path=str(proposal), proposal_sha256=hashlib.sha256(proposal.read_bytes()).hexdigest(), status=data['status'], error=data.get('error'), original_failed_source_sha256=meta['binding']['source_result_sha256'], source_review_eligible=meta['binding']['source_review']['eligible'], manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()))
keys = {(str(Path(row['proposal_path']).parents[3]), row['candidate_id']) for row in rows}
with FileLock(str(budget / 'spend.lock')):
    ledger = json.loads((budget / 'spend.json').read_text())
entries = [e for e in ledger if (e.get('run_root'), e.get('candidate_id')) in keys and e.get('stage') in ['short_draft_rewrite_200', 'grounding_200', 'review_200']]
assert len(rows) == 23
assert len({(e['run_root'], e['candidate_id'], e['stage']) for e in entries}) == len(entries)
assert all(e['status'] == 'settled' for e in entries)
audit = quality / 'short_draft_authorized_all23_audit'
audit.mkdir(exist_ok=True)
def save(path, data):
    raw = (json.dumps(data, ensure_ascii=False, indent=2)+'\n').encode()
    path.write_bytes(raw)
    path.with_suffix('.sha256').write_text(hashlib.sha256(raw).hexdigest()+'\n')
with (audit / 'raw_calls.jsonl').open('w', encoding='utf-8', newline='\n') as handle:
    for entry in entries:
        path = budget / 'raw_calls' / f"{entry['call_id']:06d}.json"
        raw = path.read_bytes()
        handle.write(json.dumps(dict(call_id=entry['call_id'], original_path=str(path), original_sha256=hashlib.sha256(raw).hexdigest(), raw=json.loads(raw)), ensure_ascii=False)+'\n')
save(audit / 'ledger_slice.json', dict(authoritative_ledger=str(budget / 'spend.json'), entries=entries))
report = dict(ABOUTME=['Completed all23 explicitly authorized short-draft recovery proposals.', 'Automatic statuses are not independent adoption decisions; failures and holds remain preserved.'], completed_at_utc=datetime.now(timezone.utc).isoformat(), authorization_sequence=['Initial18 explicitly authorized by root:17diverse then qualifiedt7_006.', 'Additional3 explicitly authorized:diverset7_070,t7_071,t7_073.', 'Final2 explicitly authorized:diverset7_075,t7_079. No further calls.'], counts=dict(Counter(row['status'] for row in rows)), physical_calls=len(entries), actual_batch_cost_usd=sum(e['charged_or_reserved_usd'] for e in entries), shared_exposure_snapshot_usd=sum(e['charged_or_reserved_usd'] for e in ledger), raw_calls_jsonl_sha256=hashlib.sha256((audit/'raw_calls.jsonl').read_bytes()).hexdigest(), rows=rows)
save(quality / 'short_draft_authorized_all23_completed.json', report)
print(json.dumps({k:v for k,v in report.items() if k not in ['rows','ABOUTME','authorization_sequence']}))
