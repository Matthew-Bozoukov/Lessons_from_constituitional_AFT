# ABOUTME: Applies the 13 root-authorized exact-hash independent nonmoral holds or rejects.
# ABOUTME: Preserves terminal bytes and any earlier exclusion evidence under the existing execution lock.
import json
from pathlib import Path
from filelock import FileLock
import run as runtime

base = Path(__file__).resolve().parents[2]
root = base / 'output/2026-09-15_dataset_refresh_sonnet_qualified'
source = base / 'output/2026-09-15_dataset_refresh_quality_screen/nonmoral_billing_retry_independent_review.json'
rows = [row for row in json.loads(source.read_text(encoding='utf-8'))['rows'] if row['selection_hold']]
assert len(rows) == 13
actions = []
with FileLock(str(root / 'execution.lock'), timeout=1):
    for row in rows:
        result = Path(row['result_path'])
        assert runtime.digest(result.read_bytes()) == row['result_sha256'], row['candidate_id']
        assert runtime.load_checkpoint(result)['status'] == 'accepted'
    for row in rows:
        result = Path(row['result_path'])
        path = result.parent / 'independent_exclusion.json'
        note = dict(result_sha256=row['result_sha256'], reason=row['reason'], evidence=row['evidence'], audit_source=str(source), audit_sha256=runtime.digest(source.read_bytes()), selection_hold=True, independent_disposition=row['independent_disposition'], authorization='Root explicitly authorized all13 nonpass billing-resume exclusions.')
        action = 'created'
        if path.exists():
            prior = runtime.load_checkpoint(path)
            assert prior['result_sha256'] == row['result_sha256']
            path = result.parent / 'independent_exclusion_billing_followup.json'
            action = 'preserved_prior_plus_supplement'
        runtime.save_checkpoint(path, note)
        assert runtime.load_result(result)['status'] == 'rejected'
        assert runtime.digest(result.read_bytes()) == row['result_sha256']
        actions.append(dict(candidate_id=row['candidate_id'], action=action, result_sha256=row['result_sha256'], exclusion_path=str(path)))
runtime.save_checkpoint(source.parent / 'nonmoral_billing_exclusions_applied.json', dict(rows=actions, count=len(actions), terminal_bytes_unchanged=True))
print(json.dumps(dict(count=len(actions), actions=actions)))
