# ABOUTME: Applies four authorized presentation holds for accurate visible-system references.
# ABOUTME: Keeps these policy holds distinct from hidden generation leakage or factual defects.
import json
from pathlib import Path
from filelock import FileLock
import run as runtime

base = Path(__file__).resolve().parents[2]
root = base / 'output/2026-09-15_dataset_refresh_diverse'
quality = base / 'output/2026-09-15_dataset_refresh_quality_screen'
source = quality / 'editing_leak_final_adjudication.json'
rows = [row for row in runtime.load_checkpoint(source)['rows']
        if row['category'] == 'visible_system_instruction_reflection']
assert len(rows) == 4
actions = []
with FileLock(str(root / 'execution.lock'), timeout=1):
    for row in rows:
        result = Path(row['result_path'])
        assert result.is_relative_to(root)
        assert runtime.digest(result.read_bytes()) == row['result_sha256']
        assert runtime.load_checkpoint(result)['status'] == 'accepted'
    for row in rows:
        result = Path(row['result_path'])
        path = result.parent / 'independent_exclusion.json'
        reason = ('PRESENTATION_HOLD under the existing release convention: the reasoning '
                  'explicitly discusses the system prompt. It accurately reflects the real '
                  'training-visible input and is not a hidden generation leak, factual error, '
                  'or content-safety defect. Withheld conservatively under the current '
                  'presentation policy; reconsider only under a future explicitly changed policy.')
        note = dict(result_sha256=row['result_sha256'], reason=reason,
                    independent_disposition='PRESENTATION_HOLD', selection_hold=True,
                    hidden_generation_or_editing_leak=False,
                    factual_error=False, content_safety_defect=False,
                    audit_source=str(source), audit_sha256=runtime.digest(source.read_bytes()),
                    evidence=row['hits'], authorization='Root explicitly authorized these four presentation holds.')
        action = 'created'
        if path.exists():
            prior = runtime.load_checkpoint(path)
            assert prior['result_sha256'] == row['result_sha256']
            if prior.get('independent_disposition') == 'PRESENTATION_HOLD':
                action = 'already_applied'
            else:
                path = result.parent / 'independent_exclusion_presentation_followup.json'
                runtime.save_checkpoint(path, note)
                action = 'preserved_prior_plus_supplement'
        else:
            runtime.save_checkpoint(path, note)
        assert runtime.load_result(result)['status'] == 'rejected'
        assert runtime.digest(result.read_bytes()) == row['result_sha256']
        actions.append(dict(candidate_id=row['candidate_id'], result_sha256=row['result_sha256'],
                            action=action, exclusion_path=str(path)))
runtime.save_checkpoint(quality / 'visible_system_presentation_holds_applied.json',
                        dict(count=len(actions), rows=actions, terminal_bytes_unchanged=True,
                             classification='PRESENTATION_HOLD; not hidden leak or factual defect'))
print(json.dumps(dict(count=len(actions), actions=actions)))
