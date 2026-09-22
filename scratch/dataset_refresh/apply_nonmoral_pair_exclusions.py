# ABOUTME: Applies five explicitly authorized nonmoral scenario-redundancy exclusions.
# ABOUTME: Binds both pair members and preserves the original review confidence and substantive distinctions.
import json
from pathlib import Path
from filelock import FileLock
import run as runtime

base = Path(__file__).resolve().parents[2]
root = base / 'output/2026-09-15_dataset_refresh_sonnet_qualified'
arm = root / 'nonmoral-advice'
source = arm / 'independent_provisional_pair_review.json'
review = runtime.load_checkpoint(source)
choices = {
    't1_037_v0': 't1_009_v0',
    't2_044_v0': 't2_016_v0',
    't5_046_v0': 't5_018_v0',
    't8_035_v0': 't1_049_v0',
    't6_033_v0': 't6_005_v0',
}
plans = []
with FileLock(str(root / 'execution.lock'), timeout=1):
    for excluded, kept in choices.items():
        pairs = [pair for pair in review['pairs']
                 if {Path(endpoint['origin_path']).parent.name for endpoint in pair['endpoints'].values()}
                 == {excluded, kept}]
        assert len(pairs) == 1
        pair = pairs[0]
        assert pair['decision'] == 'near_duplicate_scenario'
        for endpoint in pair['endpoints'].values():
            path = Path(endpoint['origin_path'])
            assert path.is_relative_to(arm)
            assert runtime.digest(path.read_bytes()) == endpoint['origin_result_sha256']
            assert runtime.load_result(path)['status'] == 'accepted'
        excluded_path = arm / 'records' / excluded / 'result.json'
        kept_path = arm / 'records' / kept / 'result.json'
        plans.append((excluded_path, dict(
            result_sha256=runtime.digest(excluded_path.read_bytes()),
            reason='Root-authorized conservative specific-decision scenario deduplication: ' + pair['reason'],
            independent_disposition='SCENARIO_DEDUPLICATION', selection_hold=True,
            excluded_candidate_id=excluded, retained_candidate_id=kept,
            retained_result_path=str(kept_path), retained_result_sha256=runtime.digest(kept_path.read_bytes()),
            audit_source=str(source), audit_sha256=runtime.digest(source.read_bytes()),
            review_confidence=pair['confidence'],
            strongest_distinction_or_overlap_caveat=pair['strongest_distinction_or_overlap_caveat'],
            evidence_quotes=pair['evidence_quotes'], mechanism=pair['mechanism'],
            original_pair_rank=pair['rank'], candidate_cosine_similarity=pair['cosine_similarity'],
            criterion='Same specific advice decision; shared domain/family or embedding score alone is insufficient.',
            nuance='Original confidence is retained. This release decision does not claim identical wording or absence of meaningful contextual differences.',
            authorization='Root explicitly selected these five excluded/retained pairs; the other five reviewed shared-family pairs remain unchanged.'
        )))
    for result, note in plans:
        exclusion = result.parent / 'independent_exclusion.json'
        assert not exclusion.exists()
        runtime.save_checkpoint(exclusion, note)
        assert runtime.digest(result.read_bytes()) == note['result_sha256']
        assert runtime.load_result(result)['status'] == 'rejected'
        assert runtime.load_result(Path(note['retained_result_path']))['status'] == 'accepted'
out = base / 'output/2026-09-15_dataset_refresh_quality_screen/nonmoral_pair_exclusions_applied.json'
runtime.save_checkpoint(out, dict(count=len(plans), terminal_bytes_unchanged=True,
                                  retained_rows_still_effective_accepted=True,
                                  rows=[note for _, note in plans]))
print(json.dumps(dict(count=len(plans), choices=choices, report=str(out))))
