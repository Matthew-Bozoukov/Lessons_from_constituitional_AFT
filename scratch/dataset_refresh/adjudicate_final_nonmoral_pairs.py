# ABOUTME: Records the root adjudication of five final nonmoral semantic pair findings.
# ABOUTME: Applies three exact-hash duplicate holds while preserving two explicitly retained constraint variations.
import json
from collections import Counter
from pathlib import Path
from filelock import FileLock
import run as runtime

base = Path(__file__).resolve().parents[2]
root = base / 'output/2026-09-15_dataset_refresh_sonnet_qualified'
arm = root / 'nonmoral-advice'
folder = base / 'output/2026-09-15_dataset_refresh_quality_screen/final_partial_nonmoral'
source = folder / 'independent_pair_review.json'
review = runtime.load_checkpoint(source)
choices = {
    5: (None, None, 'Retain both: t1_039 contains an independent cut-versus-keep trivia decision as well as spoiler placement, while t2_077 focuses on consistent before/after placement. The source decisions justify coexistence despite the strong shared family.'),
    6: (None, None, 'Retain both: t1_058 has a concrete one-page physical reference constraint, while t1_002 is a scrollable Help page. These create different space and retrieval decisions despite strongly repeated shortcut-exception content.'),
    7: ('t5_070_v0', 't5_042_v0', 'Withhold t5_070 and retain t5_042: both ask whether drag-modifier shortcuts need an action-based section outside conventional standalone shortcut documentation. Menu placement versus table wording does not sufficiently change the central decision.'),
    11: ('t1_077_v0', 't1_049_v0', 'Withhold t1_077 and retain t1_049: both six-month fourteen-chapter memoirs ask for selective time/place/companion orientation to prevent confused readers without uniformly slowing chapter openings. Different initial draft lengths remain acknowledged.'),
    12: ('t7_066_v0', 't7_010_v0', 'Withhold t7_066 and retain t7_010: both short poetry endnotes ask how to combine technical names and plain glosses for intentionally altered ghazal/sonnet forms. Particular formal departures change accurate wording but not this naming decision.'),
}
rows = []
with FileLock(str(root / 'execution.lock'), timeout=1):
    for rank, (excluded, retained, reason) in choices.items():
        pair = next(pair for pair in review['pairs'] if pair['rank'] == rank)
        assert pair['decision'] == 'near_duplicate_scenario'
        for endpoint in pair['endpoints'].values():
            path = Path(endpoint['origin_path'])
            assert path.is_relative_to(arm)
            assert runtime.digest(path.read_bytes()) == endpoint['origin_result_sha256']
            assert runtime.load_result(path)['status'] == 'accepted'
        rows.append(dict(rank=rank, disposition='SCENARIO_DEDUPLICATION' if excluded else 'RETAIN_BOTH_CONSTRAINT_VARIATION',
                         excluded_candidate_id=excluded, retained_candidate_id=retained,
                         reason=reason, endpoints=pair['endpoints'],
                         independent_proposal=pair['decision'], independent_confidence=pair['confidence'],
                         independent_counterpoint=pair['strongest_distinction_or_overlap_caveat']))
    for row in rows:
        if not row['excluded_candidate_id']:
            continue
        result = arm / 'records' / row['excluded_candidate_id'] / 'result.json'
        retained = arm / 'records' / row['retained_candidate_id'] / 'result.json'
        exclusion = result.parent / 'independent_exclusion.json'
        assert not exclusion.exists()
        note = dict(result_sha256=runtime.digest(result.read_bytes()), reason=row['reason'],
                    independent_disposition='SCENARIO_DEDUPLICATION', selection_hold=True,
                    retained_result_path=str(retained), retained_result_sha256=runtime.digest(retained.read_bytes()),
                    audit_source=str(source), audit_sha256=runtime.digest(source.read_bytes()),
                    root_adjudication=row, authorization='Root reviewed all five pairs and explicitly authorized only these three holds.')
        runtime.save_checkpoint(exclusion, note)
        assert runtime.load_result(result)['status'] == 'rejected'
        assert runtime.load_result(retained)['status'] == 'accepted'
        assert runtime.digest(result.read_bytes()) == note['result_sha256']
    effective = [runtime.load_result(path) for path in (arm/'records').glob('*/result.json')]
    accepted = [row for row in effective if row['status'] == 'accepted']
    assert len(accepted) == 631
report = dict(ABOUTME=['Root adjudication of all five final proposed near-duplicate nonmoral pairs.', 'Three exact-hash scenario deduplication holds; two pairs retained for materially different decisions or constraints.'],
              source_review=str(source), source_review_sha256=runtime.digest(source.read_bytes()),
              input_sha256=review['input_sha256'], decisions=rows,
              applied_holds=3, retained_pairs=2, effective_accepted_after=631,
              effective_trait_counts=dict(sorted(Counter(row['trait_id'] for row in accepted).items())),
              terminal_bytes_unchanged=True, further_mutations_closed=True,
              interpretation='The 634-row audit remains its original snapshot. These three documented overlays yield631 prospective rows; no audit input was rewritten.')
runtime.save_checkpoint(folder/'root_pair_adjudication.json', report)
print(json.dumps({key:value for key,value in report.items() if key not in ['decisions','ABOUTME']}))
