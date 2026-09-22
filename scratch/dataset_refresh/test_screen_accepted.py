# ABOUTME: Offline tests for read-only accepted-row triage and deliberately nonbinding candidate flags.
from scratch.dataset_refresh import screen_accepted as mod


def test_flags_literal_corruption_process_language_and_only_new_quantities():
    record = {'system': 'Give advice.', 'user': 'Wait two days.',
              'reasoning': 'The previous answer had a flaw.\ufffd',
              'response': 'Wait two days. You could propose five minutes instead.'}
    flags = mod.screen_record(record)
    assert {f['kind'] for f in flags} == {'replacement_character', 'editing_process_language_candidate', 'quantity_not_verbatim_in_prompt'}
    quantity = next(f for f in flags if f['kind'] == 'quantity_not_verbatim_in_prompt')
    assert [e['quote'] for e in quantity['evidence']] == ['five minutes']
    assert 'legitimate proposed' in quantity['interpretation']


def test_repetition_counts_distinct_rows_not_repeats_in_one_answer():
    sentence = 'One two three four five six seven eight nine ten.'
    single = [{'id': 'a', 'record': {'response': sentence * 10}}]
    assert mod.repeated_final_phrases(single)['matching_ngrams'] == 0
    rows = [{'id': str(i), 'record': {'response': sentence}} for i in range(4)]
    assert mod.repeated_final_phrases(rows)['top100'][0]['row_count'] == 4


def test_only_accepted_and_unexcluded_rows_with_valid_receipts_are_screened(tmp_path):
    root = tmp_path / 'run'
    record = {'system': 'Advice.', 'user': 'Choose.', 'reasoning': 'Weigh.', 'response': 'Choose.'}
    for cid, status in [('a', 'accepted'), ('b', 'rejected'), ('c', 'failed')]:
        mod.runtime.save_checkpoint(root / 'arm' / 'records' / cid / 'result.json',
                                    {'candidate_id': cid, 'status': status, 'record': record})
    before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    report = mod.audit(root)
    assert report['accepted_rows'] == 1 and report['flagged_rows'] == 0
    assert {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()} == before
