# ABOUTME: Validate exact repetition evidence and preserve distinct scoring states in HF browser indexes.
# ABOUTME: Exercise Unicode offset semantics and ensure ordinary recurring words do not count as loops.
import json
from src.eval.capabilities.swebench_mini.browser import repeated_span, diagnose, build_index


def test_literal_loop_unicode_and_reconstruction():
    unit = 'A paragraph with an emoji 🚀 and meaningful content. '*7 + '\n\n'
    prefix = 'Initial reasoning 🔬.\n\n'
    text = prefix + unit*10 + 'partial tail'
    span = repeated_span(text)
    assert span and span['repeats'] == 10
    assert span['start'] == len(prefix.encode('utf-16-le'))//2
    assert span['unit_length'] == len(unit.encode('utf-16-le'))//2
    assert repeated_span('A repeated phrase. '*500) is None
    assert repeated_span(unit*7) is None


def test_answer_content_is_scanned_and_gaps_are_explicit():
    text = ('A long repeated statement. '*12+'\n\n')*12
    t = {'messages': [{'role': 'assistant', 'extra': {'response': {'choices': [{'message': {'content': text}}]}}},
                      {'role': 'user', 'content': 'Tool call error: missing bash'}]}
    d = diagnose(t)
    assert d['loops'][0]['field'] == 'content'
    assert d['discarded_response_corrections'] == 1


def test_index_uses_final_valid_attempt_and_official_grading(tmp_path):
    def save(path, value):
        p = tmp_path/path; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(value))
    save('metadata/state.json', {'tasks': {
        'one': {'status': 'valid', 'attempts': [{'id': 'bad', 'valid': False}, {'id': 'good', 'valid': True, 'exit_status': 'Submitted'}]},
        'pending': {'status': 'pending', 'attempts': []}}})
    save('results/results.json', {'task_results': {'one': {'graded': True, 'resolved': False}}})
    save('rollouts/one/good/checkpoint.traj.json', {'messages': [{'role': 'exit', 'content': 'Submitted'}]})
    rows = build_index(tmp_path)['tasks']
    assert rows[0]['attempt'] == 'good' and rows[0]['resolved'] is False
    assert rows[1]['resolved'] is None and not rows[1]['graded']
