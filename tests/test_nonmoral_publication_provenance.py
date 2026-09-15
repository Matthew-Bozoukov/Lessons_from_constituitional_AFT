# ABOUTME: Checks accepted-only model attribution across mixed generation recipes and local recovery.
# ABOUTME: Uses frozen local fixture configs and makes no provider or publication calls.
import copy
import json
from pathlib import Path

import pytest

from scratch.nonmoral.broader_data import conversation_sha256, file_sha256, publication_model_provenance


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def setup(tmp_path, name='batch01', model='anthropic/claude-sonnet-5'):
    batch = tmp_path/name
    cfg = {'models': {k: {'model': model} for k in ('source', 'author', 'reviewer')},
           'stages': [{'name': k, 'model': k, 'save': {out: out}}
                      for k, out in [('source', 'user'), ('author', 'response'), ('reviewer', 'quality_decision')]]}
    for phase in ('sources', 'answers', 'review'):
        write(batch/phase/'status.json', {'config': cfg, 'config_sha256': name+'-config'})
    row = {'scenario_id': name+'_001', 'user': 'Choose.', 'reasoning': 'A fits.', 'response': 'A.',
           'quality_decision': 'accept', 'quality_issues': '', 'choice_summary': 'A'}
    info = {'review': str(batch/'answer_review.json'), 'dataset': str(batch/'answers/dataset.jsonl')}
    save_row(batch, row, info)
    return batch, cfg, row, info


def save_row(batch, row, info):
    write(batch/'answer_review.json', {'dispositions': {row['scenario_id']: {'decision': 'accept'}}})
    write(Path(info['dataset']), row)


def test_counts_actual_mixed_models_only_accepted(tmp_path):
    _, _, sonnet, a = setup(tmp_path)
    _, _, opus, b = setup(tmp_path, 'batch02', 'anthropic/claude-opus-5')
    setup(tmp_path, 'failed_unaccepted', 'ignored-model')
    report = publication_model_provenance([sonnet, opus], [a, b])
    assert report['accepted'] == 2
    assert report['model_counts']['author'] == {'anthropic/claude-opus-5': 1, 'anthropic/claude-sonnet-5': 1}
    assert report['locally_corrected'] == 0


@pytest.mark.parametrize('changed', [True, False])
def test_recovery_uses_original_author_and_actual_review(tmp_path, changed):
    parent, oldcfg, old, oldinfo = setup(tmp_path)
    batch, _, _, info = setup(tmp_path, 'batch08', 'anthropic/claude-opus-5')
    info['final_phase'] = 'review'
    row = copy.deepcopy(old)
    if changed:
        row['reasoning'] = 'A fits well.'
    row['recovery_provenance'] = dict(
        kind='local_derivative_not_model_generation', parent_batch=str(parent),
        original_author_snapshot=oldinfo['dataset'], original_author_snapshot_sha256=file_sha256(oldinfo['dataset']),
        original_conversation_sha256=conversation_sha256(old), revised_conversation_sha256=conversation_sha256(row),
        original_author_models=oldcfg['models'], original_source_models=oldcfg['models'],
        original_author_config_sha256='batch01-config', original_source_config_sha256='batch01-config',
        reviewer='Local reader', recovery_review='recovery.json', recovery_review_sha256='review-hash',
        replacements=[{'field': 'reasoning', 'old': 'fits', 'new': 'fits well', 'occurrences': 1}] if changed else [],
        reused_exact_model_review=not changed,
        prior_model_review={k: old[k] for k in ('quality_decision', 'quality_issues', 'choice_summary')})
    save_row(batch, row, info)
    report = publication_model_provenance([row], [info])
    assert report['model_counts']['author'] == {'anthropic/claude-sonnet-5': 1}
    assert report['model_counts']['source'] == {'anthropic/claude-sonnet-5': 1}
    assert report['model_counts']['model_review'] == {('anthropic/claude-opus-5' if changed else 'anthropic/claude-sonnet-5'): 1}
    assert report['locally_corrected'] == int(changed)
    assert report['reused_model_reviews'] == int(not changed)
    row['recovery_provenance']['original_author_config_sha256'] = 'stale'
    save_row(batch, row, info)
    with pytest.raises(ValueError, match='Stale original author config'):
        publication_model_provenance([row], [info])


def test_missing_stage_never_falls_back_to_current_model(tmp_path):
    batch, cfg, row, info = setup(tmp_path)
    cfg['stages'] = []
    write(batch/'sources/status.json', {'config': cfg, 'config_sha256': 'hash'})
    with pytest.raises(ValueError, match='Ambiguous model attribution'):
        publication_model_provenance([row], [info])
