# ABOUTME: Offline checks of the immutable source gate before paid broader response generation.
# ABOUTME: Reject stale or incomplete reviews and preserve accepted original user requests.
import json

import pytest

from scratch.nonmoral.broader_data import source_review_inputs
from scratch.nonmoral.pilot import file_sha256


def fixture_files(tmp_path):
    rows = [{'scenario_id':'a','user':'Original complete request.'},
            {'scenario_id':'b','user':'Incomplete request.'}]
    source=tmp_path/'source.jsonl'
    source.write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
    review=tmp_path/'review.json'
    data={'source_sha256':file_sha256(source),'reviewer':'local reviewer',
          'dispositions':{'a':{'decision':'accept','reason':'Complete.'},
                          'b':{'decision':'reject','reason':'Missing source.'}}}
    review.write_text(json.dumps(data),encoding='utf-8')
    return source,review,data,rows


def test_selects_only_accepted_original_bytes(tmp_path):
    source,review,_,rows=fixture_files(tmp_path)
    selected,_=source_review_inputs(review,source)
    assert selected==rows[:1]


def test_rejects_changed_source(tmp_path):
    source,review,_,_=fixture_files(tmp_path)
    source.write_text(source.read_text().replace('Original','Changed'),encoding='utf-8')
    with pytest.raises(ValueError,match='stale'):
        source_review_inputs(review,source)


@pytest.mark.parametrize('field,value',[('decision','maybe'),('reason','')])
def test_rejects_invalid_disposition(tmp_path,field,value):
    source,review,data,_=fixture_files(tmp_path)
    data['dispositions']['a'][field]=value
    review.write_text(json.dumps(data),encoding='utf-8')
    with pytest.raises(ValueError,match='dispositions'):
        source_review_inputs(review,source)


def test_requires_complete_review(tmp_path):
    source,review,data,_=fixture_files(tmp_path)
    del data['dispositions']['b']
    review.write_text(json.dumps(data),encoding='utf-8')
    with pytest.raises(ValueError,match='Every original'):
        source_review_inputs(review,source)


def test_prior_examples_opt_out_skips_history_and_preserves_legacy_default(tmp_path, monkeypatch):
    from omegaconf import OmegaConf
    from scratch.nonmoral import broader_data as data

    monkeypatch.setattr(data, 'ROOT', tmp_path)
    phase = tmp_path/'production/batch01/sources'
    run = phase/'run'
    run.mkdir(parents=True)
    history = [{'domain': 'craft', 'variation': 'Task 1. Craft plan',
                'user': 'Old multi-constraint request, including rejected candidates.'}]
    (run/'dataset.jsonl').write_text(json.dumps(history[0])+'\n', encoding='utf-8')
    status = phase/'status.json'
    status.write_text(json.dumps({'run_dir': str(run)}), encoding='utf-8')
    assert data.source_history() == history
    assert data.source_history(True) == history
    # Opt-out must avoid reading history at all, even if its status is malformed.
    status.write_text('not JSON', encoding='utf-8')
    assert data.source_history(False) == []
    cfg = OmegaConf.load('configs/data/synth/nonmoral-broader.yaml')
    assert cfg.production.include_prior_examples is False
    assert '{prior_examples}' not in cfg.production.source_stage.prompts.user
