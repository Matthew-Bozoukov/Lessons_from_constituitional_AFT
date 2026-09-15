# ABOUTME: Checks source/review lineage and exact replay preservation in the broader mixture.
# ABOUTME: Uses tiny text fixtures; no network, model calls or GPU rental.
import json

import pytest

from scratch.nonmoral import broader_data as data


def test_selection_is_stable_balanced_and_deduplicated():
    rows = [{'scenario_id':str(i),'domain':'a' if i<8 else 'b','user':f'Task {i}'} for i in range(12)]
    rows.append({'scenario_id':'duplicate','domain':'a','user':' Task   0 '})
    assert data.select_balanced(rows,10)==data.select_balanced(list(reversed(rows)),10)
    chosen=data.select_balanced(rows,10)
    assert len({' '.join(r['user'].split()) for r in chosen})==10
    assert sum(r['domain']=='b' for r in chosen)==4


def reviewed_batch(tmp_path):
    batch=tmp_path/'production/batch01'
    for phase in ['sources','answers']:
        (batch/phase).mkdir(parents=True)
    source={'scenario_id':'a','domain':'craft','user':'Original task'}
    answer={**source,'reasoning':'Compare feasible choices.','response':'Full artifact.'}
    for phase,row in [('sources',source),('answers',answer)]:
        p=batch/phase/'dataset.jsonl'
        p.write_text(json.dumps(row)+'\n')
        (batch/phase/'status.json').write_text(json.dumps({'run_dir':str(p.parent)}))
    for name,phase,field in [('source','sources','source_sha256'),('answer','answers','dataset_sha256')]:
        review={'reviewer':'test',field:data.file_sha256(batch/phase/'dataset.jsonl'),
                'dispositions':{'a':{'decision':'accept','reason':'Complete'}}}
        (batch/f'{name}_review.json').write_text(json.dumps(review))
    return batch,answer


def test_changed_source_cannot_enter_accepted_pool(tmp_path):
    batch,answer=reviewed_batch(tmp_path)
    assert data.accepted_production(tmp_path)[0]==[answer]
    answer['user']='Changed task'
    p=batch/'answers/dataset.jsonl';p.write_text(json.dumps(answer)+'\n')
    review_path=batch/'answer_review.json';review=json.loads(review_path.read_text())
    review['dataset_sha256']=data.file_sha256(p);review_path.write_text(json.dumps(review))
    with pytest.raises(ValueError,match='Changed or unapproved source'):
        data.accepted_production(tmp_path)


def test_shortfall_cannot_create_training_mixture(tmp_path):
    reviewed_batch(tmp_path)
    with pytest.raises(ValueError,match='Only 1 distinct'):
        data.assemble(tmp_path/'not-needed.jsonl',tmp_path)
    assert not (tmp_path/'mixture').exists()


def test_replay_lines_and_positions_survive_assembly(tmp_path,monkeypatch):
    rows=[{'scenario_id':str(i),'domain':'craft','user':f'Task {i}',
           'reasoning':f'Reason {i}','response':f'Artifact {i}'} for i in range(684)]
    monkeypatch.setattr(data,'accepted_production',lambda root:(rows,[]))
    replay=tmp_path/'historical.jsonl'
    lines=[]
    for i in range(9968):
        source='nonmoral_deliberation' if i%14==0 and i//14<684 else 'replay'
        # Deliberately unusual JSON spacing/line endings must survive unchanged.
        lines.append((json.dumps({'source':source,'text':f'old {i}'},indent=None,separators=(', ', ': '))+'\r\n').encode())
    replay.write_bytes(b''.join(lines))
    actual_hash=data.file_sha256
    monkeypatch.setattr(data,'file_sha256',lambda p:'0517ef85d288f14e42bc77f371d3e2e48866879b5824984feaf4a7451ce60561' if p==replay else actual_hash(p))
    data.assemble(replay,tmp_path)
    mixed=(tmp_path/'mixture/mixture.jsonl').read_bytes().splitlines(keepends=True)
    assert len(mixed)==len(lines)
    for old,new in zip(lines,mixed):
        if json.loads(old)['source']=='replay':
            assert old==new
        else:
            row=json.loads(new);sid=row['scenario_id']
            assert f'<think>\nReason {sid}\n</think>\n\nArtifact {sid}' in row['text']
