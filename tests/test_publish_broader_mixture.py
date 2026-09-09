# ABOUTME: Checks the broader-mixture publisher against replay-byte, selection and publication-contract failures.
# ABOUTME: Uses a complete-size synthetic fixture and local mocks; never downloads or uploads artifacts.
import json
from pathlib import Path

import pytest

from scratch.nonmoral import publish_broader_mixture as pub


@pytest.fixture
def assembled(tmp_path, monkeypatch):
    root = tmp_path/'corpus'
    out = root/'mixture'
    out.mkdir(parents=True)
    pool = [dict(scenario_id=f'example_{i}', domain=f'domain_{i%12}',
                 user=f'Choose a pattern for object {i}.', reasoning='Choose a stripe over a dot.',
                 response=f'Object {i}: stripe.') for i in range(684)]
    selected = pub.select_balanced(pool, 684)
    slots = {i*14 for i in range(684)}
    originals, mixed = [], []
    picked = iter(selected)
    for i in range(9968):
        old = dict(source='nonmoral_deliberation' if i in slots else 'table2',
                   text=f'replay text {i}')
        raw = (json.dumps(old, separators=(',', ':'))+'\r\n').encode()
        originals.append(raw)
        if i not in slots:
            mixed.append(raw)
        else:
            row = next(picked)
            new = dict(source='nonmoral_broader', scenario_id=row['scenario_id'], domain=row['domain'],
                       text=pub.render([dict(role='user', content=row['user']),
                                        dict(role='assistant', content=row['response'],
                                             reasoning_content=row['reasoning'])]))
            mixed.append((json.dumps(new, ensure_ascii=False)+'\n').encode())
    replay = tmp_path/'historical.jsonl'
    replay.write_bytes(b''.join(originals))
    monkeypatch.setattr(pub, 'REPLAY_SHA256', pub.sha(replay))
    monkeypatch.setattr(pub, 'accepted_production', lambda _: (pool, []))
    monkeypatch.setattr(pub, 'publication_model_provenance', lambda rows, _: dict(rows={
        r['scenario_id']:dict(conversation_sha256=pub.conversation_sha256(r),
            source={'model':'fixture-model'}, author={'model':'fixture-model'},
            model_review={'model':'fixture-model'}, local_correction=None) for r in rows}))
    (out/'mixture.jsonl').write_bytes(b''.join(mixed))
    (out/'selected_examples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected))
    manifest = dict(target=684, replay_rows=9284, replay_source_sha256=pub.sha(replay),
                    mixture_sha256=pub.sha(out/'mixture.jsonl'), accepted_pool=684,
                    selected_by_domain=dict(pub.Counter(r['domain'] for r in selected)),
                    selection='fixture balanced selection', reviews=[])
    (out/'manifest.json').write_text(json.dumps(manifest))
    return root, replay


def test_exact_counts_and_replay_reformat_rejected(assembled):
    root, replay = assembled
    rows, selected, _, report = pub.validate(root, replay)
    assert len(rows) == 9968 and len(selected) == 684
    assert report['replay_rows_checked_byte_for_byte'] == 9284
    p = root/'mixture/mixture.jsonl'
    lines = p.read_bytes().splitlines(keepends=True)
    # Semantic JSON equality is insufficient: the replay contract preserves raw lines.
    lines[1] = (json.dumps(json.loads(lines[1]))+'\n').encode()
    p.write_bytes(b''.join(lines))
    m = root/'mixture/manifest.json'
    manifest = json.loads(m.read_text())
    manifest['mixture_sha256'] = pub.sha(p)
    m.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Replay bytes changed at mixture index 1'):
        pub.validate(root, replay)


def test_selection_and_provenance_drift_rejected(assembled, monkeypatch):
    root, replay = assembled
    p = root/'mixture/selected_examples.jsonl'
    rows = pub.read_jsonl(p)
    rows[0], rows[1] = rows[1], rows[0]
    p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError, match='Selected examples differ'):
        pub.validate(root, replay)
    m = root/'mixture/manifest.json'
    manifest = json.loads(m.read_text())
    manifest['reviews'] = [{'review':'changed'}]
    m.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Review provenance changed'):
        pub.validate(root, replay)


def test_offline_snapshot_has_mixture_default_and_no_false_approval(assembled, tmp_path, monkeypatch):
    import yaml
    import src.infra.huggingface as hf
    root, replay = assembled
    monkeypatch.setattr(hf, 'hf_api', lambda: pytest.fail('Offline prepare attempted HF access'))
    dest, name, _, front = pub.prepare(root, replay,
        Path('configs/data/synth/nonmoral-broader.yaml'), Path('configs/train/sft.yaml'),
        output=tmp_path/'published_shape')
    assert name.endswith('-nonmoral-broader-7-mix')
    assert (dest/'mixture.jsonl').read_bytes() == (root/'mixture/mixture.jsonl').read_bytes()
    assert front['configs'][0] == dict(config_name='default', data_files='mixture.jsonl', default=True)
    readme = (dest/'README.md').read_text(encoding='utf-8')
    assert yaml.safe_load(readme.split('---')[1])['configs'] == front['configs']
    manifest = json.loads((dest/'manifest.json').read_text())
    assert manifest['token_mask_checks']['status'] == 'pending'
    assert manifest['approved_for_training'] is False
    assert not (dest/'dataset.jsonl').exists()  # No accidental competing training default.


def test_cached_tokenizer_reports_overlength_without_cutting_row(tmp_path, monkeypatch):
    from huggingface_hub.constants import HF_HUB_CACHE
    tokenizer = Path(HF_HUB_CACHE)/'models--Qwen--Qwen3.6-27B'/'snapshots'/pub.BASE_REVISION
    if not (tokenizer/'tokenizer.json').exists():
        pytest.skip('Pinned tokenizer absent; this optional integration test never downloads')
    config = tmp_path/'tiny_ceiling.yaml'
    config.write_text('train:\n  max_seq_len: 16\n')
    row = dict(source='nonmoral_broader', scenario_id='long_example', text=pub.render([
        dict(role='user', content='Choose a pattern.'),
        dict(role='assistant', reasoning_content='Compare stripes and dots. '*30,
             content='Use stripes. '*30)]))
    original = row['text']
    result = pub.token_mask_checks([row], tokenizer, config)
    assert result['status'] == 'failed'
    assert result['overlength_rows'][0]['scenario_id'] == 'long_example'
    assert result['overlength_rows'][0]['tokens'] > 16
    assert result['full_synthetic_masks_checked'] == 1
    assert row['text'] == original
    # Publishing the failed preflight is blocked before any HF method is called.
    dest = tmp_path/'failed_snapshot'
    dest.mkdir()
    (dest/'manifest.json').write_text(json.dumps({'token_mask_checks':result}))
    monkeypatch.setattr(pub, 'prepare', lambda *args: (dest, 'unused', {}, {}))
    monkeypatch.setattr('sys.argv', ['publisher', '--replay-mixture', 'unused', '--publish'])
    import src.infra.huggingface as hf
    monkeypatch.setattr(hf, 'hf_api', lambda: pytest.fail('Failed validation attempted HF access'))
    with pytest.raises(ValueError, match='Token length validation failed'):
        pub.main()


def test_selected_provenance_counts_recovered_sonnet_and_opus_without_unselected(monkeypatch):
    rows = [dict(scenario_id=str(i), user='Task'+str(i), reasoning='Reason', response='Answer') for i in range(3)]
    records = {}
    for row, model in zip(rows, ['anthropic/claude-sonnet-5', 'anthropic/claude-opus-4.8', 'unselected-model']):
        records[row['scenario_id']] = dict(conversation_sha256=pub.conversation_sha256(row),
            source={'model':model}, author={'model':model}, model_review={'model':'anthropic/claude-sonnet-5'},
            local_correction={'changed':True, 'reused_exact_model_review':False, 'replacements':[{'old':'18','new':'19'}]}
                if row is rows[0] else None)
    monkeypatch.setattr(pub, 'accepted_production', lambda _: (rows, ['frozen-batches']))
    def shared(accepted, batches):
        assert accepted == rows and batches == ['frozen-batches']
        return {'rows':records}
    monkeypatch.setattr(pub, 'publication_model_provenance', shared)
    result = pub.selected_model_provenance('unused', rows[:2])
    assert result['accepted'] == 2 and result['locally_corrected'] == 1
    assert result['model_counts']['author'] == {'anthropic/claude-sonnet-5':1, 'anthropic/claude-opus-4.8':1}
    assert result['model_counts']['model_review'] == {'anthropic/claude-sonnet-5':2}
    assert result['rows']['0']['local_correction']['replacements'] == [{'old':'18','new':'19'}]
    with pytest.raises(ValueError, match='differs from model provenance'):
        pub.selected_model_provenance('unused', [{**rows[0], 'response':'changed'}])


@pytest.mark.parametrize('missing_real_config', [False, True])
def test_recovery_audit_preserves_true_configs_and_requires_real_frozen_config(tmp_path, monkeypatch, missing_real_config):
    from omegaconf import OmegaConf
    batch=tmp_path/'batch08'
    cfg={'models':{'reviewer':{'model':'anthropic/claude-sonnet-5'}}}
    for phase in ('sources','answers','review'):
        folder=batch/phase; folder.mkdir(parents=True)
        (folder/'dataset.jsonl').write_text('{}\n')
        OmegaConf.save(OmegaConf.create(cfg),folder/'config.yaml')
        state={'run_dir':str(folder),'dataset_sha256':pub.sha(folder/'dataset.jsonl'),
               'config':cfg,'config_sha256':pub.sha(folder/'config.yaml')}
        if phase != 'review':
            state.update(origin='local_derivative_not_model_generation',paid_calls=0)
        elif not missing_real_config:
            (folder/'frozen_config.json').write_text(json.dumps(state))
        (folder/'status.json').write_text(json.dumps(state))
    (batch/'audit').mkdir()
    (batch/'audit/lineage.json').write_text('{"original_author": "Sonnet", "literal_corrections": true}')
    (batch/'audit/old_snapshot.jsonl').write_text('original bytes\n')
    monkeypatch.setattr(pub, 'final_answer_phase', lambda _: ('review',{},batch/'review/dataset.jsonl'))
    target=tmp_path/'snapshot/audit/batch08'
    if missing_real_config:
        with pytest.raises(ValueError,match='Missing real model frozen config'):
            pub.copy_batch_audit(batch,target)
        assert not target.exists()
    else:
        pub.copy_batch_audit(batch,target)
        assert (target/'audit/lineage.json').read_bytes() == (batch/'audit/lineage.json').read_bytes()
        assert (target/'audit/old_snapshot.jsonl').read_bytes() == (batch/'audit/old_snapshot.jsonl').read_bytes()
        assert (target/'sources/config.yaml').is_file()
        assert not (target/'sources/frozen_config.json').exists()
        assert (target/'review/frozen_config.json').is_file()
